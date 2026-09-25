use molla_compute::Engine;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{
    CloudFilter, CloudGrid, CloudGridCell, CloudGridConfig, CloudGridDiagnostics, CloudGridSpec,
    CloudPoint, CloudProcessor, CloudProcessorConfig, CloudSummary, EncodedCloud, EncodedCloudGrid,
    EncodedLidarBatch, PendingCloudGridInput, PendingCloudInput, PendingCloudJob,
    SensorCommandBatch, SensorFrame, SensorFramePool, SensorFramePoolConfig,
    readback::{ReadbackRing, ReadbackStamp},
};
use std::time::Duration;

const CELLS: u32 = 32 * 4 * 32;

pub struct CloudChain {
    cloud: CloudProcessor,
    grid: CloudGrid,
    cloud_pool: SensorFramePool,
    grid_pool: SensorFramePool,
    mode: &'static str,
}

impl CloudChain {
    pub fn new(engine: &Engine, rays: u32, mounts: u32, mode: &'static str) -> Self {
        assert!(["keep", "crop", "reject"].contains(&mode));
        let cloud = CloudProcessor::new(
            engine,
            CloudProcessorConfig {
                max_points: rays,
                max_jobs: mounts,
                max_buffer_bytes: 512 << 20,
            },
        )
        .unwrap();
        let grid = CloudGrid::new(
            engine,
            CloudGridConfig {
                max_points: rays,
                max_cells: CELLS,
                max_buffer_bytes: 256 << 20,
            },
        )
        .unwrap();
        let config = SensorFramePoolConfig {
            slots: 1,
            max_total_bytes: 256 << 20,
        };
        let cloud_pool = cloud.output_pool(engine, config).unwrap();
        let grid_pool = grid.output_pool(engine, config).unwrap();
        Self {
            cloud,
            grid,
            cloud_pool,
            grid_pool,
            mode,
        }
    }

    pub fn host_bytes(rays: u32, mounts: u32) -> u64 {
        u64::from(rays) * 116 + u64::from(1 + mounts * 5 + 6) * 4 + u64::from(CELLS) * 64
    }

    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        lidar: &EncodedLidarBatch,
        index: usize,
    ) -> (EncodedCloud, EncodedCloudGrid) {
        let filter = match self.mode {
            "keep" => CloudFilter::default(),
            "crop" => CloudFilter {
                bounds: Some((Vec3::new(0.25, -16.0, -128.0), Vec3::splat(128.0), true)),
                ..Default::default()
            },
            "reject" => CloudFilter {
                min_intensity: Some(2.0),
                ..Default::default()
            },
            _ => unreachable!(),
        };
        let stamp = ReadbackStamp {
            sample_id: index as u64,
            ..Default::default()
        };
        let jobs: Vec<_> = (0..lidar.scans.len())
            .map(|i| PendingCloudJob {
                input: PendingCloudInput::from_lidar(lidar, i).unwrap(),
                transform: Transform::new(Vec3::new(0.25, 0.5, 0.75), Quat::from_rotation_y(0.15)),
                filter,
                stamp: ReadbackStamp {
                    sensor_id: lidar.scans[i].scan.sensor_id,
                    ..stamp
                },
            })
            .collect();
        let cloud = self
            .cloud
            .encode_pending(batch, &jobs, &mut self.cloud_pool, stamp)
            .unwrap()
            .unwrap();
        let grid = self
            .grid
            .encode_pending(
                batch,
                PendingCloudGridInput {
                    frame: &cloud.frame,
                    channel: 0,
                    points: 0..lidar.ray_count,
                },
                CloudGridSpec {
                    minimum: [-128.0, -16.0, -128.0],
                    maximum: [128.0, 16.0, 128.0],
                    dimensions: [32, 4, 32],
                },
                &mut self.grid_pool,
                stamp,
            )
            .unwrap()
            .unwrap();
        (cloud, grid)
    }

    pub fn report(&self) {
        assert_eq!(self.cloud.stats().submissions, 0);
        assert_eq!(self.grid.stats().submissions, 0);
        eprintln!(
            "cloud={:?} grid={:?} cloud_pool={:?} grid_pool={:?}",
            self.cloud.stats(),
            self.grid.stats(),
            self.cloud_pool.stats(),
            self.grid_pool.stats()
        );
    }

    pub fn collect(
        ring: &mut ReadbackRing,
        cloud: &SensorFrame,
        grid: &SensorFrame,
        lidar: &EncodedLidarBatch,
        mode: &str,
        index: usize,
    ) {
        let points = cloud.channel::<CloudPoint>(0).unwrap();
        let counts = cloud.channel::<u32>(1).unwrap();
        let indices = grid.channel::<u32>(1).unwrap();
        let cells = grid.channel::<CloudGridCell>(2).unwrap();
        let diagnostics = grid.channel::<u32>(3).unwrap();
        let sources = [
            points.readback_source(0..points.len()).unwrap(),
            counts.readback_source(0..counts.len()).unwrap(),
            indices.readback_source(0..indices.len()).unwrap(),
            cells.readback_source(0..cells.len()).unwrap(),
            diagnostics.readback_source(0..diagnostics.len()).unwrap(),
        ];
        let ticket = ring.try_enqueue(&sources, cloud.stamp()).unwrap().unwrap();
        let collected = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
        let words = collected.channel::<u32>(1).unwrap();
        let kept = words[0];
        assert!(kept as usize <= lidar.ray_count);
        match mode {
            "keep" => assert_eq!(kept as usize, lidar.ray_count),
            "reject" => assert_eq!(kept, 0),
            "crop" => assert!(kept > 0 && (kept as usize) < lidar.ray_count),
            _ => unreachable!(),
        }
        let mut offset = 0;
        for (words, scan) in words[1..].chunks_exact(5).zip(&lidar.scans) {
            let summary = CloudSummary::from_words(words).unwrap();
            assert_eq!(summary.start, offset);
            assert_eq!(summary.misses, 0);
            assert_eq!(summary.invalid, 0);
            assert_eq!(summary.count + summary.filtered, scan.ray_count);
            offset += summary.count;
        }
        assert_eq!(offset, kept);
        let diagnostics =
            CloudGridDiagnostics::from_words(&collected.channel::<u32>(4).unwrap()).unwrap();
        assert_eq!(diagnostics.accepted, kept);
        assert_eq!(diagnostics.misses as usize, lidar.ray_count - kept as usize);
        assert_eq!(
            (
                diagnostics.invalid,
                diagnostics.outside,
                diagnostics.index_overflow,
                diagnostics.numeric_overflow_cells
            ),
            (0, 0, 0, 0)
        );
        let mut count = 0;
        for cell in collected.channel_iter::<CloudGridCell>(3).unwrap() {
            count += cell.header[1];
            assert_eq!(cell.header[2], 0);
        }
        assert_eq!(count, kept);
        for i in collected
            .channel_iter::<u32>(2)
            .unwrap()
            .take(kept as usize)
        {
            assert!(i < kept);
        }
        for point in collected
            .channel_iter::<CloudPoint>(0)
            .unwrap()
            .take(kept as usize)
        {
            assert!(point.position[..3].iter().all(|v| v.is_finite()));
            assert!(point.normal[3] > 0.0 && point.normal[3] < 30.0);
            assert_eq!(point.sample_id(), index as u64);
            assert!((point.sensor_id() as usize) < lidar.scans.len());
            assert_ne!(point.extra[2] & 1, 0);
            assert!((point.grid[3] as usize) < lidar.ray_count);
        }
        std::hint::black_box(collected);
    }
}
