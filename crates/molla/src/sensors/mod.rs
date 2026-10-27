//! Public sensor APIs from `molla-sensors`.
//!
//! GPU-gated sensors include scene queries, LiDAR, cameras, IMU, contacts and
//! frame transforms, with optional readback and copied device snapshots.
//! Simulation-time scheduling and finite-difference ray adjoints also compile
//! without GPU support. Images and point-cloud transforms/filters have GPU paths;
//! CPU convenience processing and filesystem serialization remain available.

#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub mod camera_post {
    pub use molla_sensors::camera_post::*;
}
#[cfg(feature = "gpu")]
pub mod dataset {
    pub use molla_sensors::dataset::*;
}
#[cfg(feature = "gpu")]
pub mod lidar {
    pub use molla_sensors::lidar::*;
}
#[cfg(feature = "gpu")]
pub mod pointcloud {
    pub use molla_sensors::pointcloud::*;
}
#[cfg(feature = "gpu")]
pub mod raycast;
pub mod raycast_adjoint {
    pub use molla_sensors::raycast_adjoint::*;
}
#[cfg(feature = "gpu")]
pub mod tiled_camera;

#[cfg(feature = "gpu")]
pub use molla_sensors::{CloudReadLimits, cloud_export};

#[cfg(feature = "gpu")]
pub use molla_sensors::{
    AnnotationConfig, AnnotationCounts, AnnotationDeviceFrame, AnnotationJob, AnnotationLayout,
    AnnotationProcessor, AnnotationStats, DeviceBoundingBox, EncodedAnnotations,
    PendingAnnotationJob, ShapeLabel, annotation,
};

#[cfg(feature = "gpu")]
pub use molla_sensors::{
    CloudGrid, CloudGridCell, CloudGridConfig, CloudGridDiagnostics, CloudGridFrame, CloudGridSpec,
    CloudGridStats, EncodedCloudGrid, PendingCloudGridInput, cloud_grid,
};

#[cfg(feature = "gpu")]
pub use molla_sensors::{
    CloudFilter, CloudFrame, CloudInput, CloudJob, CloudLayout, CloudPoint, CloudProcessingStats,
    CloudProcessor, CloudProcessorConfig, CloudRaster, CloudSummary, EncodedCloud,
    PendingCloudInput, PendingCloudJob, PendingCloudSource, cloud_processing,
};
#[cfg(feature = "gpu")]
pub use molla_sensors::{
    EncodedProcessedImages, ImageJob, ImageOp, ImageProcessingStats, ImageProcessor,
    ImageProcessorConfig, PendingImageJob, ProcessedImage, ProcessedImageLayout, ProcessedImages,
};
#[cfg(feature = "gpu")]
pub use molla_sensors::{
    EncodedSensorFrame, PendingSensorChannel, PendingSensorInputs, SensorChannelSpec,
    SensorCommandBatch, SensorCommandStats, SensorFrame, SensorFrameChannel, SensorFramePool,
    SensorFramePoolConfig, SensorFramePoolStats, SensorSubmission, device_results,
};
#[cfg(feature = "gpu")]
pub use molla_sensors::{INVALID_SCENE_DISTANCE, INVALID_SCENE_INDEX, INVALID_SCENE_SHAPE_ID};
#[cfg(feature = "gpu")]
pub use molla_sensors::{
    SceneAcceleration, SceneAccelerationConfig, SceneAccelerationStats, SceneAccelerationStorage,
    SceneAccelerationView, SceneBvhNode, SceneRebuildPolicy, SceneRebuildStorage,
    scene_acceleration,
};
pub use molla_sensors::{ScheduledSample, SensorSchedule, SensorScheduler, scheduler};
#[cfg(feature = "gpu")]
pub use molla_sensors::{camera_batch, image_processing, lidar_batch, readback, scene, snapshot};

#[cfg(feature = "gpu")]
pub use molla_sensors::{
    CameraBatch, CameraBatchFrame, CameraBatchLimits, CameraBatchOutputPool, CameraBatchStamp,
    CameraBatchStats, CameraDispatchStats, CameraImage, CameraImageLayout, CameraSpec,
    ContactDeviceReadings, DeviceCommandBatch, DeviceSnapshot, DeviceSnapshotConfig,
    DeviceSnapshotPool, DeviceSnapshotStats, EncodedCameraBatch, EncodedLidarBatch,
    FrameDeviceReadings, ImuDeviceReadings, LidarBatch, LidarBatchStats, LidarDeviceBatch,
    LidarPattern, LidarScan, LidarScanRange, RayDeviceHits, RayQueryStats, SceneIdentitySnapshot,
    SceneStamp, SensorBodyHandle, SensorDispatchStats, SensorScene, SensorSceneOptions,
    SensorSceneRemap, SensorShapeHandle, SnapshotChannel, UnsupportedGeometryPolicy,
    UnsupportedShape,
};

#[cfg(feature = "gpu")]
pub use contact::{BodyContactReading, ContactSensor};
#[cfg(feature = "gpu")]
pub use frame_transform::{FrameReading, FrameTarget, FrameTransformSensor};
#[cfg(feature = "gpu")]
pub use imu::{EncodedImuSample, ImuMount, ImuReading, ImuSensor};
#[cfg(feature = "gpu")]
pub use molla_sensors::{
    AerialLidar, AerialScan, AerialScanMetadata, AnnotationFrame, BoundingBox2D, Compare, Dataset,
    GridCell, HitPoint, Lidar, MISS_SHAPE_ID, Scan, ScanMetadata, bin_into_grid, distance_filter,
    read_xyz_rgb, reflectance_filter, rotate_cloud, scalar_filter, transform_cloud,
    translate_cloud, write_ply_ascii, write_ptx, write_xyz_rgb, xyz_filter,
};
#[cfg(feature = "gpu")]
pub use molla_sensors::{CameraStorageUsage, TiledCameraLimits, camera_storage};
pub use molla_sensors::{
    DiffRay, DiffSdfShape, RayScene, compute_grad_raycast, compute_loss_l2, forward_with_loss,
};
#[cfg(feature = "gpu")]
pub use molla_sensors::{LidarBatchLimits, LidarBatchStorage, LidarLayoutLimits, lidar_storage};
#[cfg(feature = "gpu")]
pub use molla_sensors::{ProcessorStorageLimits, ProcessorStorageUsage, processor_storage};
#[cfg(feature = "gpu")]
pub use molla_sensors::{SensorArrayLimits, SensorArrayStorage, sensor_storage};
#[cfg(feature = "gpu")]
pub use raycast::{EncodedRayQuery, RayHit, RayStorageUsage, Raycaster, RaycasterLimits};
#[cfg(feature = "gpu")]
pub use tiled_camera::{
    CameraChannels, EncodedCamera, PhysicalOptics, TiledCamera, TiledCameraConfig,
};
