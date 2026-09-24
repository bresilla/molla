use std::any::TypeId;

fn same<A: 'static, B: 'static>() {
    assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
}

#[test]
fn scheduler_is_available_through_facade_without_gpu() {
    use molla::sensors::{ScheduledSample, SensorSchedule, SensorScheduler, scheduler};
    same::<ScheduledSample, scheduler::ScheduledSample>();
    same::<SensorSchedule, scheduler::SensorSchedule>();
    same::<SensorScheduler, scheduler::SensorScheduler>();
}

#[test]
#[cfg(feature = "gpu")]
fn device_sensor_types_and_modules_are_exposed_by_facade() {
    use molla::sensors::*;
    same::<CameraBatch, camera_batch::CameraBatch>();
    same::<CameraBatchOutputPool, camera_batch::CameraBatchOutputPool>();
    same::<EncodedCameraBatch, camera_batch::EncodedCameraBatch>();
    same::<CameraBatchFrame<'static>, camera_batch::CameraBatchFrame<'static>>();
    same::<CameraBatchLimits, camera_batch::CameraBatchLimits>();
    same::<CameraBatchStamp, camera_batch::CameraBatchStamp>();
    same::<CameraBatchStats, camera_batch::CameraBatchStats>();
    same::<CameraImage<'static>, camera_batch::CameraImage<'static>>();
    same::<CameraImageLayout, camera_batch::CameraImageLayout>();
    same::<CameraSpec, camera_batch::CameraSpec>();
    same::<CameraDispatchStats, tiled_camera::CameraDispatchStats>();
    same::<EncodedCamera, tiled_camera::EncodedCamera>();
    same::<LidarBatch, lidar_batch::LidarBatch>();
    same::<EncodedLidarBatch, lidar_batch::EncodedLidarBatch>();
    same::<LidarBatchStats, lidar_batch::LidarBatchStats>();
    same::<LidarDeviceBatch<'static>, lidar_batch::LidarDeviceBatch<'static>>();
    same::<LidarPattern, lidar_batch::LidarPattern>();
    same::<LidarScan, lidar_batch::LidarScan>();
    same::<LidarScanRange, lidar_batch::LidarScanRange>();
    same::<RayDeviceHits<'static>, raycast::RayDeviceHits<'static>>();
    same::<RayQueryStats, raycast::RayQueryStats>();
    same::<EncodedRayQuery, raycast::EncodedRayQuery>();
    same::<ContactDeviceReadings<'static>, contact::ContactDeviceReadings<'static>>();
    same::<FrameDeviceReadings<'static>, frame_transform::FrameDeviceReadings<'static>>();
    same::<ImuDeviceReadings<'static>, imu::ImuDeviceReadings<'static>>();
    same::<EncodedImuSample, imu::EncodedImuSample>();
    same::<SceneStamp, scene::SceneStamp>();
    same::<SensorScene, scene::SensorScene>();
    same::<SensorSceneOptions, scene::SensorSceneOptions>();
    same::<UnsupportedGeometryPolicy, scene::UnsupportedGeometryPolicy>();
    same::<UnsupportedShape, scene::UnsupportedShape>();
    same::<DeviceCommandBatch, snapshot::DeviceCommandBatch>();
    same::<EncodedSensorFrame, device_results::EncodedSensorFrame>();
    same::<PendingSensorInputs<'static>, device_results::PendingSensorInputs<'static>>();
    same::<PendingSensorChannel<'static, u32>, device_results::PendingSensorChannel<'static, u32>>(
    );
    same::<SensorChannelSpec, device_results::SensorChannelSpec>();
    same::<SensorCommandBatch<'static>, device_results::SensorCommandBatch<'static>>();
    same::<SensorCommandStats, device_results::SensorCommandStats>();
    same::<SensorFrame, device_results::SensorFrame>();
    same::<SensorFrameChannel<'static, u32>, device_results::SensorFrameChannel<'static, u32>>();
    same::<SensorFramePool, device_results::SensorFramePool>();
    same::<SensorFramePoolConfig, device_results::SensorFramePoolConfig>();
    same::<SensorFramePoolStats, device_results::SensorFramePoolStats>();
    same::<SensorSubmission, device_results::SensorSubmission>();
    same::<DeviceSnapshot, snapshot::DeviceSnapshot>();
    same::<DeviceSnapshotConfig, snapshot::DeviceSnapshotConfig>();
    same::<DeviceSnapshotPool, snapshot::DeviceSnapshotPool>();
    same::<DeviceSnapshotStats, snapshot::DeviceSnapshotStats>();
    same::<SnapshotChannel<'static, u32>, snapshot::SnapshotChannel<'static, u32>>();
    same::<readback::ReadbackRing, molla_sensors::readback::ReadbackRing>();
    same::<SensorDispatchStats, molla_sensors::SensorDispatchStats>();
    same::<ImageProcessor, image_processing::ImageProcessor>();
    same::<EncodedProcessedImages, image_processing::EncodedProcessedImages>();
    same::<PendingImageJob<'static>, image_processing::PendingImageJob<'static>>();
    same::<AnnotationConfig, annotation::AnnotationConfig>();
    same::<AnnotationCounts, annotation::AnnotationCounts>();
    same::<AnnotationProcessor, annotation::AnnotationProcessor>();
    same::<EncodedAnnotations, annotation::EncodedAnnotations>();
    same::<PendingAnnotationJob<'static>, annotation::PendingAnnotationJob<'static>>();
    same::<AnnotationStats, annotation::AnnotationStats>();
    same::<AnnotationLayout, annotation::AnnotationLayout>();
    same::<ShapeLabel, annotation::ShapeLabel>();
    same::<DeviceBoundingBox, annotation::DeviceBoundingBox>();
    same::<AnnotationJob<'static>, annotation::AnnotationJob<'static>>();
    same::<AnnotationDeviceFrame<'static>, annotation::AnnotationDeviceFrame<'static>>();
    same::<ImageProcessorConfig, image_processing::ImageProcessorConfig>();
    same::<ImageOp, image_processing::ImageOp>();
    same::<ImageProcessingStats, image_processing::ImageProcessingStats>();
    same::<ImageJob<'static>, image_processing::ImageJob<'static>>();
    same::<ProcessedImages<'static>, image_processing::ProcessedImages<'static>>();
    same::<ProcessedImage<'static>, image_processing::ProcessedImage<'static>>();
    same::<ProcessedImageLayout, image_processing::ProcessedImageLayout>();
    same::<CloudProcessor, cloud_processing::CloudProcessor>();
    same::<EncodedCloud, cloud_processing::EncodedCloud>();
    same::<PendingCloudSource<'static>, cloud_processing::PendingCloudSource<'static>>();
    same::<PendingCloudInput<'static>, cloud_processing::PendingCloudInput<'static>>();
    same::<PendingCloudJob<'static>, cloud_processing::PendingCloudJob<'static>>();
    same::<CloudProcessorConfig, cloud_processing::CloudProcessorConfig>();
    same::<CloudProcessingStats, cloud_processing::CloudProcessingStats>();
    same::<CloudPoint, cloud_processing::CloudPoint>();
    same::<CloudReadLimits, cloud_export::CloudReadLimits>();
    same::<molla::compute::DeviceHealth, molla_compute::DeviceHealth>();
    same::<molla::compute::DeviceLoss, molla_compute::DeviceLoss>();
    same::<CloudGrid, cloud_grid::CloudGrid>();
    same::<EncodedCloudGrid, cloud_grid::EncodedCloudGrid>();
    same::<PendingCloudGridInput<'static>, cloud_grid::PendingCloudGridInput<'static>>();
    same::<CloudGridCell, cloud_grid::CloudGridCell>();
    same::<CloudGridConfig, cloud_grid::CloudGridConfig>();
    same::<CloudGridDiagnostics, cloud_grid::CloudGridDiagnostics>();
    same::<CloudGridSpec, cloud_grid::CloudGridSpec>();
    same::<CloudGridStats, cloud_grid::CloudGridStats>();
    same::<CloudGridFrame<'static>, cloud_grid::CloudGridFrame<'static>>();
    same::<CloudRaster, cloud_processing::CloudRaster>();
    same::<CloudSummary, cloud_processing::CloudSummary>();
    same::<CloudLayout, cloud_processing::CloudLayout>();
    same::<CloudFilter, cloud_processing::CloudFilter>();
    same::<CloudJob<'static>, cloud_processing::CloudJob<'static>>();
    same::<CloudInput<'static>, cloud_processing::CloudInput<'static>>();
    same::<CloudFrame<'static>, cloud_processing::CloudFrame<'static>>();
}
