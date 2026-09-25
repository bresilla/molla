//! Sensor models and sensor adjoints.

#[cfg(feature = "gpu")]
pub mod annotation;
#[cfg(feature = "gpu")]
pub use annotation::{
    AnnotationConfig, AnnotationCounts, AnnotationDeviceFrame, AnnotationJob, AnnotationLayout,
    AnnotationProcessor, AnnotationStats, DeviceBoundingBox, EncodedAnnotations,
    PendingAnnotationJob, ShapeLabel,
};

#[cfg(feature = "gpu")]
pub mod camera_batch;
#[cfg(feature = "gpu")]
pub use camera_batch::{
    CameraBatch, CameraBatchFrame, CameraBatchLimits, CameraBatchOutputPool, CameraBatchStamp,
    CameraBatchStats, CameraImage, CameraImageLayout, CameraSpec, EncodedCameraBatch,
};

#[cfg(feature = "gpu")]
pub mod camera_post;
#[cfg(feature = "gpu")]
pub mod cloud_export;
#[cfg(feature = "gpu")]
pub use cloud_export::CloudReadLimits;
#[cfg(feature = "gpu")]
pub mod cloud_grid;
#[cfg(feature = "gpu")]
pub use cloud_grid::{
    CloudGrid, CloudGridCell, CloudGridConfig, CloudGridDiagnostics, CloudGridFrame, CloudGridSpec,
    CloudGridStats, EncodedCloudGrid, PendingCloudGridInput,
};
#[cfg(feature = "gpu")]
pub mod cloud_processing;
#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod dataset;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod image_processing;
#[cfg(feature = "gpu")]
pub use cloud_processing::{
    CloudFilter, CloudFrame, CloudInput, CloudJob, CloudLayout, CloudPoint, CloudProcessingStats,
    CloudProcessor, CloudProcessorConfig, CloudRaster, CloudSummary, EncodedCloud,
    PendingCloudInput, PendingCloudJob, PendingCloudSource,
};
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub use image_processing::{
    EncodedProcessedImages, ImageJob, ImageOp, ImageProcessingStats, ImageProcessor,
    ImageProcessorConfig, PendingImageJob, ProcessedImage, ProcessedImageLayout, ProcessedImages,
};
#[cfg(feature = "gpu")]
pub mod device_results;
#[cfg(feature = "gpu")]
pub mod lidar;
#[cfg(feature = "gpu")]
pub mod lidar_batch;
#[cfg(feature = "gpu")]
pub mod lidar_storage;
#[cfg(feature = "gpu")]
pub use lidar_storage::{LidarBatchLimits, LidarBatchStorage, LidarLayoutLimits};
#[cfg(feature = "gpu")]
pub mod pointcloud;
#[cfg(feature = "gpu")]
pub mod raycast;
pub mod raycast_adjoint;
#[cfg(feature = "gpu")]
pub mod readback;
#[cfg(feature = "gpu")]
pub mod scene;
#[cfg(feature = "gpu")]
pub mod scene_acceleration;
#[cfg(feature = "gpu")]
mod scene_geometry;
#[cfg(feature = "gpu")]
mod scene_identity;
#[cfg(feature = "gpu")]
mod scene_rebuild;
#[cfg(feature = "gpu")]
mod scene_threading;
#[cfg(feature = "gpu")]
mod scene_trace;
#[cfg(feature = "gpu")]
pub use scene_acceleration::{
    SceneAcceleration, SceneAccelerationConfig, SceneAccelerationStats, SceneAccelerationStorage,
    SceneAccelerationView, SceneBvhNode,
};
#[cfg(feature = "gpu")]
pub use scene_rebuild::{SceneRebuildPolicy, SceneRebuildStorage};
pub mod scheduler;
#[cfg(feature = "gpu")]
pub mod sensor_storage;
#[cfg(feature = "gpu")]
pub use sensor_storage::{SensorArrayLimits, SensorArrayStorage};
#[cfg(feature = "gpu")]
mod host_arrays;
#[cfg(feature = "gpu")]
mod plan_scratch;
#[cfg(feature = "gpu")]
pub mod processor_storage;
#[cfg(feature = "gpu")]
pub use processor_storage::{ProcessorStorageLimits, ProcessorStorageUsage};
#[cfg(feature = "gpu")]
pub mod snapshot;
#[cfg(feature = "gpu")]
pub use device_results::{
    EncodedSensorFrame, PendingSensorChannel, PendingSensorInputs, SensorChannelSpec,
    SensorCommandBatch, SensorCommandStats, SensorFrame, SensorFrameChannel, SensorFramePool,
    SensorFramePoolConfig, SensorFramePoolStats, SensorSubmission,
};
#[cfg(feature = "gpu")]
pub mod camera_storage;
#[cfg(feature = "gpu")]
pub mod tiled_camera;
#[cfg(feature = "gpu")]
pub use camera_storage::{CameraStorageUsage, TiledCameraLimits};

#[cfg(feature = "gpu")]
pub use camera_post::{
    apply_brightness_contrast, apply_exposure, apply_gamma, apply_gamma_srgb, apply_gray_world_wb,
    apply_reinhard_tone_map, apply_reinhard_white_point, apply_saturation, apply_srgb_to_linear,
    apply_white_patch_wb, linear_to_srgb, pack_rgba8, srgb_to_linear, unpack_rgba8,
};
#[cfg(feature = "gpu")]
pub use contact::{BodyContactReading, ContactDeviceReadings, ContactSensor};
#[cfg(feature = "gpu")]
pub use dataset::{AnnotationFrame, BoundingBox2D, Dataset, MISS_SHAPE_ID};
#[cfg(feature = "gpu")]
pub use frame_transform::{FrameDeviceReadings, FrameReading, FrameTarget, FrameTransformSensor};
#[cfg(feature = "gpu")]
pub use imu::{EncodedImuSample, ImuDeviceReadings, ImuMount, ImuReading, ImuSensor};
#[cfg(feature = "gpu")]
pub use lidar::{AerialLidar, AerialScan, AerialScanMetadata, Lidar, Scan, ScanMetadata};
#[cfg(feature = "gpu")]
pub use lidar_batch::{
    EncodedLidarBatch, LidarBatch, LidarBatchStats, LidarDeviceBatch, LidarPattern, LidarScan,
    LidarScanRange,
};
#[cfg(feature = "gpu")]
pub use pointcloud::{
    Compare, GridCell, HitPoint, bin_into_grid, distance_filter, read_xyz_rgb, reflectance_filter,
    rotate_cloud, scalar_filter, transform_cloud, translate_cloud, write_ply_ascii, write_ptx,
    write_xyz_rgb, xyz_filter,
};
#[cfg(feature = "gpu")]
pub use raycast::{
    EncodedRayQuery, RayDeviceHits, RayHit, RayQueryStats, RayStorageUsage, Raycaster,
    RaycasterLimits,
};
#[cfg(feature = "gpu")]
mod dispatch;
#[cfg(feature = "gpu")]
pub use dispatch::SensorDispatchStats;
pub use raycast_adjoint::{
    Ray as DiffRay, RayScene, SdfShape as DiffSdfShape, compute_grad_raycast, compute_loss_l2,
    forward_with_loss,
};
#[cfg(feature = "gpu")]
pub use scene::{
    INVALID_SCENE_DISTANCE, INVALID_SCENE_INDEX, INVALID_SCENE_SHAPE_ID, SceneIdentitySnapshot,
    SceneStamp, SensorBodyHandle, SensorScene, SensorSceneOptions, SensorSceneRemap,
    SensorShapeHandle, UnsupportedGeometryPolicy, UnsupportedShape,
};
pub use scheduler::{ScheduledSample, SensorSchedule, SensorScheduler};
#[cfg(feature = "gpu")]
pub use snapshot::{
    DeviceCommandBatch, DeviceSnapshot, DeviceSnapshotConfig, DeviceSnapshotPool,
    DeviceSnapshotStats, SnapshotChannel,
};
#[cfg(feature = "gpu")]
pub use tiled_camera::{
    CameraChannels, CameraDispatchStats, EncodedCamera, PhysicalOptics, TiledCamera,
    TiledCameraConfig,
};
