use image::GenericImageView;
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr;

type FsbHandle = *mut c_void;
type FsbCreate = unsafe extern "C" fn() -> FsbHandle;
type FsbDestroy = unsafe extern "C" fn(FsbHandle);
type FsbLoad = unsafe extern "C" fn(FsbHandle, *const FsbConfig) -> c_int;
type FsbLoadEx = unsafe extern "C" fn(FsbHandle, *const FsbConfigEx) -> c_int;
type FsbResultVersion = unsafe extern "C" fn() -> c_int;
type FsbIsLoaded = unsafe extern "C" fn(FsbHandle) -> c_int;
type FsbLastError = unsafe extern "C" fn(FsbHandle) -> *const c_char;
type FsbProcessBgr =
    unsafe extern "C" fn(FsbHandle, *const u8, c_int, c_int, *mut FsbResult, c_int) -> c_int;
type FsbProcessBgrMesh = unsafe extern "C" fn(
    FsbHandle,
    *const u8,
    c_int,
    c_int,
    *mut FsbResult,
    *mut c_int,
    *mut f32,
    c_int,
    c_int,
) -> c_int;
type FsbProcessImageBvh = unsafe extern "C" fn(
    FsbHandle,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    c_int,
) -> c_int;
type FsbVideoProgressCallback = unsafe extern "C" fn(c_int, c_int, c_int, f32, *mut c_void);
type FsbVideoMeshCallback = unsafe extern "C" fn(c_int, c_int, c_int, *const f32, *mut c_void);
type FsbVideoMeshExCallback = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut c_void,
);
type FsbVideoMeshTrackedCallback = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut c_void,
);
type FsbVideoMeshOverlayCallback = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut c_void,
);
type FsbProcessVideo = unsafe extern "C" fn(
    FsbHandle,
    *const FsbVideoConfig,
    *mut FsbVideoSummary,
    Option<FsbVideoProgressCallback>,
    *mut c_void,
) -> c_int;
type FsbProcessVideoMesh = unsafe extern "C" fn(
    FsbHandle,
    *const FsbVideoConfig,
    *mut FsbVideoSummary,
    Option<FsbVideoProgressCallback>,
    Option<FsbVideoMeshCallback>,
    *mut c_void,
) -> c_int;
type FsbProcessVideoMeshEx = unsafe extern "C" fn(
    FsbHandle,
    *const FsbVideoConfig,
    *mut FsbVideoSummary,
    Option<FsbVideoProgressCallback>,
    Option<FsbVideoMeshExCallback>,
    *mut c_void,
) -> c_int;
type FsbProcessVideoMeshTracked = unsafe extern "C" fn(
    FsbHandle,
    *const FsbVideoConfig,
    *mut FsbVideoSummary,
    Option<FsbVideoProgressCallback>,
    Option<FsbVideoMeshTrackedCallback>,
    *mut c_void,
) -> c_int;
type FsbProcessVideoMeshOverlay = unsafe extern "C" fn(
    FsbHandle,
    *const FsbVideoConfig,
    *mut FsbVideoSummary,
    Option<FsbVideoProgressCallback>,
    Option<FsbVideoMeshOverlayCallback>,
    *mut c_void,
) -> c_int;

const FSB_MESH_VERTEX_COUNT: usize = 18_439;
const FSB_MESH_FLOAT_COUNT: usize = FSB_MESH_VERTEX_COUNT * 3;

#[repr(C)]
struct FsbConfig {
    onnx_dir: *const c_char,
    gguf_path: *const c_char,
    yolo_path: *const c_char,
    cuda_device: c_int,
    skip_body_model: c_int,
    person_thresh: f32,
    person_nms_iou: f32,
    max_persons: c_int,
    focal_x: f32,
    focal_y: f32,
    principal_x: f32,
    principal_y: f32,
    zero_face_params: c_int,
}

#[repr(C)]
struct FsbConfigEx {
    struct_size: u32,
    base: FsbConfig,
    backbone_name: *const c_char,
    use_trt_ep: c_int,
    use_fp16: c_int,
}

#[repr(C)]
struct FsbVideoConfig {
    input_path: *const c_char,
    bvh_path: *const c_char,
    bvh_template: *const c_char,
    lbs_path: *const c_char,
    csv_path: *const c_char,
    max_frames: c_int,
    frame_stride: c_int,
    max_results: c_int,
    butterworth: c_int,
    bw_cutoff: f32,
    butterworth_root_rotation: c_int,
    rot_clamp_deg: f32,
}

#[repr(C)]
#[derive(Default)]
struct FsbVideoSummary {
    frames_read: c_int,
    frames_processed: c_int,
    persons_total: c_int,
    source_fps: f64,
    total_ms: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FsbResult {
    bbox: [f32; 4],
    focal_length: f32,
    pred_cam_t: [f32; 3],
    global_rot: [f32; 3],
    body_pose: [f32; 133],
    shape: [f32; 45],
    scale: [f32; 28],
    hand_pose: [f32; 108],
    face_params: [f32; 72],
    yolo_kps: [f32; 51],
    has_yolo_kps: c_int,
    kps_3d: [f32; 210],
    kps_2d: [f32; 140],
    has_kps: c_int,
    pred_pose_raw: [f32; 266],
    pred_cam_raw: [f32; 3],
    mhr_model_params: [f32; 204],
}

impl Default for FsbResult {
    fn default() -> Self {
        // FsbResult is a C float/int POD mirror. Zero initialization is valid.
        unsafe { std::mem::zeroed() }
    }
}

struct FsbApi {
    _library: Library,
    create: FsbCreate,
    destroy: FsbDestroy,
    _load: FsbLoad,
    load_ex: FsbLoadEx,
    result_version: FsbResultVersion,
    is_loaded: FsbIsLoaded,
    last_error: FsbLastError,
    process_bgr: FsbProcessBgr,
    process_bgr_mesh: Option<FsbProcessBgrMesh>,
    process_image_bvh: Option<FsbProcessImageBvh>,
    process_video: FsbProcessVideo,
    process_video_mesh: Option<FsbProcessVideoMesh>,
    process_video_mesh_ex: Option<FsbProcessVideoMeshEx>,
    process_video_mesh_tracked: Option<FsbProcessVideoMeshTracked>,
    process_video_mesh_overlay: Option<FsbProcessVideoMeshOverlay>,
}

pub struct Engine {
    api: FsbApi,
    handle: FsbHandle,
}

unsafe impl Send for Engine {}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadEngineRequest {
    pub library_path: String,
    pub onnx_dir: String,
    pub gguf_path: String,
    pub yolo_path: String,
    pub backbone_name: String,
    pub cuda_device: i32,
    pub use_trt_ep: bool,
    pub use_fp16: bool,
    pub skip_body_model: bool,
    pub max_persons: i32,
    pub person_thresh: f32,
    pub person_nms_iou: f32,
}

#[derive(Debug, Serialize)]
pub struct LoadEngineResponse {
    pub loaded: bool,
    pub result_version: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessImageRequest {
    pub image_path: String,
    pub max_results: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessImageBvhRequest {
    pub image_path: String,
    pub bvh_path: String,
    pub bvh_template_path: String,
    pub lbs_path: String,
    pub max_results: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessVideoRequest {
    pub video_path: String,
    pub bvh_path: String,
    pub bvh_template_path: String,
    pub lbs_path: String,
    pub csv_path: String,
    pub max_frames: i32,
    pub frame_stride: i32,
    pub max_results: i32,
    pub butterworth: bool,
    pub bw_cutoff: f32,
    pub butterworth_root_rotation: bool,
    pub rot_clamp_deg: f32,
}

#[derive(Debug, Serialize)]
pub struct ProcessImageResponse {
    pub width: u32,
    pub height: u32,
    pub persons: Vec<PersonResult>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessVideoResponse {
    pub frames_read: i32,
    pub frames_processed: i32,
    pub persons_total: i32,
    pub mesh_previews: i32,
    pub source_fps: f64,
    pub total_ms: f64,
    pub bvh_path: String,
    pub csv_path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoProgress {
    pub frame_index: i32,
    pub total_frames: i32,
    pub persons: i32,
    pub frame_ms: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoMeshPreview {
    pub frame_index: i32,
    pub person_index: i32,
    pub track_id: Option<i32>,
    pub frame_width: Option<i32>,
    pub frame_height: Option<i32>,
    pub bbox: Option<[f32; 4]>,
    pub pred_cam_t: Option<[f32; 3]>,
    pub kps3d: Option<Vec<[f32; 3]>>,
    pub kps2d: Option<Vec<[f32; 2]>>,
    pub mesh_vertices: Vec<[f32; 3]>,
}

#[derive(Debug, Serialize)]
pub struct PersonResult {
    pub bbox: [f32; 4],
    pub focal_length: f32,
    pub pred_cam_t: [f32; 3],
    pub global_rot: [f32; 3],
    pub yolo_kps: Option<Vec<[f32; 3]>>,
    pub kps_2d: Option<Vec<[f32; 2]>>,
    pub kps_3d: Option<Vec<[f32; 3]>>,
    pub mesh_vertices: Option<Vec<[f32; 3]>>,
}

impl Engine {
    pub fn load(request: LoadEngineRequest) -> Result<Self, String> {
        let api = unsafe { FsbApi::load(&request.library_path)? };
        let handle = unsafe { (api.create)() };
        if handle.is_null() {
            return Err("fsb_create returned null".to_string());
        }

        let mut engine = Self { api, handle };
        engine.load_pipeline(request)?;
        Ok(engine)
    }

    pub fn is_loaded(&self) -> bool {
        unsafe { (self.api.is_loaded)(self.handle) != 0 }
    }

    pub fn result_version(&self) -> i32 {
        unsafe { (self.api.result_version)() as i32 }
    }

    pub fn process_image(
        &self,
        request: ProcessImageRequest,
    ) -> Result<ProcessImageResponse, String> {
        let image_path = Path::new(&request.image_path);
        let image = image::open(image_path)
            .map_err(|err| format!("Could not open image '{}': {err}", image_path.display()))?;
        let (width, height) = image.dimensions();
        let rgb = image.to_rgb8();

        let mut bgr = Vec::with_capacity((width * height * 3) as usize);
        for pixel in rgb.pixels() {
            bgr.push(pixel[2]);
            bgr.push(pixel[1]);
            bgr.push(pixel[0]);
        }

        let max_results = request.max_results.clamp(1, 64);
        if let Some(process_bgr_mesh) = self.api.process_bgr_mesh {
            let mut results = vec![FsbResult::default(); max_results];
            let mut vertex_counts = vec![0 as c_int; max_results];
            let mut vertices = vec![0.0f32; max_results * FSB_MESH_FLOAT_COUNT];
            let count = unsafe {
                process_bgr_mesh(
                    self.handle,
                    bgr.as_ptr(),
                    width as c_int,
                    height as c_int,
                    results.as_mut_ptr(),
                    vertex_counts.as_mut_ptr(),
                    vertices.as_mut_ptr(),
                    max_results as c_int,
                    FSB_MESH_FLOAT_COUNT as c_int,
                )
            };

            if count < 0 {
                return Err(self
                    .last_error()
                    .unwrap_or_else(|| "fsb_process_bgr_mesh failed".to_string()));
            }

            let persons = results
                .iter()
                .zip(vertex_counts.iter())
                .zip(vertices.chunks_exact(FSB_MESH_FLOAT_COUNT))
                .take(count as usize)
                .map(|((result, vertex_count), vertex_slice)| {
                    PersonResult::from_mesh(result, *vertex_count, vertex_slice)
                })
                .collect();

            return Ok(ProcessImageResponse {
                width,
                height,
                persons,
            });
        }

        let mut results = vec![FsbResult::default(); max_results];
        let count = unsafe {
            (self.api.process_bgr)(
                self.handle,
                bgr.as_ptr(),
                width as c_int,
                height as c_int,
                results.as_mut_ptr(),
                max_results as c_int,
            )
        };

        if count < 0 {
            return Err(self
                .last_error()
                .unwrap_or_else(|| "fsb_process_bgr failed".to_string()));
        }

        let persons = results
            .iter()
            .take(count as usize)
            .map(PersonResult::from)
            .collect();

        Ok(ProcessImageResponse {
            width,
            height,
            persons,
        })
    }

    pub fn process_image_bvh(&self, request: ProcessImageBvhRequest) -> Result<i32, String> {
        let Some(process_image_bvh) = self.api.process_image_bvh else {
            return Err("The loaded engine DLL does not expose image BVH export. Rebuild or select the bundled DLL.".to_string());
        };

        let image_path = CString::new(request.image_path.clone())
            .map_err(|_| "Image path contains NUL".to_string())?;
        let bvh_path = CString::new(request.bvh_path.clone())
            .map_err(|_| "BVH path contains NUL".to_string())?;
        let bvh_template = CString::new(request.bvh_template_path.clone())
            .map_err(|_| "BVH template path contains NUL".to_string())?;
        let lbs_path = CString::new(request.lbs_path.clone())
            .map_err(|_| "LBS path contains NUL".to_string())?;

        let count = unsafe {
            process_image_bvh(
                self.handle,
                image_path.as_ptr(),
                bvh_path.as_ptr(),
                bvh_template.as_ptr(),
                lbs_path.as_ptr(),
                request.max_results.clamp(1, 64),
            )
        };
        if count < 0 {
            return Err(self
                .last_error()
                .unwrap_or_else(|| "fsb_process_image_bvh failed".to_string()));
        }
        Ok(count as i32)
    }

    pub fn process_video<F, M>(
        &self,
        request: ProcessVideoRequest,
        mut progress: F,
        mut mesh_preview: M,
    ) -> Result<ProcessVideoResponse, String>
    where
        F: FnMut(VideoProgress),
        M: FnMut(VideoMeshPreview),
    {
        let input_path = CString::new(request.video_path.clone())
            .map_err(|_| "Video path contains NUL".to_string())?;
        let bvh_path = CString::new(request.bvh_path.clone())
            .map_err(|_| "BVH path contains NUL".to_string())?;
        let bvh_template = CString::new(request.bvh_template_path.clone())
            .map_err(|_| "BVH template path contains NUL".to_string())?;
        let lbs_path = CString::new(request.lbs_path.clone())
            .map_err(|_| "LBS path contains NUL".to_string())?;
        let csv_path = CString::new(request.csv_path.clone())
            .map_err(|_| "CSV path contains NUL".to_string())?;

        let config = FsbVideoConfig {
            input_path: input_path.as_ptr(),
            bvh_path: bvh_path.as_ptr(),
            bvh_template: bvh_template.as_ptr(),
            lbs_path: lbs_path.as_ptr(),
            csv_path: csv_path.as_ptr(),
            max_frames: request.max_frames.max(0),
            frame_stride: request.frame_stride.max(1),
            max_results: request.max_results.clamp(1, 64),
            butterworth: request.butterworth as c_int,
            bw_cutoff: request.bw_cutoff.max(0.0),
            butterworth_root_rotation: request.butterworth_root_rotation as c_int,
            rot_clamp_deg: request.rot_clamp_deg.max(0.0),
        };
        let mut summary = FsbVideoSummary::default();
        let mut mesh_previews = 0;
        let ok = {
            let mut bridge = VideoCallbackBridge {
                progress_callback: &mut progress,
                mesh_callback: &mut mesh_preview,
                mesh_previews: &mut mesh_previews,
            };

            if let Some(process_video_mesh_overlay) = self.api.process_video_mesh_overlay {
                unsafe {
                    process_video_mesh_overlay(
                        self.handle,
                        &config,
                        &mut summary,
                        Some(video_progress_trampoline::<F, M>),
                        Some(video_mesh_overlay_trampoline::<F, M>),
                        &mut bridge as *mut VideoCallbackBridge<'_, F, M> as *mut c_void,
                    )
                }
            } else if let Some(process_video_mesh_tracked) = self.api.process_video_mesh_tracked {
                unsafe {
                    process_video_mesh_tracked(
                        self.handle,
                        &config,
                        &mut summary,
                        Some(video_progress_trampoline::<F, M>),
                        Some(video_mesh_tracked_trampoline::<F, M>),
                        &mut bridge as *mut VideoCallbackBridge<'_, F, M> as *mut c_void,
                    )
                }
            } else if let Some(process_video_mesh_ex) = self.api.process_video_mesh_ex {
                unsafe {
                    process_video_mesh_ex(
                        self.handle,
                        &config,
                        &mut summary,
                        Some(video_progress_trampoline::<F, M>),
                        Some(video_mesh_ex_trampoline::<F, M>),
                        &mut bridge as *mut VideoCallbackBridge<'_, F, M> as *mut c_void,
                    )
                }
            } else if let Some(process_video_mesh) = self.api.process_video_mesh {
                unsafe {
                    process_video_mesh(
                        self.handle,
                        &config,
                        &mut summary,
                        Some(video_progress_trampoline::<F, M>),
                        Some(video_mesh_trampoline::<F, M>),
                        &mut bridge as *mut VideoCallbackBridge<'_, F, M> as *mut c_void,
                    )
                }
            } else {
                unsafe {
                    (self.api.process_video)(
                        self.handle,
                        &config,
                        &mut summary,
                        Some(video_progress_trampoline::<F, M>),
                        &mut bridge as *mut VideoCallbackBridge<'_, F, M> as *mut c_void,
                    )
                }
            }
        };

        if ok == 0 {
            return Err(self
                .last_error()
                .unwrap_or_else(|| "fsb_process_video failed".to_string()));
        }

        Ok(ProcessVideoResponse {
            frames_read: summary.frames_read,
            frames_processed: summary.frames_processed,
            persons_total: summary.persons_total,
            mesh_previews,
            source_fps: summary.source_fps,
            total_ms: summary.total_ms,
            bvh_path: request.bvh_path,
            csv_path: request.csv_path,
        })
    }

    fn load_pipeline(&mut self, request: LoadEngineRequest) -> Result<(), String> {
        let onnx_dir =
            CString::new(request.onnx_dir).map_err(|_| "ONNX path contains NUL".to_string())?;
        let gguf_path =
            CString::new(request.gguf_path).map_err(|_| "GGUF path contains NUL".to_string())?;
        let yolo_path =
            CString::new(request.yolo_path).map_err(|_| "YOLO path contains NUL".to_string())?;
        let backbone_name = CString::new(request.backbone_name)
            .map_err(|_| "Backbone filename contains NUL".to_string())?;

        let base = FsbConfig {
            onnx_dir: onnx_dir.as_ptr(),
            gguf_path: gguf_path.as_ptr(),
            yolo_path: yolo_path.as_ptr(),
            cuda_device: request.cuda_device,
            skip_body_model: request.skip_body_model as c_int,
            person_thresh: request.person_thresh,
            person_nms_iou: request.person_nms_iou,
            max_persons: request.max_persons,
            focal_x: 0.0,
            focal_y: 0.0,
            principal_x: 0.0,
            principal_y: 0.0,
            zero_face_params: 1,
        };
        let config = FsbConfigEx {
            struct_size: std::mem::size_of::<FsbConfigEx>() as u32,
            base,
            backbone_name: backbone_name.as_ptr(),
            use_trt_ep: request.use_trt_ep as c_int,
            use_fp16: request.use_fp16 as c_int,
        };

        let ok = unsafe { (self.api.load_ex)(self.handle, &config) };
        if ok == 0 {
            return Err(self
                .last_error()
                .unwrap_or_else(|| "fsb_load failed".to_string()));
        }
        Ok(())
    }

    fn last_error(&self) -> Option<String> {
        let ptr = unsafe { (self.api.last_error)(self.handle) };
        if ptr.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.api.destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl FsbApi {
    unsafe fn load(path: &str) -> Result<Self, String> {
        let library =
            load_engine_library(path).map_err(|err| format!("Could not load engine DLL: {err}"))?;

        let create = *library
            .get::<FsbCreate>(b"fsb_create\0")
            .map_err(symbol_error("fsb_create"))?;
        let destroy = *library
            .get::<FsbDestroy>(b"fsb_destroy\0")
            .map_err(symbol_error("fsb_destroy"))?;
        let load = *library
            .get::<FsbLoad>(b"fsb_load\0")
            .map_err(symbol_error("fsb_load"))?;
        let load_ex = *library
            .get::<FsbLoadEx>(b"fsb_load_ex\0")
            .map_err(symbol_error("fsb_load_ex"))?;
        let result_version = *library
            .get::<FsbResultVersion>(b"fsb_result_version\0")
            .map_err(symbol_error("fsb_result_version"))?;
        let is_loaded = *library
            .get::<FsbIsLoaded>(b"fsb_is_loaded\0")
            .map_err(symbol_error("fsb_is_loaded"))?;
        let last_error = *library
            .get::<FsbLastError>(b"fsb_last_error\0")
            .map_err(symbol_error("fsb_last_error"))?;
        let process_bgr = *library
            .get::<FsbProcessBgr>(b"fsb_process_bgr\0")
            .map_err(symbol_error("fsb_process_bgr"))?;
        let process_bgr_mesh = library
            .get::<FsbProcessBgrMesh>(b"fsb_process_bgr_mesh\0")
            .map(|symbol| *symbol)
            .ok();
        let process_image_bvh = library
            .get::<FsbProcessImageBvh>(b"fsb_process_image_bvh\0")
            .map(|symbol| *symbol)
            .ok();
        let process_video = *library
            .get::<FsbProcessVideo>(b"fsb_process_video\0")
            .map_err(symbol_error("fsb_process_video"))?;
        let process_video_mesh = library
            .get::<FsbProcessVideoMesh>(b"fsb_process_video_mesh\0")
            .map(|symbol| *symbol)
            .ok();
        let process_video_mesh_ex = library
            .get::<FsbProcessVideoMeshEx>(b"fsb_process_video_mesh_ex\0")
            .map(|symbol| *symbol)
            .ok();
        let process_video_mesh_tracked = library
            .get::<FsbProcessVideoMeshTracked>(b"fsb_process_video_mesh_tracked\0")
            .map(|symbol| *symbol)
            .ok();
        let process_video_mesh_overlay = library
            .get::<FsbProcessVideoMeshOverlay>(b"fsb_process_video_mesh_overlay\0")
            .map(|symbol| *symbol)
            .ok();

        Ok(Self {
            _library: library,
            create,
            destroy,
            _load: load,
            load_ex,
            result_version,
            is_loaded,
            last_error,
            process_bgr,
            process_bgr_mesh,
            process_image_bvh,
            process_video,
            process_video_mesh,
            process_video_mesh_ex,
            process_video_mesh_tracked,
            process_video_mesh_overlay,
        })
    }
}

struct VideoCallbackBridge<'a, F, M>
where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    progress_callback: &'a mut F,
    mesh_callback: &'a mut M,
    mesh_previews: &'a mut i32,
}

unsafe extern "C" fn video_progress_trampoline<F, M>(
    frame_index: c_int,
    total_frames: c_int,
    persons: c_int,
    frame_ms: f32,
    user: *mut c_void,
) where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    if user.is_null() {
        return;
    }
    let bridge = &mut *(user as *mut VideoCallbackBridge<'_, F, M>);
    (bridge.progress_callback)(VideoProgress {
        frame_index,
        total_frames,
        persons,
        frame_ms,
    });
}

unsafe extern "C" fn video_mesh_trampoline<F, M>(
    frame_index: c_int,
    person_index: c_int,
    vertex_count: c_int,
    pred_vertices: *const f32,
    user: *mut c_void,
) where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    if user.is_null() || pred_vertices.is_null() || vertex_count <= 0 {
        return;
    }
    let vertex_floats = (vertex_count as usize * 3).min(FSB_MESH_FLOAT_COUNT);
    let vertices = std::slice::from_raw_parts(pred_vertices, vertex_floats)
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let bridge = &mut *(user as *mut VideoCallbackBridge<'_, F, M>);
    *bridge.mesh_previews += 1;
    (bridge.mesh_callback)(VideoMeshPreview {
        frame_index,
        person_index,
        track_id: None,
        frame_width: None,
        frame_height: None,
        bbox: None,
        pred_cam_t: None,
        kps3d: None,
        kps2d: None,
        mesh_vertices: vertices,
    });
}

unsafe extern "C" fn video_mesh_ex_trampoline<F, M>(
    frame_index: c_int,
    person_index: c_int,
    frame_width: c_int,
    frame_height: c_int,
    bbox: *const f32,
    pred_cam_t: *const f32,
    kps_count: c_int,
    kps_3d: *const f32,
    vertex_count: c_int,
    pred_vertices: *const f32,
    user: *mut c_void,
) where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    if user.is_null() || pred_vertices.is_null() || vertex_count <= 0 {
        return;
    }
    let vertex_floats = (vertex_count as usize * 3).min(FSB_MESH_FLOAT_COUNT);
    let vertices = std::slice::from_raw_parts(pred_vertices, vertex_floats)
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let bbox = if bbox.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(bbox, 4);
        Some([values[0], values[1], values[2], values[3]])
    };
    let pred_cam_t = if pred_cam_t.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(pred_cam_t, 3);
        Some([values[0], values[1], values[2]])
    };
    let kps3d = if kps_3d.is_null() || kps_count <= 0 {
        None
    } else {
        Some(
            std::slice::from_raw_parts(kps_3d, (kps_count as usize) * 3)
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect(),
        )
    };
    let bridge = &mut *(user as *mut VideoCallbackBridge<'_, F, M>);
    *bridge.mesh_previews += 1;
    (bridge.mesh_callback)(VideoMeshPreview {
        frame_index,
        person_index,
        track_id: None,
        frame_width: Some(frame_width),
        frame_height: Some(frame_height),
        bbox,
        pred_cam_t,
        kps3d,
        kps2d: None,
        mesh_vertices: vertices,
    });
}

unsafe extern "C" fn video_mesh_tracked_trampoline<F, M>(
    frame_index: c_int,
    person_index: c_int,
    track_id: c_int,
    frame_width: c_int,
    frame_height: c_int,
    bbox: *const f32,
    pred_cam_t: *const f32,
    kps_count: c_int,
    kps_3d: *const f32,
    vertex_count: c_int,
    pred_vertices: *const f32,
    user: *mut c_void,
) where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    if user.is_null() || pred_vertices.is_null() || vertex_count <= 0 {
        return;
    }
    let vertex_floats = (vertex_count as usize * 3).min(FSB_MESH_FLOAT_COUNT);
    let vertices = std::slice::from_raw_parts(pred_vertices, vertex_floats)
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let bbox = if bbox.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(bbox, 4);
        Some([values[0], values[1], values[2], values[3]])
    };
    let pred_cam_t = if pred_cam_t.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(pred_cam_t, 3);
        Some([values[0], values[1], values[2]])
    };
    let kps3d = if kps_3d.is_null() || kps_count <= 0 {
        None
    } else {
        Some(
            std::slice::from_raw_parts(kps_3d, (kps_count as usize) * 3)
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect(),
        )
    };
    let bridge = &mut *(user as *mut VideoCallbackBridge<'_, F, M>);
    *bridge.mesh_previews += 1;
    (bridge.mesh_callback)(VideoMeshPreview {
        frame_index,
        person_index,
        track_id: Some(track_id),
        frame_width: Some(frame_width),
        frame_height: Some(frame_height),
        bbox,
        pred_cam_t,
        kps3d,
        kps2d: None,
        mesh_vertices: vertices,
    });
}

unsafe extern "C" fn video_mesh_overlay_trampoline<F, M>(
    frame_index: c_int,
    person_index: c_int,
    track_id: c_int,
    frame_width: c_int,
    frame_height: c_int,
    bbox: *const f32,
    pred_cam_t: *const f32,
    kps_count: c_int,
    kps_3d: *const f32,
    kps2d_count: c_int,
    kps_2d: *const f32,
    vertex_count: c_int,
    pred_vertices: *const f32,
    user: *mut c_void,
) where
    F: FnMut(VideoProgress),
    M: FnMut(VideoMeshPreview),
{
    if user.is_null() || pred_vertices.is_null() || vertex_count <= 0 {
        return;
    }
    let vertex_floats = (vertex_count as usize * 3).min(FSB_MESH_FLOAT_COUNT);
    let vertices = std::slice::from_raw_parts(pred_vertices, vertex_floats)
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let bbox = if bbox.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(bbox, 4);
        Some([values[0], values[1], values[2], values[3]])
    };
    let pred_cam_t = if pred_cam_t.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(pred_cam_t, 3);
        Some([values[0], values[1], values[2]])
    };
    let kps3d = if kps_3d.is_null() || kps_count <= 0 {
        None
    } else {
        Some(
            std::slice::from_raw_parts(kps_3d, (kps_count as usize) * 3)
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect(),
        )
    };
    let kps2d = if kps_2d.is_null() || kps2d_count <= 0 {
        None
    } else {
        Some(
            std::slice::from_raw_parts(kps_2d, (kps2d_count as usize) * 2)
                .chunks_exact(2)
                .map(|chunk| [chunk[0], chunk[1]])
                .collect(),
        )
    };
    let bridge = &mut *(user as *mut VideoCallbackBridge<'_, F, M>);
    *bridge.mesh_previews += 1;
    (bridge.mesh_callback)(VideoMeshPreview {
        frame_index,
        person_index,
        track_id: Some(track_id),
        frame_width: Some(frame_width),
        frame_height: Some(frame_height),
        bbox,
        pred_cam_t,
        kps3d,
        kps2d,
        mesh_vertices: vertices,
    });
}

#[cfg(windows)]
unsafe fn load_engine_library(path: &str) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{
        Library as WindowsLibrary, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_USER_DIRS,
    };
    use std::env;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;

    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
    }

    const LOAD_LIBRARY_SEARCH_APPLICATION_DIR: u32 = 0x0000_0200;
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

    unsafe fn add_dll_directory(path: &Path) {
        if path.is_dir() {
            let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            let _ = AddDllDirectory(wide.as_ptr());
        }
    }

    let _ = SetDefaultDllDirectories(
        LOAD_LIBRARY_SEARCH_APPLICATION_DIR
            | LOAD_LIBRARY_SEARCH_SYSTEM32
            | LOAD_LIBRARY_SEARCH_USER_DIRS,
    );

    if let Some(parent) = Path::new(path).parent() {
        add_dll_directory(parent);
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_bin = PathBuf::from(cuda_path).join("bin");
        add_dll_directory(&cuda_bin);
        add_dll_directory(&cuda_bin.join("x64"));
    }

    let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
        | LOAD_LIBRARY_SEARCH_USER_DIRS;
    WindowsLibrary::load_with_flags(Path::new(path), flags).map(Into::into)
}

#[cfg(not(windows))]
unsafe fn load_engine_library(path: &str) -> Result<Library, libloading::Error> {
    Library::new(path)
}

impl From<&FsbResult> for PersonResult {
    fn from(value: &FsbResult) -> Self {
        Self {
            bbox: value.bbox,
            focal_length: value.focal_length,
            pred_cam_t: value.pred_cam_t,
            global_rot: value.global_rot,
            yolo_kps: (value.has_yolo_kps != 0).then(|| triplets(&value.yolo_kps)),
            kps_2d: (value.has_kps != 0).then(|| pairs(&value.kps_2d)),
            kps_3d: (value.has_kps != 0).then(|| triplets(&value.kps_3d)),
            mesh_vertices: None,
        }
    }
}

impl PersonResult {
    fn from_mesh(value: &FsbResult, vertex_count: c_int, vertices: &[f32]) -> Self {
        let mut person = PersonResult::from(value);
        let vertex_floats = (vertex_count.max(0) as usize * 3).min(FSB_MESH_FLOAT_COUNT);
        if vertex_floats > 0 {
            person.mesh_vertices = Some(
                vertices[..vertex_floats]
                    .chunks_exact(3)
                    .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect(),
            );
        }
        person
    }
}

fn symbol_error(name: &'static str) -> impl Fn(libloading::Error) -> String {
    move |err| format!("Missing C API symbol {name}: {err}")
}

fn triplets<const N: usize>(items: &[f32; N]) -> Vec<[f32; 3]> {
    items
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn pairs<const N: usize>(items: &[f32; N]) -> Vec<[f32; 2]> {
    items
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect()
}
