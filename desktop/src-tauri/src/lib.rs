mod fsb;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use fsb::{
    Engine, LoadEngineRequest, LoadEngineResponse, ProcessImageBvhRequest, ProcessImageRequest,
    ProcessImageResponse, ProcessVideoRequest, ProcessVideoResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{Emitter, Manager};

#[derive(Default)]
struct AppState {
    engine: Arc<Mutex<Option<Engine>>>,
    load_lock: Arc<Mutex<()>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MeshTopology {
    vertex_count: u32,
    indices: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelPaths {
    onnx_dir: String,
    gguf_path: String,
    yolo_path: String,
    backbone_name: String,
    lbs_path: String,
    bvh_template_path: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportMeshObjRequest {
    path: String,
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveTextFileRequest {
    path: String,
    content: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveBinaryFileRequest {
    path: String,
    base64: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CopyMotionOutputsRequest {
    from_path: String,
    to_path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CopyMotionOutputsResponse {
    paths: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageBvhResponse {
    people: i32,
    paths: Vec<String>,
}

#[tauri::command]
async fn load_engine(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    mut request: LoadEngineRequest,
) -> Result<LoadEngineResponse, String> {
    if request.library_path.trim().is_empty() {
        request.library_path = bundled_engine_library_path(&app)?
            .ok_or_else(|| "Bundled engine DLL was not found".to_string())?;
    }

    if request.onnx_dir.trim().is_empty()
        || request.gguf_path.trim().is_empty()
        || request.yolo_path.trim().is_empty()
    {
        if let Some(paths) = find_model_paths(&app)? {
            if request.onnx_dir.trim().is_empty() {
                request.onnx_dir = paths.onnx_dir;
            }
            if request.gguf_path.trim().is_empty() {
                request.gguf_path = paths.gguf_path;
            }
            if request.yolo_path.trim().is_empty() {
                request.yolo_path = paths.yolo_path;
            }
            if request.backbone_name.trim().is_empty() {
                request.backbone_name = paths.backbone_name;
            }
        }
    }

    normalize_load_request(&mut request);

    let engine_store = state.engine.clone();
    let load_lock = state.load_lock.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _load_guard = load_lock
            .lock()
            .map_err(|_| "Engine load lock poisoned".to_string())?;

        {
            let engine = engine_store
                .lock()
                .map_err(|_| "Engine lock poisoned".to_string())?;
            if let Some(engine) = engine.as_ref().filter(|engine| engine.is_loaded()) {
                return Ok(LoadEngineResponse {
                    loaded: true,
                    result_version: engine.result_version(),
                });
            }
        }

        validate_load_request(&request)?;

        {
            let mut engine = engine_store
                .lock()
                .map_err(|_| "Engine lock poisoned".to_string())?;
            *engine = None;
        }

        let engine = Engine::load(request).map_err(load_error_with_context)?;
        let response = LoadEngineResponse {
            loaded: engine.is_loaded(),
            result_version: engine.result_version(),
        };
        *engine_store
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())? = Some(engine);
        Ok(response)
    })
    .await
    .map_err(|err| format!("Engine load task failed: {err}"))?
}

#[tauri::command]
fn bundled_engine_path(app: tauri::AppHandle) -> Result<Option<String>, String> {
    bundled_engine_library_path(&app)
}

#[tauri::command]
fn default_model_paths(app: tauri::AppHandle) -> Result<Option<ModelPaths>, String> {
    find_model_paths(&app)
}

#[tauri::command]
fn model_paths_for_dir(app: tauri::AppHandle, path: String) -> Result<ModelPaths, String> {
    let dir = PathBuf::from(normalize_path_string(&path));
    let mut paths = model_paths_from_onnx_dir(&dir).ok_or_else(|| {
        format!(
            "{} is not a complete SAM3DBody ONNX model folder",
            dir.display()
        )
    })?;
    paths.bvh_template_path = find_bvh_template_path(&app)
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(paths)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HardwareSnapshot {
    gpu_name: String,
    total_vram: u64,
    used_vram: u64,
    free_vram: u64,
    gpu_utilization: u32,
    temperature: u32,
    power_draw: f32,
    power_limit: f32,
    process_ram: u64,
    total_ram: u64,
    used_ram: u64,
    message: String,
}

#[tauri::command]
async fn hardware_snapshot() -> Result<HardwareSnapshot, String> {
    tauri::async_runtime::spawn_blocking(hardware_snapshot_blocking)
        .await
        .map_err(|err| format!("Hardware monitor task failed: {err}"))
}

#[tauri::command]
fn mesh_topology(app: tauri::AppHandle) -> Result<MeshTopology, String> {
    let path =
        find_body_mesh_path(&app).ok_or_else(|| "body_mesh.tri was not found".to_string())?;
    parse_tri_topology(&path)
}

#[tauri::command]
fn image_data_url(path: String) -> Result<String, String> {
    let path = PathBuf::from(normalize_path_string(&path));
    if !path.is_file() {
        return Err(format!("Image was not found: {}", path.display()));
    }
    let bytes =
        fs::read(&path).map_err(|err| format!("Could not read {}: {err}", path.display()))?;
    let mime = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "application/octet-stream",
    };
    Ok(format!(
        "data:{mime};base64,{}",
        BASE64_STANDARD.encode(bytes)
    ))
}

#[tauri::command]
fn export_mesh_obj(request: ExportMeshObjRequest) -> Result<String, String> {
    let path = PathBuf::from(normalize_path_string(&request.path));
    if path.as_os_str().is_empty() {
        return Err("OBJ output path is required".to_string());
    }
    if request.vertices.is_empty() {
        return Err("Run image inference with mesh output before exporting OBJ".to_string());
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
        }
    }

    let mut file =
        File::create(&path).map_err(|err| format!("Could not create {}: {err}", path.display()))?;
    writeln!(file, "# SAM3DBody mesh export")
        .map_err(|err| format!("Could not write {}: {err}", path.display()))?;
    for [x, y, z] in &request.vertices {
        writeln!(file, "v {x} {y} {z}")
            .map_err(|err| format!("Could not write {}: {err}", path.display()))?;
    }
    for face in request.indices.chunks_exact(3) {
        writeln!(file, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)
            .map_err(|err| format!("Could not write {}: {err}", path.display()))?;
    }

    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
fn save_text_file(request: SaveTextFileRequest) -> Result<String, String> {
    let path = PathBuf::from(normalize_path_string(&request.path));
    if path.as_os_str().is_empty() {
        return Err("Output path is required".to_string());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
        }
    }
    fs::write(&path, request.content)
        .map_err(|err| format!("Could not write {}: {err}", path.display()))?;
    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    let path = PathBuf::from(normalize_path_string(&path));
    fs::read_to_string(&path).map_err(|err| format!("Could not read {}: {err}", path.display()))
}

#[tauri::command]
fn save_binary_file(request: SaveBinaryFileRequest) -> Result<String, String> {
    let path = PathBuf::from(normalize_path_string(&request.path));
    if path.as_os_str().is_empty() {
        return Err("Output path is required".to_string());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
        }
    }
    let bytes = BASE64_STANDARD
        .decode(request.base64.as_bytes())
        .map_err(|err| format!("Could not decode binary payload: {err}"))?;
    fs::write(&path, bytes).map_err(|err| format!("Could not write {}: {err}", path.display()))?;
    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
fn copy_motion_outputs(
    request: CopyMotionOutputsRequest,
) -> Result<CopyMotionOutputsResponse, String> {
    let from_path = PathBuf::from(normalize_path_string(&request.from_path));
    let to_path = PathBuf::from(normalize_path_string(&request.to_path));
    if from_path.as_os_str().is_empty() {
        return Err("Generated motion path is missing".to_string());
    }
    if to_path.as_os_str().is_empty() {
        return Err("BVH output path is required".to_string());
    }
    if let Some(parent) = to_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
        }
    }

    let sources = find_motion_output_files(&from_path)?;
    let source_stem = from_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("motion");
    let target_stem = to_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("motion");
    let target_ext = to_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("bvh");

    let mut written = Vec::new();
    for source in sources {
        let source_file_stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(source_stem);
        let suffix = source_file_stem
            .strip_prefix(source_stem)
            .unwrap_or_default();
        let target = if suffix.is_empty() {
            to_path.clone()
        } else {
            to_path.with_file_name(format!("{target_stem}{suffix}.{target_ext}"))
        };
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
            }
        }
        fs::copy(&source, &target).map_err(|err| {
            format!(
                "Could not copy {} to {}: {err}",
                source.display(),
                target.display()
            )
        })?;
        written.push(target.to_string_lossy().into_owned());
    }

    Ok(CopyMotionOutputsResponse { paths: written })
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("Only http and https links can be opened".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        Command::new("rundll32.exe")
            .args(["url.dll,FileProtocolHandler", &url])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|err| format!("Could not open link: {err}"))?;
        return Ok(());
    }
    #[cfg(not(target_os = "windows"))]
    {
        Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|err| format!("Could not open link: {err}"))?;
        Ok(())
    }
}

#[tauri::command]
fn unload_engine(state: tauri::State<'_, AppState>) -> Result<(), String> {
    *state
        .engine
        .lock()
        .map_err(|_| "Engine lock poisoned".to_string())? = None;
    Ok(())
}

#[tauri::command]
fn engine_status(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let guard = state
        .engine
        .lock()
        .map_err(|_| "Engine lock poisoned".to_string())?;
    Ok(guard
        .as_ref()
        .map(|engine| engine.is_loaded())
        .unwrap_or(false))
}

#[tauri::command]
async fn process_image(
    state: tauri::State<'_, AppState>,
    mut request: ProcessImageRequest,
) -> Result<ProcessImageResponse, String> {
    request.image_path = normalize_path_string(&request.image_path);
    let engine_store = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let guard = engine_store
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        let engine = guard
            .as_ref()
            .ok_or_else(|| "Load the engine before processing an image".to_string())?;
        engine.process_image(request)
    })
    .await
    .map_err(|err| format!("Image processing task failed: {err}"))?
}

#[tauri::command]
async fn process_image_bvh(
    state: tauri::State<'_, AppState>,
    mut request: ProcessImageBvhRequest,
) -> Result<ImageBvhResponse, String> {
    normalize_image_bvh_request(&mut request);
    let output_path = PathBuf::from(request.bvh_path.clone());
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Could not create {}: {err}", parent.display()))?;
        }
    }

    let engine_store = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let guard = engine_store
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        let engine = guard
            .as_ref()
            .ok_or_else(|| "Load the engine before exporting image BVH".to_string())?;
        let people = engine.process_image_bvh(request)?;
        let paths = if people > 0 {
            find_motion_output_files(&output_path)?
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect()
        } else {
            Vec::new()
        };
        Ok(ImageBvhResponse { people, paths })
    })
    .await
    .map_err(|err| format!("Image BVH export task failed: {err}"))?
}

#[tauri::command]
async fn process_video(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    mut request: ProcessVideoRequest,
) -> Result<ProcessVideoResponse, String> {
    normalize_video_request(&mut request);
    if request.bvh_path.trim().is_empty() {
        request.bvh_path = temp_motion_output_path(&app, &request.video_path)?;
    }
    let engine_store = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let guard = engine_store
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        let engine = guard
            .as_ref()
            .ok_or_else(|| "Load the engine before processing a video".to_string())?;
        let progress_app = app.clone();
        let mesh_app = app.clone();
        engine.process_video(
            request,
            move |progress| {
                let _ = progress_app.emit("video-progress", progress);
            },
            move |preview| {
                let _ = mesh_app.emit("video-mesh-preview", preview);
            },
        )
    })
    .await
    .map_err(|err| format!("Video processing task failed: {err}"))?
}

fn temp_motion_output_path(app: &tauri::AppHandle, video_path: &str) -> Result<String, String> {
    let base_dir = app
        .path()
        .app_cache_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("SAM3DBody Desktop"));
    let dir = base_dir.join("generated-motion");
    fs::create_dir_all(&dir)
        .map_err(|err| format!("Could not create motion cache {}: {err}", dir.display()))?;
    let stem = Path::new(video_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("motion");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(dir
        .join(format!("{stem}_{stamp}.bvh"))
        .to_string_lossy()
        .into_owned())
}

fn bundled_engine_library_path(app: &tauri::AppHandle) -> Result<Option<String>, String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|err| format!("Could not resolve app resource directory: {err}"))?;

    let candidates = [
        resource_dir.join("resources/engine/fast_sam_3dbody.dll"),
        resource_dir.join("engine/fast_sam_3dbody.dll"),
        resource_dir.join("fast_sam_3dbody.dll"),
    ];

    Ok(candidates
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned()))
}

fn find_model_paths(app: &tauri::AppHandle) -> Result<Option<ModelPaths>, String> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        push_search_roots(&resource_dir, &mut roots, &mut seen);
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_search_roots(exe_dir, &mut roots, &mut seen);
        }
    }
    if let Ok(app_data_dir) = app.path().app_data_dir() {
        push_search_roots(&app_data_dir, &mut roots, &mut seen);
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_search_roots(&cwd, &mut roots, &mut seen);
    }

    for root in roots {
        let candidates = [
            root.clone(),
            root.join("onnx"),
            root.join("models").join("onnx"),
            root.join("resources").join("onnx"),
            root.join("resources").join("models").join("onnx"),
        ];
        for candidate in candidates {
            if let Some(mut paths) = model_paths_from_onnx_dir(&candidate) {
                paths.bvh_template_path = find_bvh_template_path(app)
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Ok(Some(paths));
            }
        }
    }

    Ok(None)
}

fn push_search_roots(start: &Path, roots: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>) {
    for ancestor in start.ancestors().take(8) {
        push_unique(ancestor.to_path_buf(), roots, seen);
    }
}

fn push_unique(path: PathBuf, roots: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>) {
    let key = path.canonicalize().unwrap_or_else(|_| path.clone());
    if seen.insert(key) {
        roots.push(path);
    }
}

fn model_paths_from_onnx_dir(onnx_dir: &Path) -> Option<ModelPaths> {
    let gguf_path = onnx_dir.join("pipeline.gguf");
    let yolo_path = onnx_dir.join("yolo.onnx");
    let backbone_name = if onnx_dir.join("backbone.onnx").is_file() {
        "backbone.onnx"
    } else if onnx_dir.join("backbone_fp32.onnx").is_file() {
        "backbone_fp32.onnx"
    } else {
        return None;
    };

    if !gguf_path.is_file()
        || !yolo_path.is_file()
        || !onnx_dir.join("decoder.onnx").is_file()
        || !onnx_dir.join("body_model.lbs").is_file()
        || !onnx_dir.join("correctives.bin").is_file()
        || !onnx_dir.join("keypoint_mapping.bin").is_file()
        || !onnx_dir.join(format!("{backbone_name}.data")).is_file()
    {
        return None;
    }

    Some(ModelPaths {
        onnx_dir: path_to_string(onnx_dir),
        gguf_path: path_to_string(&gguf_path),
        yolo_path: path_to_string(&yolo_path),
        backbone_name: backbone_name.to_string(),
        lbs_path: path_to_string(&onnx_dir.join("body_model.lbs")),
        bvh_template_path: String::new(),
    })
}

fn find_bvh_template_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        push_search_roots(&resource_dir, &mut roots, &mut seen);
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_search_roots(exe_dir, &mut roots, &mut seen);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_search_roots(&cwd, &mut roots, &mut seen);
    }

    for root in roots {
        let candidates = [
            root.join("body.bvh"),
            root.join("resources").join("templates").join("body.bvh"),
            root.join("templates").join("body.bvh"),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_body_mesh_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        push_search_roots(&resource_dir, &mut roots, &mut seen);
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_search_roots(exe_dir, &mut roots, &mut seen);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_search_roots(&cwd, &mut roots, &mut seen);
    }

    for root in roots {
        let candidates = [
            root.join("body_mesh.tri"),
            root.join("resources")
                .join("templates")
                .join("body_mesh.tri"),
            root.join("templates").join("body_mesh.tri"),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn parse_tri_topology(path: &Path) -> Result<MeshTopology, String> {
    const TRI_HEADER_SIZE: usize = 136;

    let bytes =
        fs::read(path).map_err(|err| format!("Could not read {}: {err}", path.display()))?;
    if bytes.len() < TRI_HEADER_SIZE || &bytes[0..5] != b"TRI3D" {
        return Err(format!("{} is not a TRI mesh", path.display()));
    }

    let name_size = read_u32_le(&bytes, 12)? as usize;
    let float_size = read_u32_le(&bytes, 16)?;
    let vertex_float_count = read_u32_le(&bytes, 24)? as usize;
    let normal_float_count = read_u32_le(&bytes, 28)? as usize;
    let texture_float_count = read_u32_le(&bytes, 32)? as usize;
    let color_float_count = read_u32_le(&bytes, 36)? as usize;
    let index_count = read_u32_le(&bytes, 40)? as usize;

    if float_size != 4 || vertex_float_count % 3 != 0 {
        return Err(format!("{} has an unsupported TRI layout", path.display()));
    }

    let indices_offset = TRI_HEADER_SIZE
        .checked_add(name_size)
        .and_then(|offset| offset.checked_add(vertex_float_count.checked_mul(4)?))
        .and_then(|offset| offset.checked_add(normal_float_count.checked_mul(4)?))
        .and_then(|offset| offset.checked_add(texture_float_count.checked_mul(4)?))
        .and_then(|offset| offset.checked_add(color_float_count.checked_mul(4)?))
        .ok_or_else(|| "TRI mesh offsets overflowed".to_string())?;
    let indices_bytes = index_count
        .checked_mul(4)
        .ok_or_else(|| "TRI index count overflowed".to_string())?;
    if bytes.len() < indices_offset + indices_bytes {
        return Err(format!("{} is truncated before index data", path.display()));
    }

    let mut indices = Vec::with_capacity(index_count);
    for offset in (indices_offset..indices_offset + indices_bytes).step_by(4) {
        indices.push(read_u32_le(&bytes, offset)?);
    }

    Ok(MeshTopology {
        vertex_count: (vertex_float_count / 3) as u32,
        indices,
    })
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset + 4;
    if bytes.len() < end {
        return Err("Unexpected end of TRI mesh".to_string());
    }
    Ok(u32::from_le_bytes(
        bytes[offset..end]
            .try_into()
            .map_err(|_| "Could not read TRI integer".to_string())?,
    ))
}

fn hardware_snapshot_blocking() -> HardwareSnapshot {
    let mut snapshot = HardwareSnapshot {
        gpu_name: "NVIDIA GPU".to_string(),
        total_vram: 0,
        used_vram: 0,
        free_vram: 0,
        gpu_utilization: 0,
        temperature: 0,
        power_draw: 0.0,
        power_limit: 0.0,
        process_ram: 0,
        total_ram: 0,
        used_ram: 0,
        message: "Waiting for nvidia-smi".to_string(),
    };

    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]);
    suppress_child_console(&mut command);
    let output = command.output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().find(|line| !line.trim().is_empty()) {
                let parts: Vec<_> = line.split(',').map(|part| part.trim()).collect();
                if parts.len() >= 8 {
                    snapshot.gpu_name = parts[0].to_string();
                    snapshot.total_vram = mib_to_bytes(parse_u64(parts[1]));
                    snapshot.used_vram = mib_to_bytes(parse_u64(parts[2]));
                    snapshot.free_vram = mib_to_bytes(parse_u64(parts[3]));
                    snapshot.gpu_utilization = parse_u32(parts[4]);
                    snapshot.temperature = parse_u32(parts[5]);
                    snapshot.power_draw = parse_f32(parts[6]);
                    snapshot.power_limit = parse_f32(parts[7]);
                    snapshot.message = "GPU telemetry".to_string();
                }
            }
        }
        Ok(output) => {
            snapshot.message = String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(160)
                .collect();
            if snapshot.message.is_empty() {
                snapshot.message = "nvidia-smi returned no GPU data".to_string();
            }
        }
        Err(err) => {
            snapshot.message = format!("nvidia-smi unavailable: {err}");
        }
    }

    let (total_ram, used_ram, process_ram) = memory_snapshot();
    snapshot.total_ram = total_ram;
    snapshot.used_ram = used_ram;
    snapshot.process_ram = process_ram;
    snapshot
}

fn parse_u64(value: &str) -> u64 {
    value
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| value.max(0.0) as u64)
        .unwrap_or(0)
}

fn parse_u32(value: &str) -> u32 {
    parse_u64(value).min(u32::MAX as u64) as u32
}

fn parse_f32(value: &str) -> f32 {
    value
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0)
}

fn mib_to_bytes(value: u64) -> u64 {
    value.saturating_mul(1024 * 1024)
}

#[cfg(windows)]
fn suppress_child_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn suppress_child_console(_command: &mut Command) {}

#[cfg(windows)]
fn memory_snapshot() -> (u64, u64, u64) {
    use std::ffi::c_void;

    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    let mut memory = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let mut total = 0;
    let mut used = 0;
    if unsafe { GlobalMemoryStatusEx(&mut memory) } != 0 {
        total = memory.ull_total_phys;
        used = memory.ull_total_phys.saturating_sub(memory.ull_avail_phys);
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let process_ram = if unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    } != 0
    {
        counters.working_set_size as u64
    } else {
        0
    };

    (total, used, process_ram)
}

#[cfg(not(windows))]
fn memory_snapshot() -> (u64, u64, u64) {
    (0, 0, 0)
}

fn validate_load_request(request: &LoadEngineRequest) -> Result<(), String> {
    ensure_file("Engine DLL", &request.library_path)?;
    ensure_dir("ONNX folder", &request.onnx_dir)?;
    ensure_file("GGUF", &request.gguf_path)?;
    ensure_file("YOLO", &request.yolo_path)?;

    let backbone_path = resolve_model_file(&request.onnx_dir, &request.backbone_name);
    ensure_file("Backbone", &backbone_path.to_string_lossy())?;
    ensure_file(
        "Backbone external data",
        &PathBuf::from(format!("{}.data", backbone_path.to_string_lossy())).to_string_lossy(),
    )?;
    ensure_file(
        "Decoder",
        &Path::new(request.onnx_dir.trim())
            .join("decoder.onnx")
            .to_string_lossy(),
    )?;

    if !request.skip_body_model {
        let onnx_dir = Path::new(request.onnx_dir.trim());
        ensure_file(
            "Body model LBS",
            &onnx_dir.join("body_model.lbs").to_string_lossy(),
        )?;
        ensure_file(
            "Body correctives",
            &onnx_dir.join("correctives.bin").to_string_lossy(),
        )?;
        ensure_file(
            "Keypoint mapping",
            &onnx_dir.join("keypoint_mapping.bin").to_string_lossy(),
        )?;
    }

    Ok(())
}

fn normalize_load_request(request: &mut LoadEngineRequest) {
    request.library_path = normalize_path_string(&request.library_path);
    request.onnx_dir = normalize_path_string(&request.onnx_dir);
    request.gguf_path = normalize_path_string(&request.gguf_path);
    request.yolo_path = normalize_path_string(&request.yolo_path);
    request.backbone_name = normalize_path_string(&request.backbone_name);
}

fn normalize_video_request(request: &mut ProcessVideoRequest) {
    request.video_path = normalize_path_string(&request.video_path);
    request.bvh_path = normalize_path_string(&request.bvh_path);
    request.bvh_template_path = normalize_path_string(&request.bvh_template_path);
    request.lbs_path = normalize_path_string(&request.lbs_path);
    request.csv_path = normalize_path_string(&request.csv_path);
}

fn normalize_image_bvh_request(request: &mut ProcessImageBvhRequest) {
    request.image_path = normalize_path_string(&request.image_path);
    request.bvh_path = normalize_path_string(&request.bvh_path);
    request.bvh_template_path = normalize_path_string(&request.bvh_template_path);
    request.lbs_path = normalize_path_string(&request.lbs_path);
}

fn find_motion_output_files(base_path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    if base_path.is_file() {
        files.push(base_path.to_path_buf());
    }

    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let ext = base_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("bvh");

    if parent.is_dir() && !stem.is_empty() {
        for entry in fs::read_dir(parent)
            .map_err(|err| format!("Could not inspect {}: {err}", parent.display()))?
        {
            let path = entry
                .map_err(|err| format!("Could not inspect {}: {err}", parent.display()))?
                .path();
            if !path.is_file() {
                continue;
            }
            let Some(file_stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let file_ext = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if file_ext.eq_ignore_ascii_case(ext)
                && file_stem.starts_with(&format!("{stem}_"))
                && !files.iter().any(|candidate| candidate == &path)
            {
                files.push(path);
            }
        }
    }

    files.sort();
    if files.is_empty() {
        Err(format!(
            "Generated BVH files were not found beside {}",
            base_path.display()
        ))
    } else {
        Ok(files)
    }
}

fn normalize_path_string(path: &str) -> String {
    let trimmed = path.trim().trim_matches('"');
    if let Some(rest) = trimmed.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{rest}")
    } else if let Some(rest) = trimmed.strip_prefix("\\\\?\\") {
        rest.to_string()
    } else {
        trimmed.to_string()
    }
}

fn path_to_string(path: &Path) -> String {
    normalize_path_string(&path.to_string_lossy())
}

fn resolve_model_file(onnx_dir: &str, name_or_path: &str) -> PathBuf {
    let model_path = PathBuf::from(name_or_path.trim());
    if model_path.is_absolute() {
        model_path
    } else {
        Path::new(onnx_dir.trim()).join(model_path)
    }
}

fn ensure_file(label: &str, path: &str) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} path is required"));
    }
    let path = Path::new(trimmed);
    if !path.is_file() {
        return Err(format!("{label} was not found: {}", path.display()));
    }
    Ok(())
}

fn ensure_dir(label: &str, path: &str) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} path is required"));
    }
    let path = Path::new(trimmed);
    if !path.is_dir() {
        return Err(format!("{label} was not found: {}", path.display()));
    }
    Ok(())
}

fn load_error_with_context(err: String) -> String {
    if err.contains("Pipeline load failed") {
        format!(
            "{err}\n\nThe selected model folder looks incomplete or mismatched. Click Find Models, or point ONNX folder at the directory containing backbone.onnx, backbone.onnx.data, decoder.onnx, yolo.onnx, pipeline.gguf, body_model.lbs, correctives.bin, and keypoint_mapping.bin."
        )
    } else {
        err
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            load_engine,
            bundled_engine_path,
            default_model_paths,
            model_paths_for_dir,
            hardware_snapshot,
            mesh_topology,
            image_data_url,
            export_mesh_obj,
            save_text_file,
            read_text_file,
            save_binary_file,
            copy_motion_outputs,
            open_external_url,
            unload_engine,
            engine_status,
            process_image,
            process_image_bvh,
            process_video
        ])
        .run(tauri::generate_context!())
        .expect("error while running SAM3DBody Desktop");
}
