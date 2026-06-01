#pragma once
// ============================================================================
// fast_sam_3dbody_capi.h  –  Plain C API for ctypes / cffi access
// ============================================================================

#include <stdint.h>

#if defined(_WIN32)
#  if defined(FSB_BUILDING_DLL)
#    define FSB_API __declspec(dllexport)
#  else
#    define FSB_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define FSB_API __attribute__((visibility("default")))
#else
#  define FSB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FSB_RESULT_VERSION 1
#define FSB_MESH_VERTEX_COUNT 18439
#define FSB_MESH_FLOAT_COUNT (FSB_MESH_VERTEX_COUNT * 3)

// Opaque pipeline handle
typedef void* FsbHandle;

typedef void (*FsbVideoProgressCallback)(int frame_index,
                                         int total_frames,
                                         int persons,
                                         float frame_ms,
                                         void* user);

typedef void (*FsbVideoMeshCallback)(int frame_index,
                                     int person_index,
                                     int vertex_count,
                                     const float* pred_vertices,
                                     void* user);

typedef void (*FsbVideoMeshExCallback)(int frame_index,
                                       int person_index,
                                       int frame_width,
                                       int frame_height,
                                       const float* bbox,
                                       const float* pred_cam_t,
                                       int kps_count,
                                       const float* kps_3d,
                                       int vertex_count,
                                       const float* pred_vertices,
                                       void* user);

typedef void (*FsbVideoMeshTrackedCallback)(int frame_index,
                                            int person_index,
                                            int track_id,
                                            int frame_width,
                                            int frame_height,
                                            const float* bbox,
                                            const float* pred_cam_t,
                                            int kps_count,
                                            const float* kps_3d,
                                            int vertex_count,
                                            const float* pred_vertices,
                                            void* user);

typedef void (*FsbVideoMeshOverlayCallback)(int frame_index,
                                            int person_index,
                                            int track_id,
                                            int frame_width,
                                            int frame_height,
                                            const float* bbox,
                                            const float* pred_cam_t,
                                            int kps_count,
                                            const float* kps_3d,
                                            int kps2d_count,
                                            const float* kps_2d,
                                            int vertex_count,
                                            const float* pred_vertices,
                                            void* user);

// ── Config passed from Python ─────────────────────────────────────────────────
typedef struct {
    const char* onnx_dir;
    const char* gguf_path;
    const char* yolo_path;
    int   cuda_device;      // -1 = CPU
    int   skip_body_model;  // 0/1
    float person_thresh;
    float person_nms_iou;
    int   max_persons;      // 0 = unlimited
    float focal_x;
    float focal_y;
    float principal_x;
    float principal_y;
    int   zero_face_params;   // 0/1  — force face expression coefficients to zero
} FsbConfig;

typedef struct {
    uint32_t struct_size;    // set to sizeof(FsbConfigEx)
    FsbConfig base;
    const char* backbone_name;
    int   use_trt_ep;        // 0/1
    int   use_fp16;          // 0/1
} FsbConfigEx;

typedef struct {
    const char* input_path;
    const char* bvh_path;       // optional; writes per-person BVH motion files
    const char* bvh_template;   // required when bvh_path is set
    const char* lbs_path;       // required when bvh_path is set
    const char* csv_path;       // optional diagnostic 3-D keypoint CSV
    int max_frames;     // 0 = all frames
    int frame_stride;   // 1 = every frame
    int max_results;    // result capacity per frame
    int butterworth;     // 0/1; smooth keypoints, pose, hands, root translation
    float bw_cutoff;     // Hz; ignored when >= Nyquist
    int butterworth_root_rotation; // 0/1; quaternion root-rotation smoothing
    float rot_clamp_deg; // deg/frame outlier clamp; 0 = disabled
} FsbVideoConfig;

typedef struct {
    int frames_read;
    int frames_processed;
    int persons_total;
    double source_fps;
    double total_ms;
} FsbVideoSummary;

// ── Per-person result (fixed-size for easy ctypes mapping) ────────────────────
typedef struct {
    float bbox[4];          // x1, y1, x2, y2  (original image pixels)
    float focal_length;
    float pred_cam_t[3];    // tx, ty, tz
    float global_rot[3];    // Euler ZYX

    float body_pose[133];
    float shape[45];
    float scale[28];
    float hand_pose[108];
    float face_params[72];

    // 2-D YOLO keypoints: 17 COCO joints × [x, y, confidence]
    float yolo_kps[51];     // layout: [kp0_x, kp0_y, kp0_vis, kp1_x, ...]
    int   has_yolo_kps;     // 1 if YOLO ran and detected this person

    // 3-D keypoints (from body model; 0 when skip_body_model=1)
    float kps_3d[210];      // [70 × 3]
    float kps_2d[140];      // [70 × 2]
    int   has_kps;

    // ── Second-pass fields (added for --two-passes Python second decoder pass) ──
    //
    // pred_pose_raw: the raw MHR FFN output BEFORE Euler conversion.
    //   Layout (mirrors fast_sam_3dbody.cpp parse block ≈ line 755):
    //     [0:6]    global_rot_6d (6D continuous rotation, Zhou et al. 2019)
    //     [6:266]  body_cont[260] (23×6D + 58×sincos + 6trans)
    //   Together these 266 floats form the first part of the second-pass
    //   prev_estimate tensor: cat(pred_pose_raw, shape, scale, hand, face).
    //   Shape[45], scale[28], hand[108], face[72] are already in the fields above.
    //
    // pred_cam_raw: raw camera head FFN output [3] before the nonlinear
    //   s/tx/ty → pred_cam_t conversion.  Appended to prev_estimate when the
    //   loaded Python model has an init_camera attribute.
    //
    // IMPORTANT: these fields are appended at the END of FsbResult so that the
    // ctypes struct layout for older code is not disturbed.
    float pred_pose_raw[266];  // global_rot_6d[6] + body_cont[260]
    float pred_cam_raw[3];     // raw cam head output before s/tx/ty decode

    // Assembled model_params[204] used by native C LBS (hand + scale already decoded).
    // Layout: [0:3]=global_trans*10, [3:6]=global_rot ZYX, [6:136]=body_pose[:130],
    //         [136:204]=scale_out.  Mirrors Python mhr_forward(..., return_model_params=True).
    float mhr_model_params[204];
} FsbResult;

// ── Lifecycle ─────────────────────────────────────────────────────────────────
FSB_API FsbHandle fsb_create(void);
FSB_API void      fsb_destroy(FsbHandle h);

// Returns 1 on success, 0 on failure.
FSB_API int fsb_load(FsbHandle h, const FsbConfig* cfg);
FSB_API int fsb_load_ex(FsbHandle h, const FsbConfigEx* cfg);

// ABI/runtime helpers for desktop app integrations.
FSB_API int         fsb_result_version(void);
FSB_API int         fsb_is_loaded(FsbHandle h);
FSB_API const char* fsb_last_error(FsbHandle h);

// ── Inference ─────────────────────────────────────────────────────────────────
// Process a BGR uint8 image.
// results    : pre-allocated array of FsbResult with at least max_results entries.
// max_results: capacity of results[].
// Returns number of persons written (<= max_results), or -1 on error.
FSB_API int fsb_process_bgr(FsbHandle        h,
                            const uint8_t*   bgr,
                            int              width,
                            int              height,
                            FsbResult*       results,
                            int              max_results);

// Extended image processing for integrations that want mesh preview/export.
// results uses the stable FsbResult ABI; pred_vertices is a caller-owned flat
// buffer sized max_results * vertex_float_capacity floats.
FSB_API int fsb_process_bgr_mesh(FsbHandle        h,
                                 const uint8_t*   bgr,
                                 int              width,
                                 int              height,
                                 FsbResult*       results,
                                 int*             vertex_counts,
                                 float*           pred_vertices,
                                 int              max_results,
                                 int              vertex_float_capacity);

FSB_API int fsb_process_image_bvh(FsbHandle  h,
                                  const char* image_path,
                                  const char* bvh_path,
                                  const char* bvh_template,
                                  const char* lbs_path,
                                  int         max_results);

FSB_API int fsb_process_video(FsbHandle                 h,
                              const FsbVideoConfig*     cfg,
                              FsbVideoSummary*          summary,
                              FsbVideoProgressCallback  progress_cb,
                              void*                     user);

FSB_API int fsb_process_video_mesh(FsbHandle                 h,
                                   const FsbVideoConfig*     cfg,
                                   FsbVideoSummary*          summary,
                                   FsbVideoProgressCallback  progress_cb,
                                   FsbVideoMeshCallback      mesh_cb,
                                   void*                     user);

FSB_API int fsb_process_video_mesh_ex(FsbHandle                 h,
                                      const FsbVideoConfig*     cfg,
                                      FsbVideoSummary*          summary,
                                      FsbVideoProgressCallback  progress_cb,
                                      FsbVideoMeshExCallback    mesh_cb,
                                      void*                     user);

FSB_API int fsb_process_video_mesh_tracked(FsbHandle                    h,
                                           const FsbVideoConfig*        cfg,
                                           FsbVideoSummary*             summary,
                                           FsbVideoProgressCallback     progress_cb,
                                           FsbVideoMeshTrackedCallback  mesh_cb,
                                           void*                        user);

FSB_API int fsb_process_video_mesh_overlay(FsbHandle                    h,
                                           const FsbVideoConfig*        cfg,
                                           FsbVideoSummary*             summary,
                                           FsbVideoProgressCallback     progress_cb,
                                           FsbVideoMeshOverlayCallback  mesh_cb,
                                           void*                        user);

#ifdef __cplusplus
}
#endif
