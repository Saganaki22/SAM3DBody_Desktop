// ============================================================================
// fast_sam_3dbody_capi.cpp  –  Plain C wrapper for ctypes / cffi
// ============================================================================

#include "fast_sam_3dbody_capi.h"
#include "fast_sam_3dbody.h"
#include "bvh_writer.h"
#include "outputFiltering.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct FsbContext
{
    fsb::Pipeline pipeline;
    std::string last_error;
};

static FsbContext* as_ctx(FsbHandle h)
{
    return static_cast<FsbContext*>(h);
}

static void set_error(FsbContext* ctx, const std::string& message)
{
    if (ctx) ctx->last_error = message;
}

static void clear_error(FsbContext* ctx)
{
    if (ctx) ctx->last_error.clear();
}

static void copy_vec(float* dst, const std::vector<float>& src, int n)
{
    int cnt = std::min((int)src.size(), n);
    if (cnt > 0) std::memcpy(dst, src.data(), cnt * sizeof(float));
}

static void copy_result(FsbResult& out, const fsb::MHRResult& r)
{
    std::memset(&out, 0, sizeof(FsbResult));

    std::memcpy(out.bbox,        r.bbox.data(),        4  * sizeof(float));
    out.focal_length = r.focal_length;
    std::memcpy(out.pred_cam_t,  r.pred_cam_t.data(),  3  * sizeof(float));
    std::memcpy(out.global_rot,  r.global_rot.data(),  3  * sizeof(float));

    copy_vec(out.body_pose,   r.body_pose,   133);
    copy_vec(out.shape,       r.shape,        45);
    copy_vec(out.scale,       r.scale,        28);
    copy_vec(out.hand_pose,   r.hand_pose,   108);
    copy_vec(out.face_params, r.face_params,  72);

    if (!r.keypoints_yolo.empty())
    {
        copy_vec(out.yolo_kps, r.keypoints_yolo, 51);
        out.has_yolo_kps = 1;
    }

    if (!r.keypoints_3d.empty())
    {
        copy_vec(out.kps_3d, r.keypoints_3d, 210);
        copy_vec(out.kps_2d, r.keypoints_2d, 140);
        out.has_kps = 1;
    }

    std::memcpy(out.pred_pose_raw, r.pred_pose_raw.data(), 266 * sizeof(float));
    std::memcpy(out.pred_cam_raw,  r.pred_cam_raw.data(),  3   * sizeof(float));
    std::memcpy(out.mhr_model_params, r.mhr_model_params.data(), 204 * sizeof(float));
}

static void write_video_csv_header(std::ofstream& f)
{
    f << "frame,person,bbox_x1,bbox_y1,bbox_x2,bbox_y2,focal,cam_x,cam_y,cam_z,root_rx,root_ry,root_rz";
    for (int j = 0; j < 70; ++j)
        f << ",kp" << j << "_x,kp" << j << "_y,kp" << j << "_z";
    f << "\n";
}

static void write_video_csv_rows(std::ofstream& f, int frame_no,
                                 const std::vector<fsb::MHRResult>& results)
{
    for (int i = 0; i < (int)results.size(); ++i)
    {
        const auto& r = results[i];
        f << frame_no << "," << i
          << "," << r.bbox[0] << "," << r.bbox[1]
          << "," << r.bbox[2] << "," << r.bbox[3]
          << "," << r.focal_length
          << "," << r.pred_cam_t[0] << "," << r.pred_cam_t[1] << "," << r.pred_cam_t[2]
          << "," << r.global_rot[0] << "," << r.global_rot[1] << "," << r.global_rot[2];

        for (int j = 0; j < 70; ++j)
        {
            if ((int)r.keypoints_3d.size() >= (j + 1) * 3)
            {
                f << "," << r.keypoints_3d[j*3]
                  << "," << r.keypoints_3d[j*3+1]
                  << "," << r.keypoints_3d[j*3+2];
            }
            else
            {
                f << ",,,";
            }
        }
        f << "\n";
    }
}

static double elapsed_ms(std::chrono::steady_clock::time_point t0)
{
    auto dt = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count();
}

struct PersonFilters
{
    std::array<ButterWorth, 70*3> kp3d{};
    std::array<ButterWorth, 133>  body_pose{};
    std::array<ButterWorth, 108>  hand_pose{};
    std::array<ButterWorth, 3>    cam_t{};
    QuatLPF                       root_rot{};
};

struct ButterworthState
{
    bool requested = false;
    bool use_scalar = false;
    bool filter_root_rot = false;
    float fps = 30.0f;
    float cutoff = 6.0f;
    float rot_clamp_deg = 0.0f;
    std::vector<PersonFilters> filters;
};

struct CapiTrack
{
    int id = -1;
    std::array<float, 4> bbox{};
    int last_seen_frame = 0;
};

struct TrackAssignment
{
    std::vector<int> track_ids;
    std::vector<int> pad_ids;
};

constexpr float CAPI_TRACK_IOU_THRESH = 0.10f;
constexpr int CAPI_TRACK_MAX_MISSING = 90;

static bool bbox_looks_valid(const std::array<float, 4>& b)
{
    const float w = b[2] - b[0];
    const float h = b[3] - b[1];
    if (w < 8.f || h < 8.f) return false;
    if (b[0] == 0.f && b[1] == 0.f) return false;
    return true;
}

static float bbox_iou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float ix1 = std::max(a[0], b[0]);
    const float iy1 = std::max(a[1], b[1]);
    const float ix2 = std::min(a[2], b[2]);
    const float iy2 = std::min(a[3], b[3]);
    const float iw = std::max(0.f, ix2 - ix1);
    const float ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    if (inter <= 0.f) return 0.f;
    const float aa = std::max(0.f, a[2] - a[0]) * std::max(0.f, a[3] - a[1]);
    const float bb = std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
    const float u = aa + bb - inter;
    return u > 0.f ? inter / u : 0.f;
}

static TrackAssignment assign_capi_tracks(const std::vector<fsb::MHRResult>& results,
                                          std::vector<CapiTrack>& tracks,
                                          int& next_track_id,
                                          int frame_index)
{
    TrackAssignment assignment;
    assignment.track_ids.assign(results.size(), -1);

    struct Pair
    {
        int det;
        int track;
        float iou;
    };
    std::vector<Pair> pairs;
    pairs.reserve(results.size() * std::max((size_t)1, tracks.size()));
    for (size_t d = 0; d < results.size(); ++d)
    {
        const auto& db = results[d].bbox;
        for (size_t t = 0; t < tracks.size(); ++t)
        {
            const float v = bbox_iou(db, tracks[t].bbox);
            if (v >= CAPI_TRACK_IOU_THRESH)
                pairs.push_back({(int)d, (int)t, v});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b)
    {
        return a.iou > b.iou;
    });

    std::vector<char> det_taken(results.size(), 0);
    std::vector<char> track_taken(tracks.size(), 0);
    for (const auto& pair : pairs)
    {
        if (det_taken[pair.det] || track_taken[pair.track]) continue;
        det_taken[pair.det] = 1;
        track_taken[pair.track] = 1;
        assignment.track_ids[pair.det] = tracks[pair.track].id;
        tracks[pair.track].bbox = results[pair.det].bbox;
        tracks[pair.track].last_seen_frame = frame_index;
    }

    for (size_t d = 0; d < results.size(); ++d)
    {
        if (det_taken[d]) continue;
        CapiTrack track;
        track.id = next_track_id++;
        track.bbox = results[d].bbox;
        track.last_seen_frame = frame_index;
        tracks.push_back(track);
        assignment.track_ids[d] = track.id;
    }

    auto it = std::remove_if(tracks.begin(), tracks.end(), [frame_index](const CapiTrack& track)
    {
        return (frame_index - track.last_seen_frame) > CAPI_TRACK_MAX_MISSING;
    });
    tracks.erase(it, tracks.end());

    for (const auto& track : tracks)
    {
        if (track.last_seen_frame != frame_index)
            assignment.pad_ids.push_back(track.id);
    }
    return assignment;
}

static ButterworthState make_butterworth_state(const FsbVideoConfig* cfg, double source_fps)
{
    ButterworthState state;
    state.requested = cfg && cfg->butterworth != 0;
    state.filter_root_rot = cfg && cfg->butterworth_root_rotation != 0;
    state.fps = source_fps > 0.0 ? (float)source_fps : 30.0f;
    state.cutoff = cfg && cfg->bw_cutoff > 0.0f ? cfg->bw_cutoff : 6.0f;
    state.rot_clamp_deg = cfg && cfg->rot_clamp_deg > 0.0f ? cfg->rot_clamp_deg : 0.0f;
    state.use_scalar = state.requested && state.cutoff < state.fps * 0.5f;
    return state;
}

static void init_person_filters(PersonFilters& pf, const ButterworthState& state)
{
    if (state.use_scalar)
    {
        for (auto& f : pf.kp3d)      initButterWorth(&f, state.fps, state.cutoff);
        for (auto& f : pf.body_pose) initButterWorth(&f, state.fps, state.cutoff);
        for (auto& f : pf.hand_pose) initButterWorth(&f, state.fps, state.cutoff);
        for (auto& f : pf.cam_t)     initButterWorth(&f, state.fps, state.cutoff);
    }
    if (state.filter_root_rot)
        init_quat_lpf(&pf.root_rot, state.fps, state.cutoff);
}

static void apply_butterworth(ButterworthState& state, std::vector<fsb::MHRResult>& results)
{
    if (!state.requested && !state.filter_root_rot)
        return;

    while (state.filters.size() < results.size())
    {
        state.filters.emplace_back();
        init_person_filters(state.filters.back(), state);
    }

    for (int i = 0; i < (int)results.size(); ++i)
    {
        auto& r = results[i];
        auto& pf = state.filters[i];

        if (state.use_scalar)
        {
            for (int k = 0; k < (int)r.keypoints_3d.size() && k < 70*3; ++k)
                r.keypoints_3d[k] = filter(&pf.kp3d[k], r.keypoints_3d[k]);

            for (int k = 0; k < (int)r.body_pose.size() && k < 133; ++k)
                r.body_pose[k] = filter(&pf.body_pose[k], r.body_pose[k]);

            for (int k = 0; k < (int)r.hand_pose.size() && k < 108; ++k)
                r.hand_pose[k] = filter(&pf.hand_pose[k], r.hand_pose[k]);

            for (int k = 0; k < 3; ++k)
                r.pred_cam_t[k] = filter(&pf.cam_t[k], r.pred_cam_t[k]);
        }

        if (state.filter_root_rot)
        {
            float in_q[4], out_q[4];
            euler_zyx_to_quat(r.global_rot[0], r.global_rot[1], r.global_rot[2], in_q);
            const float max_step_rad = state.rot_clamp_deg > 0.0f
                ? state.rot_clamp_deg * (3.14159265359f / 180.0f)
                : 0.0f;
            filter_quat(&pf.root_rot, in_q, max_step_rad, out_q);
            quat_to_euler_zyx(out_q, &r.global_rot[0], &r.global_rot[1], &r.global_rot[2]);
        }
    }
}

static fsb::PipelineConfig make_pipeline_config(const FsbConfig& cfg)
{
    fsb::PipelineConfig pc;
    if (cfg.onnx_dir)   pc.onnx_dir      = cfg.onnx_dir;
    if (cfg.gguf_path)  pc.gguf_path     = cfg.gguf_path;
    if (cfg.yolo_path)  pc.yolo_path     = cfg.yolo_path;
    pc.cuda_device      = cfg.cuda_device;
    pc.skip_body_model  = cfg.skip_body_model != 0;
    pc.person_thresh    = cfg.person_thresh;
    pc.person_nms_iou   = cfg.person_nms_iou;
    pc.max_persons      = cfg.max_persons;
    pc.focal_x          = cfg.focal_x;
    pc.focal_y          = cfg.focal_y;
    pc.principal_x      = cfg.principal_x;
    pc.principal_y      = cfg.principal_y;
    pc.zero_face_params = cfg.zero_face_params != 0;
    return pc;
}

static int load_pipeline(FsbContext* ctx, const fsb::PipelineConfig& pc)
{
    try
    {
        if (!ctx->pipeline.load(pc))
        {
            std::ostringstream message;
            message << "Pipeline load failed"
                    << " (onnx_dir='" << pc.onnx_dir
                    << "', backbone='" << pc.backbone_name
                    << "', gguf='" << pc.gguf_path
                    << "', yolo='" << pc.yolo_path
                    << "', cuda_device=" << pc.cuda_device
                    << ", fp16=" << (pc.use_fp16 ? "on" : "off")
                    << ", tensorrt=" << (pc.use_trt_ep ? "on" : "off")
                    << ")";
            set_error(ctx, message.str());
            return 0;
        }
        clear_error(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        set_error(ctx, e.what());
        return 0;
    }
    catch (...)
    {
        set_error(ctx, "Unknown exception while loading pipeline");
        return 0;
    }
}

} // namespace

extern "C" {

    FsbHandle fsb_create(void)
    {
        try
        {
            return static_cast<FsbHandle>(new FsbContext());
        }
        catch (...)
        {
            return nullptr;
        }
    }

    void fsb_destroy(FsbHandle h)
    {
        if (h)
        {
            auto* ctx = as_ctx(h);
            ctx->pipeline.free();
            delete ctx;
        }
    }

    int fsb_load(FsbHandle h, const FsbConfig* cfg)
    {
        if (!h || !cfg) return 0;
        auto* ctx = as_ctx(h);

        return load_pipeline(ctx, make_pipeline_config(*cfg));
    }

    int fsb_load_ex(FsbHandle h, const FsbConfigEx* cfg)
    {
        if (!h || !cfg || cfg->struct_size < offsetof(FsbConfigEx, backbone_name)) return 0;
        auto* ctx = as_ctx(h);

        fsb::PipelineConfig pc = make_pipeline_config(cfg->base);

        if (cfg->struct_size >= offsetof(FsbConfigEx, use_trt_ep) && cfg->backbone_name)
            pc.backbone_name = cfg->backbone_name;
        if (cfg->struct_size >= offsetof(FsbConfigEx, use_fp16))
            pc.use_trt_ep = cfg->use_trt_ep != 0;
        if (cfg->struct_size >= sizeof(FsbConfigEx))
            pc.use_fp16 = cfg->use_fp16 != 0;

        return load_pipeline(ctx, pc);
    }

    int fsb_result_version(void)
    {
        return FSB_RESULT_VERSION;
    }

    int fsb_is_loaded(FsbHandle h)
    {
        if (!h) return 0;
        auto* ctx = as_ctx(h);
        return ctx->pipeline.is_loaded() ? 1 : 0;
    }

    const char* fsb_last_error(FsbHandle h)
    {
        if (!h) return "Invalid FsbHandle";
        auto* ctx = as_ctx(h);
        return ctx->last_error.empty() ? nullptr : ctx->last_error.c_str();
    }

    int fsb_process_bgr(FsbHandle      h,
                        const uint8_t* bgr,
                        int            width,
                        int            height,
                        FsbResult*     results,
                        int            max_results)
    {
        if (!h || !bgr || !results || max_results <= 0) return -1;
        auto* ctx = as_ctx(h);

        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            std::vector<fsb::MHRResult> res = ctx->pipeline.process_bgr(bgr, width, height);
            int n = std::min((int)res.size(), max_results);

            for (int i = 0; i < n; ++i)
            {
                copy_result(results[i], res[i]);
            }
            clear_error(ctx);
            return n;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing BGR image");
            return -1;
        }
    }

    int fsb_process_bgr_mesh(FsbHandle        h,
                             const uint8_t*   bgr,
                             int              width,
                             int              height,
                             FsbResult*       results,
                             int*             vertex_counts,
                             float*           pred_vertices,
                             int              max_results,
                             int              vertex_float_capacity)
    {
        if (!h || !bgr || !results || max_results <= 0) return -1;
        auto* ctx = as_ctx(h);

        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            std::vector<fsb::MHRResult> res = ctx->pipeline.process_bgr(bgr, width, height);
            int n = std::min((int)res.size(), max_results);

            for (int i = 0; i < n; ++i)
            {
                copy_result(results[i], res[i]);
                if (vertex_counts)
                    vertex_counts[i] = 0;
                if (vertex_counts &&
                    pred_vertices &&
                    vertex_float_capacity >= FSB_MESH_FLOAT_COUNT &&
                    (int)res[i].pred_vertices.size() >= FSB_MESH_FLOAT_COUNT)
                {
                    vertex_counts[i] = FSB_MESH_VERTEX_COUNT;
                    std::memcpy(pred_vertices + (i * vertex_float_capacity),
                                res[i].pred_vertices.data(),
                                FSB_MESH_FLOAT_COUNT * sizeof(float));
                }
            }
            clear_error(ctx);
            return n;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing BGR image");
            return -1;
        }
    }

    int fsb_process_image_bvh(FsbHandle  h,
                              const char* image_path,
                              const char* bvh_path,
                              const char* bvh_template,
                              const char* lbs_path,
                              int         max_results)
    {
        if (!h || !image_path || !image_path[0] || !bvh_path || !bvh_path[0])
            return -1;

        auto* ctx = as_ctx(h);
        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }
        if (!bvh_template || !bvh_template[0])
        {
            set_error(ctx, "BVH template path is required for BVH export");
            return -1;
        }
        if (!lbs_path || !lbs_path[0])
        {
            set_error(ctx, "body_model.lbs path is required for BVH export");
            return -1;
        }

        try
        {
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            if (image.empty())
            {
                set_error(ctx, std::string("Could not read image: ") + image_path);
                return -1;
            }
            if (!image.isContinuous())
                image = image.clone();

            std::vector<fsb::MHRResult> res =
                ctx->pipeline.process_bgr(image.data, image.cols, image.rows);
            const int limit = max_results > 0 ? std::min(max_results, 64) : 64;
            if ((int)res.size() > limit)
                res.resize(limit);

            if (res.empty())
            {
                clear_error(ctx);
                return 0;
            }

            BVHWriter bvh_writer;
            if (!bvh_writer.open(bvh_template,
                                 bvh_path,
                                 1.0f / 30.0f,
                                 lbs_path,
                                 true,
                                 true,
                                 true,
                                 false,
                                 false,
                                 false))
            {
                set_error(ctx, "BVH writer failed to open");
                return -1;
            }
            bvh_writer.write_frame(res);
            bvh_writer.close();
            clear_error(ctx);
            return (int)res.size();
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while exporting image BVH");
            return -1;
        }
    }

    int fsb_process_video(FsbHandle                 h,
                          const FsbVideoConfig*     cfg,
                          FsbVideoSummary*          summary,
                          FsbVideoProgressCallback  progress_cb,
                          void*                     user)
    {
        return fsb_process_video_mesh(h, cfg, summary, progress_cb, nullptr, user);
    }

    int fsb_process_video_mesh(FsbHandle                 h,
                               const FsbVideoConfig*     cfg,
                               FsbVideoSummary*          summary,
                               FsbVideoProgressCallback  progress_cb,
                               FsbVideoMeshCallback      mesh_cb,
                               void*                     user)
    {
        if (!summary) return -1;
        std::memset(summary, 0, sizeof(FsbVideoSummary));

        if (!h || !cfg || !cfg->input_path)
        {
            return -1;
        }

        auto* ctx = as_ctx(h);
        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            cv::VideoCapture cap;
            cap.open(cfg->input_path, cv::CAP_FFMPEG);
            if (!cap.isOpened())
                cap.open(cfg->input_path);
            if (!cap.isOpened())
            {
                set_error(ctx, std::string("Could not open video: ") + cfg->input_path);
                return -1;
            }

            const bool write_csv = cfg->csv_path && cfg->csv_path[0];
            const bool write_bvh = cfg->bvh_path && cfg->bvh_path[0];
            if (!write_csv && !write_bvh)
            {
                set_error(ctx, "Set a BVH output path or a CSV output path");
                return -1;
            }

            std::ofstream csv;
            if (write_csv)
            {
                csv.open(cfg->csv_path);
                if (!csv)
                {
                    set_error(ctx, std::string("Could not open CSV for writing: ") + cfg->csv_path);
                    return -1;
                }
                write_video_csv_header(csv);
            }

            int stride = cfg->frame_stride > 0 ? cfg->frame_stride : 1;
            int max_frames = cfg->max_frames > 0 ? cfg->max_frames : 0;
            int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
            summary->source_fps = cap.get(cv::CAP_PROP_FPS);
            const float frame_time = summary->source_fps > 0.0
                                   ? (float)(1.0 / summary->source_fps)
                                   : (1.0f / 30.0f);
            ButterworthState bw_state = make_butterworth_state(cfg, summary->source_fps);

            BVHWriter bvh_writer;
            if (write_bvh)
            {
                if (!cfg->bvh_template || !cfg->bvh_template[0])
                {
                    set_error(ctx, "BVH template path is required for BVH export");
                    return -1;
                }
                if (!cfg->lbs_path || !cfg->lbs_path[0])
                {
                    set_error(ctx, "body_model.lbs path is required for BVH export");
                    return -1;
                }
                if (!bvh_writer.open(cfg->bvh_template,
                                     cfg->bvh_path,
                                     frame_time,
                                     cfg->lbs_path,
                                     true,
                                     true,
                                     true,
                                     false,
                                     false,
                                     false))
                {
                    set_error(ctx, "BVH writer failed to open");
                    return -1;
                }
            }

            auto total_t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            while (cap.read(frame))
            {
                ++summary->frames_read;
                if (max_frames > 0 && summary->frames_processed >= max_frames)
                    break;
                if ((summary->frames_read - 1) % stride != 0)
                    continue;

                if (frame.empty())
                    continue;
                if (frame.channels() == 4)
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                else if (frame.channels() == 1)
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                if (!frame.isContinuous())
                    frame = frame.clone();

                auto frame_t0 = std::chrono::steady_clock::now();
                std::vector<fsb::MHRResult> res =
                    ctx->pipeline.process_bgr(frame.data, frame.cols, frame.rows);
                if (cfg->max_results > 0 && (int)res.size() > cfg->max_results)
                    res.resize(cfg->max_results);
                apply_butterworth(bw_state, res);
                float frame_ms = (float)elapsed_ms(frame_t0);

                if (write_csv)
                    write_video_csv_rows(csv, summary->frames_read - 1, res);
                if (bvh_writer.is_open())
                    bvh_writer.write_frame(res);
                if (mesh_cb)
                {
                    for (int i = 0; i < (int)res.size(); ++i)
                    {
                        if ((int)res[i].pred_vertices.size() >= FSB_MESH_FLOAT_COUNT)
                        {
                            mesh_cb(summary->frames_read - 1,
                                    i,
                                    FSB_MESH_VERTEX_COUNT,
                                    res[i].pred_vertices.data(),
                                    user);
                        }
                    }
                }
                ++summary->frames_processed;
                summary->persons_total += (int)res.size();

                if (progress_cb)
                {
                    progress_cb(summary->frames_read - 1, total_frames,
                                (int)res.size(), frame_ms, user);
                }
            }

            summary->total_ms = elapsed_ms(total_t0);
            if (bvh_writer.is_open())
                bvh_writer.close();
            clear_error(ctx);
            return 1;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing video");
            return -1;
        }
    }

    int fsb_process_video_mesh_ex(FsbHandle                 h,
                                  const FsbVideoConfig*     cfg,
                                  FsbVideoSummary*          summary,
                                  FsbVideoProgressCallback  progress_cb,
                                  FsbVideoMeshExCallback    mesh_cb,
                                  void*                     user)
    {
        if (!summary) return -1;
        std::memset(summary, 0, sizeof(FsbVideoSummary));

        if (!h || !cfg || !cfg->input_path)
        {
            return -1;
        }

        auto* ctx = as_ctx(h);
        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            cv::VideoCapture cap;
            cap.open(cfg->input_path, cv::CAP_FFMPEG);
            if (!cap.isOpened())
                cap.open(cfg->input_path);
            if (!cap.isOpened())
            {
                set_error(ctx, std::string("Could not open video: ") + cfg->input_path);
                return -1;
            }

            const bool write_csv = cfg->csv_path && cfg->csv_path[0];
            const bool write_bvh = cfg->bvh_path && cfg->bvh_path[0];
            if (!write_csv && !write_bvh)
            {
                set_error(ctx, "Set a BVH output path or a CSV output path");
                return -1;
            }

            std::ofstream csv;
            if (write_csv)
            {
                csv.open(cfg->csv_path);
                if (!csv)
                {
                    set_error(ctx, std::string("Could not open CSV for writing: ") + cfg->csv_path);
                    return -1;
                }
                write_video_csv_header(csv);
            }

            int stride = cfg->frame_stride > 0 ? cfg->frame_stride : 1;
            int max_frames = cfg->max_frames > 0 ? cfg->max_frames : 0;
            int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
            summary->source_fps = cap.get(cv::CAP_PROP_FPS);
            const float frame_time = summary->source_fps > 0.0
                                   ? (float)(1.0 / summary->source_fps)
                                   : (1.0f / 30.0f);
            ButterworthState bw_state = make_butterworth_state(cfg, summary->source_fps);

            BVHWriter bvh_writer;
            if (write_bvh)
            {
                if (!cfg->bvh_template || !cfg->bvh_template[0])
                {
                    set_error(ctx, "BVH template path is required for BVH export");
                    return -1;
                }
                if (!cfg->lbs_path || !cfg->lbs_path[0])
                {
                    set_error(ctx, "body_model.lbs path is required for BVH export");
                    return -1;
                }
                if (!bvh_writer.open(cfg->bvh_template,
                                     cfg->bvh_path,
                                     frame_time,
                                     cfg->lbs_path,
                                     true,
                                     true,
                                     true,
                                     false,
                                     false,
                                     false))
                {
                    set_error(ctx, "BVH writer failed to open");
                    return -1;
                }
            }

            auto total_t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            while (cap.read(frame))
            {
                ++summary->frames_read;
                if (max_frames > 0 && summary->frames_processed >= max_frames)
                    break;
                if ((summary->frames_read - 1) % stride != 0)
                    continue;

                if (frame.empty())
                    continue;
                if (frame.channels() == 4)
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                else if (frame.channels() == 1)
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                if (!frame.isContinuous())
                    frame = frame.clone();

                auto frame_t0 = std::chrono::steady_clock::now();
                std::vector<fsb::MHRResult> res =
                    ctx->pipeline.process_bgr(frame.data, frame.cols, frame.rows);
                if (cfg->max_results > 0 && (int)res.size() > cfg->max_results)
                    res.resize(cfg->max_results);
                apply_butterworth(bw_state, res);
                float frame_ms = (float)elapsed_ms(frame_t0);

                if (write_csv)
                    write_video_csv_rows(csv, summary->frames_read - 1, res);
                if (bvh_writer.is_open())
                    bvh_writer.write_frame(res);
                if (mesh_cb)
                {
                    for (int i = 0; i < (int)res.size(); ++i)
                    {
                        if ((int)res[i].pred_vertices.size() >= FSB_MESH_FLOAT_COUNT)
                        {
                            const int kps_count = (int)res[i].keypoints_3d.size() / 3;
                            mesh_cb(summary->frames_read - 1,
                                    i,
                                    frame.cols,
                                    frame.rows,
                                    res[i].bbox.data(),
                                    res[i].pred_cam_t.data(),
                                    kps_count,
                                    kps_count > 0 ? res[i].keypoints_3d.data() : nullptr,
                                    FSB_MESH_VERTEX_COUNT,
                                    res[i].pred_vertices.data(),
                                    user);
                        }
                    }
                }
                ++summary->frames_processed;
                summary->persons_total += (int)res.size();

                if (progress_cb)
                {
                    progress_cb(summary->frames_read - 1, total_frames,
                                (int)res.size(), frame_ms, user);
                }
            }

            summary->total_ms = elapsed_ms(total_t0);
            if (bvh_writer.is_open())
                bvh_writer.close();
            clear_error(ctx);
            return 1;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing video");
            return -1;
        }
    }

    int fsb_process_video_mesh_tracked(FsbHandle                    h,
                                       const FsbVideoConfig*        cfg,
                                       FsbVideoSummary*             summary,
                                       FsbVideoProgressCallback     progress_cb,
                                       FsbVideoMeshTrackedCallback  mesh_cb,
                                       void*                        user)
    {
        if (!summary) return -1;
        std::memset(summary, 0, sizeof(FsbVideoSummary));

        if (!h || !cfg || !cfg->input_path)
            return -1;

        auto* ctx = as_ctx(h);
        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            cv::VideoCapture cap;
            cap.open(cfg->input_path, cv::CAP_FFMPEG);
            if (!cap.isOpened())
                cap.open(cfg->input_path);
            if (!cap.isOpened())
            {
                set_error(ctx, std::string("Could not open video: ") + cfg->input_path);
                return -1;
            }

            const bool write_csv = cfg->csv_path && cfg->csv_path[0];
            const bool write_bvh = cfg->bvh_path && cfg->bvh_path[0];
            if (!write_csv && !write_bvh)
            {
                set_error(ctx, "Set a BVH output path or a CSV output path");
                return -1;
            }

            std::ofstream csv;
            if (write_csv)
            {
                csv.open(cfg->csv_path);
                if (!csv)
                {
                    set_error(ctx, std::string("Could not open CSV for writing: ") + cfg->csv_path);
                    return -1;
                }
                write_video_csv_header(csv);
            }

            int stride = cfg->frame_stride > 0 ? cfg->frame_stride : 1;
            int max_frames = cfg->max_frames > 0 ? cfg->max_frames : 0;
            int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
            summary->source_fps = cap.get(cv::CAP_PROP_FPS);
            const float frame_time = summary->source_fps > 0.0
                                   ? (float)(1.0 / summary->source_fps)
                                   : (1.0f / 30.0f);
            ButterworthState bw_state = make_butterworth_state(cfg, summary->source_fps);
            std::vector<CapiTrack> tracks;
            int next_track_id = 0;
            int processed_session_frame = 0;

            BVHWriter bvh_writer;
            if (write_bvh)
            {
                if (!cfg->bvh_template || !cfg->bvh_template[0])
                {
                    set_error(ctx, "BVH template path is required for BVH export");
                    return -1;
                }
                if (!cfg->lbs_path || !cfg->lbs_path[0])
                {
                    set_error(ctx, "body_model.lbs path is required for BVH export");
                    return -1;
                }
                if (!bvh_writer.open(cfg->bvh_template,
                                     cfg->bvh_path,
                                     frame_time,
                                     cfg->lbs_path,
                                     true,
                                     true,
                                     true,
                                     false,
                                     false,
                                     false))
                {
                    set_error(ctx, "BVH writer failed to open");
                    return -1;
                }
            }

            auto total_t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            while (cap.read(frame))
            {
                ++summary->frames_read;
                if (max_frames > 0 && summary->frames_processed >= max_frames)
                    break;
                if ((summary->frames_read - 1) % stride != 0)
                    continue;

                if (frame.empty())
                    continue;
                if (frame.channels() == 4)
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                else if (frame.channels() == 1)
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                if (!frame.isContinuous())
                    frame = frame.clone();

                auto frame_t0 = std::chrono::steady_clock::now();
                std::vector<fsb::MHRResult> raw =
                    ctx->pipeline.process_bgr(frame.data, frame.cols, frame.rows);
                if (cfg->max_results > 0 && (int)raw.size() > cfg->max_results)
                    raw.resize(cfg->max_results);
                apply_butterworth(bw_state, raw);

                std::vector<fsb::MHRResult> res;
                res.reserve(raw.size());
                for (const auto& r : raw)
                {
                    if (bbox_looks_valid(r.bbox))
                        res.push_back(r);
                }

                TrackAssignment assignment =
                    assign_capi_tracks(res, tracks, next_track_id, processed_session_frame);

                float frame_ms = (float)elapsed_ms(frame_t0);

                if (write_csv)
                    write_video_csv_rows(csv, summary->frames_read - 1, res);
                if (bvh_writer.is_open())
                    bvh_writer.write_frame_external(res, assignment.track_ids, assignment.pad_ids);
                if (mesh_cb)
                {
                    for (int i = 0; i < (int)res.size(); ++i)
                    {
                        if ((int)res[i].pred_vertices.size() >= FSB_MESH_FLOAT_COUNT)
                        {
                            const int kps_count = (int)res[i].keypoints_3d.size() / 3;
                            mesh_cb(summary->frames_read - 1,
                                    i,
                                    i < (int)assignment.track_ids.size() ? assignment.track_ids[i] : i,
                                    frame.cols,
                                    frame.rows,
                                    res[i].bbox.data(),
                                    res[i].pred_cam_t.data(),
                                    kps_count,
                                    kps_count > 0 ? res[i].keypoints_3d.data() : nullptr,
                                    FSB_MESH_VERTEX_COUNT,
                                    res[i].pred_vertices.data(),
                                    user);
                        }
                    }
                }
                ++summary->frames_processed;
                ++processed_session_frame;
                summary->persons_total += (int)res.size();

                if (progress_cb)
                {
                    progress_cb(summary->frames_read - 1, total_frames,
                                (int)res.size(), frame_ms, user);
                }
            }

            summary->total_ms = elapsed_ms(total_t0);
            if (bvh_writer.is_open())
                bvh_writer.close();
            clear_error(ctx);
            return 1;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing tracked video");
            return -1;
        }
    }

    int fsb_process_video_mesh_overlay(FsbHandle                    h,
                                       const FsbVideoConfig*        cfg,
                                       FsbVideoSummary*             summary,
                                       FsbVideoProgressCallback     progress_cb,
                                       FsbVideoMeshOverlayCallback  mesh_cb,
                                       void*                        user)
    {
        if (!summary) return -1;
        std::memset(summary, 0, sizeof(FsbVideoSummary));

        if (!h || !cfg || !cfg->input_path)
            return -1;

        auto* ctx = as_ctx(h);
        if (!ctx->pipeline.is_loaded())
        {
            set_error(ctx, "Pipeline is not loaded");
            return -1;
        }

        try
        {
            cv::VideoCapture cap;
            cap.open(cfg->input_path, cv::CAP_FFMPEG);
            if (!cap.isOpened())
                cap.open(cfg->input_path);
            if (!cap.isOpened())
            {
                set_error(ctx, std::string("Could not open video: ") + cfg->input_path);
                return -1;
            }

            const bool write_csv = cfg->csv_path && cfg->csv_path[0];
            const bool write_bvh = cfg->bvh_path && cfg->bvh_path[0];
            if (!write_csv && !write_bvh)
            {
                set_error(ctx, "Set a BVH output path or a CSV output path");
                return -1;
            }

            std::ofstream csv;
            if (write_csv)
            {
                csv.open(cfg->csv_path);
                if (!csv)
                {
                    set_error(ctx, std::string("Could not open CSV for writing: ") + cfg->csv_path);
                    return -1;
                }
                write_video_csv_header(csv);
            }

            int stride = cfg->frame_stride > 0 ? cfg->frame_stride : 1;
            int max_frames = cfg->max_frames > 0 ? cfg->max_frames : 0;
            int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
            summary->source_fps = cap.get(cv::CAP_PROP_FPS);
            const float frame_time = summary->source_fps > 0.0
                                   ? (float)(1.0 / summary->source_fps)
                                   : (1.0f / 30.0f);
            ButterworthState bw_state = make_butterworth_state(cfg, summary->source_fps);
            std::vector<CapiTrack> tracks;
            int next_track_id = 0;
            int processed_session_frame = 0;

            BVHWriter bvh_writer;
            if (write_bvh)
            {
                if (!cfg->bvh_template || !cfg->bvh_template[0])
                {
                    set_error(ctx, "BVH template path is required for BVH export");
                    return -1;
                }
                if (!cfg->lbs_path || !cfg->lbs_path[0])
                {
                    set_error(ctx, "body_model.lbs path is required for BVH export");
                    return -1;
                }
                if (!bvh_writer.open(cfg->bvh_template,
                                     cfg->bvh_path,
                                     frame_time,
                                     cfg->lbs_path,
                                     true,
                                     true,
                                     true,
                                     false,
                                     false,
                                     false))
                {
                    set_error(ctx, "BVH writer failed to open");
                    return -1;
                }
            }

            auto total_t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            while (cap.read(frame))
            {
                ++summary->frames_read;
                if (max_frames > 0 && summary->frames_processed >= max_frames)
                    break;
                if ((summary->frames_read - 1) % stride != 0)
                    continue;

                if (frame.empty())
                    continue;
                if (frame.channels() == 4)
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                else if (frame.channels() == 1)
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                if (!frame.isContinuous())
                    frame = frame.clone();

                auto frame_t0 = std::chrono::steady_clock::now();
                std::vector<fsb::MHRResult> raw =
                    ctx->pipeline.process_bgr(frame.data, frame.cols, frame.rows);
                if (cfg->max_results > 0 && (int)raw.size() > cfg->max_results)
                    raw.resize(cfg->max_results);
                apply_butterworth(bw_state, raw);

                std::vector<fsb::MHRResult> res;
                res.reserve(raw.size());
                for (const auto& r : raw)
                {
                    if (bbox_looks_valid(r.bbox))
                        res.push_back(r);
                }

                TrackAssignment assignment =
                    assign_capi_tracks(res, tracks, next_track_id, processed_session_frame);

                float frame_ms = (float)elapsed_ms(frame_t0);

                if (write_csv)
                    write_video_csv_rows(csv, summary->frames_read - 1, res);
                if (bvh_writer.is_open())
                    bvh_writer.write_frame_external(res, assignment.track_ids, assignment.pad_ids);
                if (mesh_cb)
                {
                    for (int i = 0; i < (int)res.size(); ++i)
                    {
                        if ((int)res[i].pred_vertices.size() >= FSB_MESH_FLOAT_COUNT)
                        {
                            const int kps_count = (int)res[i].keypoints_3d.size() / 3;
                            const int kps2d_count = (int)res[i].keypoints_2d.size() / 2;
                            mesh_cb(summary->frames_read - 1,
                                    i,
                                    i < (int)assignment.track_ids.size() ? assignment.track_ids[i] : i,
                                    frame.cols,
                                    frame.rows,
                                    res[i].bbox.data(),
                                    res[i].pred_cam_t.data(),
                                    kps_count,
                                    kps_count > 0 ? res[i].keypoints_3d.data() : nullptr,
                                    kps2d_count,
                                    kps2d_count > 0 ? res[i].keypoints_2d.data() : nullptr,
                                    FSB_MESH_VERTEX_COUNT,
                                    res[i].pred_vertices.data(),
                                    user);
                        }
                    }
                }
                ++summary->frames_processed;
                ++processed_session_frame;
                summary->persons_total += (int)res.size();

                if (progress_cb)
                {
                    progress_cb(summary->frames_read - 1, total_frames,
                                (int)res.size(), frame_ms, user);
                }
            }

            summary->total_ms = elapsed_ms(total_t0);
            if (bvh_writer.is_open())
                bvh_writer.close();
            clear_error(ctx);
            return 1;
        }
        catch (const std::exception& e)
        {
            set_error(ctx, e.what());
            return -1;
        }
        catch (...)
        {
            set_error(ctx, "Unknown exception while processing overlay video");
            return -1;
        }
    }

} // extern "C"
