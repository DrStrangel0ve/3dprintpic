from __future__ import annotations

from copy import deepcopy

from .benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_MODEL_REVISION,
)
from .benchmark.trellis2_models import (
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
)
from .benchmark.triposg_models import (
    DEFAULT_TRIPOSG_MODEL,
    DEFAULT_TRIPOSG_MODEL_REVISION,
    DEFAULT_TRIPOSG_REMBG_MODEL,
    DEFAULT_TRIPOSG_REMBG_REVISION,
)


MODEL_PROFILE_SCHEMA_VERSION = 1
DEFAULT_MODEL_PROFILE_ID = "verified-print-v1"


MODEL_PROFILES = (
    {
        "id": DEFAULT_MODEL_PROFILE_ID,
        "label": "Recommended print",
        "status": "recommended",
        "description": (
            "Measured defaults for printable photo, relief, and multiview STL routes. "
            "Advanced controls can override any setting per run."
        ),
        "models": {
            "selection": "detr-resnet-50-panoptic",
            "depth": "depth-anything/Depth-Anything-V2-Large-hf",
            "image_to_mesh": "triposg",
            "video_reconstruction": "multiview-visual-hull",
            "stl_postprocess": "trimesh-repair",
        },
        "production_features": {
            "photo_object_selection": {
                "model": "facebook/detr-resnet-50-panoptic",
                "runtime_id": "detr-resnet-50-panoptic",
                "reason": (
                    "It is the locally configured click-to-segment path with cached panoptic "
                    "precomputation; alternatives remain benchmark-only until they have equivalent "
                    "offline readiness and regression coverage."
                ),
                "evidence": "backend/tests/test_main_stl_contract.py",
            },
            "photo_relief_depth": {
                "model": "depth-anything/Depth-Anything-V2-Large-hf",
                "runtime_id": "depth-anything/Depth-Anything-V2-Large-hf",
                "reason": (
                    "It is the verified CUDA-backed depth source used by the 20/30/40 mm relief, "
                    "face-detail, background-retention, and exact-shell regression matrix."
                ),
                "evidence": "docs/relief-face-context-v2.md",
            },
            "known_mask_prefill": {
                "model": "bounded biharmonic interpolation",
                "runtime_id": "biharmonic-prefill",
                "reason": (
                    "The promoted TripoSG lane used biharmonic prefill and won all ten paired "
                    "comparisons while every per-sample printability gate passed."
                ),
                "evidence": (
                    "docs/benchmark-evidence/"
                    "g4_stl_first_triposg_inferred_adaptive_s40_n10"
                ),
            },
            "single_image_mesh": {
                "model": DEFAULT_TRIPOSG_MODEL,
                "runtime_id": "triposg",
                "revision": DEFAULT_TRIPOSG_MODEL_REVISION,
                "reason": (
                    "It is the only measured single-image challenger promoted on the held-out "
                    "ten-object STL-quality slice with zero failed checks."
                ),
                "evidence": (
                    "docs/benchmark-evidence/"
                    "g4_stl_first_triposg_inferred_adaptive_s40_n10"
                ),
            },
            "turntable_segmentation": {
                "model": "OpenCV GrabCut with temporal mask prior",
                "runtime_id": "turntable-grabcut",
                "reason": (
                    "It is deterministic, offline, and directly covered by the controlled "
                    "turntable-video mask and STL regressions."
                ),
                "evidence": "docs/video-to-stl.md",
            },
            "tracked_video_segmentation": {
                "model": "facebook/sam2.1-hiera-tiny",
                "runtime_id": "sam2.1-hiera-tiny-video",
                "reason": (
                    "It is the attached temporal propagation route that fits the local GPU budget; "
                    "larger and gated video segmenters remain setup candidates."
                ),
                "evidence": "backend/tests/test_video_subject_relief.py",
            },
            "video_frame_selection": {
                "model": "OpenCV Laplacian and optical-flow selector",
                "runtime_id": "sharpness-motion-selector",
                "reason": (
                    "Tracked relief needs focus and motion coverage; controlled turntable mesh uses "
                    "the single deterministic uniform-frame assignment for angular coverage."
                ),
                "evidence": "backend/benchmark/run_video_pipeline_smoke.py",
            },
            "turntable_reconstruction": {
                "model": "silhouette visual hull",
                "runtime_id": "multiview-visual-hull",
                "reason": (
                    "Resolution 32 was the best measured deployable multiview lane and passed every "
                    "hard STL gate on the ten-object procedural slice."
                ),
                "evidence": "docs/stl-first-architecture.md",
            },
            "stl_repair": {
                "model": "trimesh",
                "runtime_id": "trimesh-repair",
                "reason": (
                    "The shared repair and diagnostics path is the one exercised by watertightness, "
                    "manifoldness, winding, volume, component, and complexity promotion gates."
                ),
                "evidence": "backend/benchmark/direct_mesh.py",
            },
        },
        "routes": {
            "photo-relief": {
                "settings": {
                    "depth_provider": "transformers",
                    "depth_model": "depth-anything/Depth-Anything-V2-Large-hf",
                    "minimum_samples_per_feature": 2.0,
                    "max_relief_slope": 2.0,
                },
                "evidence_status": "operational-default",
            },
            "photo-full-mesh": {
                "settings": {
                    "provider": "triposg",
                    "mesh_repair": "printable",
                    "mesh_repair_preconditioner": "adaptive-voxel-close",
                    "mesh_repair_voxel_resolution": 128,
                    "mesh_repair_voxel_fill_method": "orthographic",
                    "mesh_repair_smoothing_iterations": 2,
                    "mesh_allow_convex_hull_fallback": False,
                    "mesh_target_max_dimension": 96.0,
                    "mesh_min_bbox_dimension": 0.0,
                    "mesh_max_bbox_aspect_ratio": 0.0,
                    "mesh_target_bbox_mode": "uniform-max",
                    "mesh_target_faces": 10000,
                    "mesh_max_normalized_face_density_log1p": 9.95,
                    "low_vram": True,
                    "chunk_size": 8192,
                    "mc_resolution": 256,
                    "texture_resolution": None,
                    "num_inference_steps": 50,
                    "guidance_scale": 7.0,
                    "octree_resolution": 256,
                    "num_chunks": 8000,
                    "seed": None,
                    "disable_progress": True,
                    "triposg_model_revision": DEFAULT_TRIPOSG_MODEL_REVISION,
                    "triposg_rembg_revision": DEFAULT_TRIPOSG_REMBG_REVISION,
                },
                "preprocessing_recommendation": (
                    "Use biharmonic prefill when a known missing-region mask exists; otherwise preserve "
                    "the selected object and neutralize only its background."
                ),
                "evidence_status": "verified-benchmark-core",
            },
            "multiview-full-mesh": {
                "settings": {
                    "provider": "multiview-visual-hull",
                    "mesh_repair": "printable",
                    "mesh_target_max_dimension": 96.0,
                    "mesh_min_bbox_dimension": 12.0,
                    "mesh_max_bbox_aspect_ratio": 0.0,
                    "mesh_target_faces": 40000,
                    "visual_hull_resolution": 32,
                    "visual_hull_grid_extent": 1.9,
                    "visual_hull_ortho_scale": 2.0,
                    "visual_hull_mask_dilate": 1,
                },
                "evidence_status": "verified",
            },
        },
        "model_revisions": {
            "image_to_mesh": {
                "repo_id": DEFAULT_TRIPOSG_MODEL,
                "revision": DEFAULT_TRIPOSG_MODEL_REVISION,
            },
            "foreground_removal": {
                "repo_id": DEFAULT_TRIPOSG_REMBG_MODEL,
                "revision": DEFAULT_TRIPOSG_REMBG_REVISION,
            },
        },
        "evidence": {
            "single_image": {
                "dataset": "ModelNet10 held-out rows 40-49",
                "sample_count": 10,
                "paired_wins_vs_mirror": 10,
                "rank_score": 1.1038838312237167,
                "mesh_surface_chamfer_l1_median": 0.15931294523185047,
                "mesh_surface_hausdorff95_median": 0.3849048861652087,
                "stl_faces_median": 3855.0,
                "all_printability_gates_passed": True,
                "source": "docs/benchmark-evidence/g4_stl_first_triposg_inferred_adaptive_s40_n10",
            },
            "multiview": {
                "dataset": "10-object procedural multiview set",
                "sample_count": 10,
                "tested_resolutions": [32, 36, 40],
                "selected_resolution": 32,
                "rank_score": 1.1759648962533198,
                "mesh_surface_chamfer_l1_median": 0.07696597368348078,
                "mesh_surface_hausdorff95_median": 0.2148467724667556,
                "stl_faces_median": 6508.0,
                "all_printability_gates_passed": True,
                "source": (
                    "backend/output/completion-benchmark/experiments/"
                    "stl_first_visual_hull_local_s0_n10_res32"
                ),
            },
        },
        "limitations": [
            (
                "The winning benchmark used a deployable depth-relief bounding-box prepass. The live "
                "image runner currently uses minimum-dimension and aspect safeguards instead."
            ),
            (
                "Biharmonic completion is applicable only when the client supplies a known missing-region "
                "mask; ordinary photos are not automatically treated as half-missing images."
            ),
        ],
    },
)


MODEL_CANDIDATES = (
    {
        "id": "pixal3d-detail",
        "provider": "pixal3d",
        "model": DEFAULT_PIXAL3D_MODEL,
        "revision": DEFAULT_PIXAL3D_MODEL_REVISION,
        "status": "provisional",
        "reason": (
            "The repaired STL passed printability checks on one object, but surface Chamfer and H95 "
            "regressed against TripoSG. It needs the same paired 10-object evaluation before promotion."
        ),
        "sample_count": 1,
    },
    {
        "id": "trellis2-detail",
        "provider": "trellis2",
        "model": DEFAULT_TRELLIS2_MODEL,
        "revision": DEFAULT_TRELLIS2_MODEL_REVISION,
        "source_revision": DEFAULT_TRELLIS2_SOURCE_REVISION,
        "status": "unbenchmarked",
        "reason": "The pinned geometry adapter is ready, but no paired STL-quality result is recorded yet.",
        "sample_count": 0,
    },
    {
        "id": "hunyuan3d-shape",
        "provider": "hunyuan3d-shape",
        "model": "tencent/Hunyuan3D-2.1",
        "status": "held",
        "reason": "The recovered 10-object run produced zero successful candidate meshes.",
        "sample_count": 10,
        "success_count": 0,
    },
)


_PROFILE_INDEX = {profile["id"]: profile for profile in MODEL_PROFILES}


def resolve_model_profile(profile_id: str | None = None) -> dict:
    selected_id = str(profile_id or DEFAULT_MODEL_PROFILE_ID).strip() or DEFAULT_MODEL_PROFILE_ID
    try:
        return deepcopy(_PROFILE_INDEX[selected_id])
    except KeyError as exc:
        valid = ", ".join(sorted(_PROFILE_INDEX))
        raise KeyError(f"Unknown model profile '{selected_id}'. Valid: {valid}") from exc


def model_profile_catalog() -> dict:
    return {
        "schema_version": MODEL_PROFILE_SCHEMA_VERSION,
        "default_profile": DEFAULT_MODEL_PROFILE_ID,
        "profiles": deepcopy(list(MODEL_PROFILES)),
        "candidates": deepcopy(list(MODEL_CANDIDATES)),
    }


def route_settings(profile: dict, route: str) -> dict:
    try:
        return deepcopy(profile["routes"][route]["settings"])
    except KeyError as exc:
        raise KeyError(f"Model profile '{profile.get('id', '<unknown>')}' has no route '{route}'") from exc
