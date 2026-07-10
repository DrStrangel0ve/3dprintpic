import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI(title="3D Print Pic Video and Selection Planner")

DEFAULT_LOCAL_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
CORS_ORIGINS = list(
    dict.fromkeys(
        origin.strip()
        for origin in [
            *DEFAULT_LOCAL_ORIGINS,
            *os.getenv("CORS_ORIGINS", "").split(","),
            *os.getenv("VIDEO_CORS_ORIGINS", "").split(","),
        ]
        if origin.strip()
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_GROUPS = {
    "selection": [
        {
            "id": "sam2.1-hiera-large",
            "label": "SAM 2.1 Hiera Large",
            "model": "facebook/sam2.1-hiera-large",
            "role": "promptable object and video mask propagation",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Primary interactive selection model for selected-frame and object-selection routes.",
        },
        {
            "id": "sam2.1-hiera-base-plus",
            "label": "SAM 2.1 Hiera Base+",
            "model": "facebook/sam2.1-hiera-base-plus",
            "role": "faster promptable mask propagation",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Lower-latency option when the large checkpoint is too slow.",
        },
        {
            "id": "grounding-dino-sam2",
            "label": "Grounding DINO + SAM 2.1",
            "model": "IDEA-Research/GroundingDINO + facebook/sam2.1",
            "role": "text-prompted object box plus mask",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Useful when the user names an object instead of clicking it.",
        },
        {
            "id": "rmbg-2.0",
            "label": "RMBG 2.0",
            "model": "briaai/RMBG-2.0",
            "role": "automatic foreground matte",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast fallback for product-style foreground/background separation.",
        },
    ],
    "frame_selection": [
        {
            "id": "uniform-frame-sampler",
            "label": "Uniform frame sampler",
            "model": "opencv-videoio",
            "role": "deterministic every-n-frame sampling",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Baseline sampler for full-video routes.",
        },
        {
            "id": "scenedetect-adaptive",
            "label": "PySceneDetect adaptive",
            "model": "scenedetect-adaptive",
            "role": "shot and motion change sampling",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Keeps coverage without flooding reconstruction with near-duplicate frames.",
        },
        {
            "id": "sharpness-motion-selector",
            "label": "Sharpness/motion selector",
            "model": "opencv-laplacian-optical-flow",
            "role": "blur rejection and parallax coverage",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Ranks selected frames by focus, motion, and pose diversity.",
        },
    ],
    "camera_pose": [
        {
            "id": "colmap-sift",
            "label": "COLMAP SIFT",
            "model": "COLMAP",
            "role": "camera matching and sparse reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Classic robust camera recovery for full frames.",
        },
        {
            "id": "hloc-lightglue",
            "label": "hloc + LightGlue",
            "model": "SuperPoint/DISK + LightGlue",
            "role": "learned feature matching for camera poses",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Modern learned matching lane for selected-frame camera/keypoint retention.",
        },
        {
            "id": "vggt-camera",
            "label": "VGGT camera head",
            "model": "VGGT",
            "role": "feed-forward camera/depth/point prediction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast learned geometry prior for short clips and sparse image sets.",
        },
    ],
    "video_reconstruction": [
        {
            "id": "colmap-openmvs",
            "label": "COLMAP + OpenMVS",
            "model": "COLMAP/OpenMVS",
            "role": "photogrammetry mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "STL-first baseline because it naturally produces a mesh.",
        },
        {
            "id": "gaussian-splatting-mesh",
            "label": "Gaussian Splatting + mesh",
            "model": "3D Gaussian Splatting + TSDF/Poisson extraction",
            "role": "reconstruct splats, extract mesh, repair STL",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Useful only if extracted STL quality beats photogrammetry/depth baselines.",
        },
        {
            "id": "vggt-fusion",
            "label": "VGGT fusion",
            "model": "VGGT",
            "role": "multi-frame depth/point fusion",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Candidate learned reconstruction path for short videos.",
        },
        {
            "id": "dust3r-mast3r",
            "label": "DUSt3R/MASt3R",
            "model": "DUSt3R or MASt3R",
            "role": "dense correspondence and 3D point prediction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Modern pairwise/multiview matching candidate before mesh extraction.",
        },
        {
            "id": "nerfstudio-mesh",
            "label": "Nerfstudio mesh",
            "model": "Nerfstudio",
            "role": "NeRF training, mesh extraction, STL repair",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Kept as an optional backend; visual quality alone is not enough for promotion.",
        },
    ],
    "image_to_mesh": [
        {
            "id": "triposg",
            "label": "TripoSG",
            "model": "TripoSG-style direct mesh",
            "role": "single image to mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Preferred direct mesh lane when available.",
        },
        {
            "id": "hunyuan3d-shape",
            "label": "Hunyuan3D Shape",
            "model": "Hunyuan3D Shape",
            "role": "single image to shape mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Direct image-to-3D candidate for full STL output.",
        },
        {
            "id": "triposr",
            "label": "TripoSR",
            "model": "TripoSR",
            "role": "single image sparse-view reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast local baseline for direct mesh smoke tests.",
        },
        {
            "id": "stable-fast-3d",
            "label": "Stable Fast 3D",
            "model": "SF3D",
            "role": "single image to textured mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast mesh candidate to compare against TripoSR/TripoSG-style outputs.",
        },
        {
            "id": "spar3d",
            "label": "SPAR3D",
            "model": "SPAR3D",
            "role": "single image sparse 3D reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Candidate already represented in benchmark configs.",
        },
    ],
    "stl_postprocess": [
        {
            "id": "trimesh-repair",
            "label": "Trimesh repair",
            "model": "trimesh",
            "role": "mesh cleanup and STL export",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Default lightweight repair and diagnostics path.",
        },
        {
            "id": "manifold3d",
            "label": "Manifold3D repair",
            "model": "manifold3d",
            "role": "watertight boolean/manifold conversion",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Promotion target for bad direct-mesh outputs.",
        },
        {
            "id": "pymeshlab-remesh",
            "label": "PyMeshLab remesh",
            "model": "pymeshlab",
            "role": "surface repair, decimation, simplification",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Useful for printable complexity controls before STL export.",
        },
    ],
}

DEFAULTS = {
    "selection": "sam2.1-hiera-large",
    "frame_selection": "uniform-frame-sampler",
    "camera_pose": "hloc-lightglue",
    "video_reconstruction": "colmap-openmvs",
    "image_to_mesh": "triposg",
    "stl_postprocess": "trimesh-repair",
}

STL_METRICS = [
    "watertightness",
    "manifoldness",
    "positive volume",
    "single component",
    "minimum printable thickness",
    "bbox aspect ratio",
    "surface Chamfer when ground truth exists",
]


def _model_index() -> dict[str, dict]:
    return {
        row["id"]: {**row, "group": group}
        for group, rows in MODEL_GROUPS.items()
        for row in rows
    }


def _selected_model(model_id: str | None, group: str) -> dict:
    candidate_id = model_id or DEFAULTS[group]
    row = _model_index().get(candidate_id)
    if not row or row["group"] != group:
        valid = ", ".join(model["id"] for model in MODEL_GROUPS[group])
        raise HTTPException(status_code=400, detail=f"Unsupported {group} model '{candidate_id}'. Valid: {valid}")
    return row


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "video-selection-planner",
        "mode": "planner-only",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/models")
async def models():
    return {
        "service": "video-selection-planner",
        "mode": "planner-only",
        "defaults": DEFAULTS,
        "groups": MODEL_GROUPS,
        "metrics": STL_METRICS,
        "notes": "This companion service exposes the video, selection, camera, direct-mesh, and STL-repair model surface. Heavy runners can attach behind the same ids.",
    }


@app.post("/plan")
async def plan(payload: dict):
    route = payload.get("route") or {}
    print_volume = payload.get("print_volume") or {}
    media_type = (payload.get("input") or {}).get("media_type", "photo")
    model_ids = payload.get("models") or {}

    selection = _selected_model(model_ids.get("selection"), "selection")
    stl_postprocess = _selected_model(model_ids.get("stl_postprocess"), "stl_postprocess")

    stages = []
    if media_type == "video":
        frame_selection = _selected_model(model_ids.get("frame_selection"), "frame_selection")
        camera_pose = _selected_model(model_ids.get("camera_pose"), "camera_pose")
        video_reconstruction = _selected_model(model_ids.get("video_reconstruction"), "video_reconstruction")
        stages = [
            {"id": "frame-selection", "model": frame_selection},
            {"id": "object-selection", "model": selection},
            {"id": "camera-pose", "model": camera_pose},
            {"id": "video-reconstruction", "model": video_reconstruction},
            {"id": "stl-postprocess", "model": stl_postprocess},
        ]
    else:
        image_to_mesh = _selected_model(model_ids.get("image_to_mesh"), "image_to_mesh")
        stages = [
            {"id": "object-selection", "model": selection},
            {"id": "image-to-mesh", "model": image_to_mesh},
            {"id": "stl-postprocess", "model": stl_postprocess},
        ]

    return {
        "status": "planned",
        "run_id": uuid4().hex,
        "service": "video-selection-planner",
        "execution_mode": "planner-only",
        "route": route,
        "print_volume": print_volume,
        "stages": stages,
        "metrics": STL_METRICS,
        "next_backend_contract": {
            "input": "uploaded media path plus selected frame/mask artifacts",
            "output": "watertight STL path plus diagnostics JSON",
            "promotion_gate": "all hard STL checks must pass before model promotion",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VIDEO_SELECTION_PORT", "8005")))
