from __future__ import annotations


STL_MODE_DEPTH_RELIEF = "depth-relief"
STL_MODE_SINGLE_IMAGE_MESH = "single-image-mesh"
STL_MODE_MULTIVIEW_MESH = "multiview-mesh"
STL_MODE_SOURCE_MESH_ORACLE = "source-mesh-oracle"

STL_MODES = (
    STL_MODE_DEPTH_RELIEF,
    STL_MODE_SINGLE_IMAGE_MESH,
    STL_MODE_MULTIVIEW_MESH,
    STL_MODE_SOURCE_MESH_ORACLE,
)

DIRECT_METHOD_STL_MODES = {
    "source-mesh-oracle": STL_MODE_SOURCE_MESH_ORACLE,
    "external-image-to-mesh": STL_MODE_SINGLE_IMAGE_MESH,
    "external-multiview-to-mesh": STL_MODE_MULTIVIEW_MESH,
}


def validate_stl_mode(value: str) -> str:
    mode = str(value or "").strip()
    if mode and mode not in STL_MODES:
        expected = ", ".join(STL_MODES)
        raise ValueError(f"Unknown stl_mode `{mode}`. Expected one of: {expected}")
    return mode


def infer_stl_mode(method: str, *, emit_stl: bool = False) -> str:
    direct_mode = DIRECT_METHOD_STL_MODES.get(str(method or ""))
    if direct_mode:
        return direct_mode
    return STL_MODE_DEPTH_RELIEF if emit_stl else ""


def experiment_stl_mode(experiment: dict, *, default_emit_stl: bool = False) -> str:
    explicit = validate_stl_mode(experiment.get("stl_mode", ""))
    if explicit:
        return explicit
    return infer_stl_mode(
        experiment.get("method", ""),
        emit_stl=bool(experiment.get("emit_stl", default_emit_stl)),
    )
