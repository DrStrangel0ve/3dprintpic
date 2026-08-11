from __future__ import annotations

import os

import gradio as gr

try:
    import spaces
except ImportError:  # Local UI/testing shim; Hugging Face injects the real module.
    class _Spaces:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def decorator(function):
                return function

            return decorator

    spaces = _Spaces()

try:
    from .space_runtime import (
        DEPTH_MODEL,
        PROJECT_REVISION,
        SAM3_MODEL,
        TRIPOSG_MODEL_REVISION,
        cleanup_expired_outputs,
        generate_diorama,
        generate_full_mesh,
        generate_relief,
        image_dimensions_mm,
        select_object,
    )
except ImportError:  # Hugging Face runs app.py from the Space repository root.
    from space_runtime import (
        DEPTH_MODEL,
        PROJECT_REVISION,
        SAM3_MODEL,
        TRIPOSG_MODEL_REVISION,
        cleanup_expired_outputs,
        generate_diorama,
        generate_full_mesh,
        generate_relief,
        image_dimensions_mm,
        select_object,
    )


CSS = """
:root {
  --studio-green: #08775b;
  --studio-green-hover: #066249;
}
.gradio-container,
main.app {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: 1440px !important;
  margin: 0 auto !important;
  color: var(--body-text-color) !important;
  background: var(--body-background-fill) !important;
}
.app-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 20px;
  padding: 10px 2px 16px;
  border-bottom: 1px solid var(--border-color-primary);
}
.app-title {
  color: var(--body-text-color);
  font-size: 26px;
  font-weight: 680;
  line-height: 1.1;
  letter-spacing: 0;
}
.app-subtitle { color: var(--body-text-color-subdued); font-size: 13px; margin-top: 5px; }
.runtime-note { color: var(--body-text-color-subdued); font-size: 12px; text-align: right; }
.tool-panel {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 6px !important;
  background: var(--background-fill-primary) !important;
}
.primary-button {
  background: var(--studio-green) !important;
  border-color: var(--studio-green) !important;
  color: #ffffff !important;
}
.primary-button:hover {
  background: var(--studio-green-hover) !important;
  border-color: var(--studio-green-hover) !important;
}
.selection-status { min-height: 34px; color: var(--body-text-color-subdued); font-size: 13px; }
.model-note {
  color: var(--body-text-color-subdued);
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.model-note code { white-space: normal; }
.footer-note {
  border-top: 1px solid var(--border-color-primary);
  margin-top: 18px;
  padding-top: 12px;
  color: var(--body-text-color-subdued);
  font-size: 12px;
}
button, input, textarea, select { letter-spacing: 0 !important; }
@media (max-width: 700px) {
  main.app { padding: 12px !important; }
  .app-header {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }
  .runtime-note { text-align: left; }
}
"""


def _dimensions(image_path, long_edge_mm):
    return image_dimensions_mm(image_path, long_edge_mm)


def _clear_selection(scope):
    status = "Click an object in the uploaded photo." if str(scope).startswith("Select") else "Using the full scene."
    return None, None, status


@spaces.GPU(duration=75)
def _select_from_click(image_path, scope, previous_selection, event: gr.SelectData):
    if not str(scope).startswith("Select"):
        return image_path, None, "Using the full scene."
    if not image_path:
        raise gr.Error("Upload a photo before selecting an object")
    try:
        x, y = event.index
        result = select_object(image_path, float(x), float(y))
        labels = ", ".join(result.get("labels") or ["object"])
        coverage = 100.0 * float(result.get("mask_coverage", 0.0))
        return result["overlay"], result, f"Selected {labels} with SAM 3 ({coverage:.1f}% of image)."
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


@spaces.GPU(duration=180)
def _generate_relief_ui(image, scope, selection, long_edge, relief_height, samples, background_ratio):
    try:
        cleanup_expired_outputs()
        return generate_relief(
            image,
            scope,
            selection,
            long_edge,
            relief_height,
            int(samples),
            background_ratio,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


@spaces.GPU(duration=180)
def _generate_diorama_ui(image, selection, max_size, scene_depth, base_thickness):
    try:
        cleanup_expired_outputs()
        return generate_diorama(image, selection, max_size, scene_depth, base_thickness)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


@spaces.GPU(duration=300)
def _generate_full_mesh_ui(image, scope, selection, max_dimension, seed):
    try:
        cleanup_expired_outputs()
        return generate_full_mesh(image, scope, selection, max_dimension, int(seed))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _selection_controls(prefix: str):
    selection_state = gr.State(None)
    scope = gr.Radio(
        choices=["Full scene", "Select object"],
        value="Full scene",
        label="Scope",
    )
    source = gr.Image(
        type="filepath",
        image_mode="RGB",
        sources=["upload", "webcam", "clipboard"],
        label="Photo",
        height=430,
        elem_id=f"{prefix}-source",
    )
    selection_preview = gr.Image(
        type="filepath",
        label="Selection",
        interactive=False,
        height=250,
        visible=True,
    )
    selection_status = gr.Markdown("Using the full scene.", elem_classes="selection-status")
    source.select(
        _select_from_click,
        inputs=[source, scope, selection_state],
        outputs=[selection_preview, selection_state, selection_status],
        show_progress="full",
    )
    source.change(
        lambda: (None, None, "Upload complete. Choose a scope."),
        outputs=[selection_preview, selection_state, selection_status],
        show_progress="hidden",
    )
    scope.change(
        _clear_selection,
        inputs=[scope],
        outputs=[selection_preview, selection_state, selection_status],
        show_progress="hidden",
    )
    return source, scope, selection_state


theme = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="zinc",
    radius_size="sm",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

with gr.Blocks(title="3D Print a Picture", theme=theme, css=CSS) as demo:
    gr.HTML(
        f"""
        <div class="app-header">
          <div>
            <div class="app-title">3D Print a Picture</div>
            <div class="app-subtitle">Photo-to-STL reliefs, layered scenes, and complete printable meshes</div>
          </div>
          <div class="runtime-note">ZeroGPU public demo<br>Original pixels only | no inpainting</div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Printable Relief"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tool-panel"):
                    relief_image, relief_scope, relief_selection = _selection_controls("relief")
                with gr.Column(scale=4, elem_classes="tool-panel"):
                    long_edge = gr.Slider(48, 240, value=128, step=1, label="Long edge (mm)")
                    with gr.Row():
                        x_size = gr.Number(value=128, label="X width (mm)", interactive=False)
                        y_size = gr.Number(value=128, label="Y height (mm)", interactive=False)
                    relief_height = gr.Slider(2, 40, value=20, step=1, label="Relief height Z (mm)")
                    detail_samples = gr.Slider(240, 720, value=420, step=20, label="Surface detail")
                    background_ratio = gr.Slider(
                        0.35,
                        0.85,
                        value=0.65,
                        step=0.05,
                        label="Background depth",
                    )
                    relief_generate = gr.Button("Generate relief", variant="primary", elem_classes="primary-button")
                    gr.Markdown(f"Depth: `{DEPTH_MODEL}` | Selection: `{SAM3_MODEL}`", elem_classes="model-note")
            with gr.Row(equal_height=False):
                relief_model = gr.Model3D(label="Printable STL", display_mode="solid", height=500)
                with gr.Column():
                    relief_preview = gr.Image(label="Relief surface", height=320, interactive=False)
                    relief_file = gr.File(label="Download STL")
                    relief_diagnostics = gr.JSON(label="Print checks")

            relief_image.change(
                _dimensions,
                inputs=[relief_image, long_edge],
                outputs=[x_size, y_size],
                show_progress="hidden",
            )
            long_edge.change(
                _dimensions,
                inputs=[relief_image, long_edge],
                outputs=[x_size, y_size],
                show_progress="hidden",
            )
            relief_generate.click(
                _generate_relief_ui,
                inputs=[
                    relief_image,
                    relief_scope,
                    relief_selection,
                    long_edge,
                    relief_height,
                    detail_samples,
                    background_ratio,
                ],
                outputs=[relief_model, relief_file, relief_preview, relief_diagnostics],
            )

        with gr.Tab("Scene Diorama"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tool-panel"):
                    diorama_image, _diorama_scope, diorama_selection = _selection_controls("diorama")
                with gr.Column(scale=4, elem_classes="tool-panel"):
                    diorama_size = gr.Slider(80, 240, value=180, step=5, label="Maximum size (mm)")
                    diorama_depth = gr.Slider(20, 100, value=64, step=2, label="Scene depth (mm)")
                    diorama_base = gr.Slider(1.2, 6, value=2.4, step=0.2, label="Base thickness (mm)")
                    diorama_generate = gr.Button("Build diorama", variant="primary", elem_classes="primary-button")
                    gr.Markdown(f"Depth: `{DEPTH_MODEL}` | Selected layers: `{SAM3_MODEL}`", elem_classes="model-note")
            with gr.Row(equal_height=False):
                diorama_model = gr.Model3D(label="Colored scene", display_mode="solid", height=500)
                with gr.Column():
                    diorama_preview = gr.Image(label="Scene preview", height=300, interactive=False)
                    with gr.Row():
                        diorama_stl = gr.File(label="Download STL")
                        diorama_glb = gr.File(label="Download GLB")
                    diorama_diagnostics = gr.JSON(label="Print checks")
            diorama_generate.click(
                _generate_diorama_ui,
                inputs=[diorama_image, diorama_selection, diorama_size, diorama_depth, diorama_base],
                outputs=[
                    diorama_model,
                    diorama_stl,
                    diorama_glb,
                    diorama_preview,
                    diorama_diagnostics,
                ],
            )

        with gr.Tab("Full Mesh STL"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tool-panel"):
                    mesh_image, mesh_scope, mesh_selection = _selection_controls("mesh")
                with gr.Column(scale=4, elem_classes="tool-panel"):
                    mesh_size = gr.Slider(40, 220, value=96, step=2, label="Maximum dimension (mm)")
                    mesh_seed = gr.Number(value=42, precision=0, label="Seed")
                    mesh_generate = gr.Button("Generate full mesh", variant="primary", elem_classes="primary-button")
                    gr.Markdown(
                        "Mesh: `VAST-AI/TripoSG` | Repair: `trimesh` | Convex-hull fallback disabled",
                        elem_classes="model-note",
                    )
            with gr.Row(equal_height=False):
                mesh_model = gr.Model3D(label="Printable full mesh", display_mode="solid", height=520)
                with gr.Column():
                    mesh_stl = gr.File(label="Download STL")
                    mesh_glb = gr.File(label="Download GLB")
                    mesh_diagnostics = gr.JSON(label="Print checks")
            mesh_generate.click(
                _generate_full_mesh_ui,
                inputs=[mesh_image, mesh_scope, mesh_selection, mesh_size, mesh_seed],
                outputs=[mesh_model, mesh_stl, mesh_glb, mesh_diagnostics],
            )

    gr.HTML(
        f"""
        <div class="footer-note">
          Free, non-commercial research demo. Uploads and generated files remain in temporary Space storage
          and are not used for training. Project `{PROJECT_REVISION[:8]}` | TripoSG weights `{TRIPOSG_MODEL_REVISION[:8]}`.
        </div>
        """
    )

demo.queue(default_concurrency_limit=1, max_size=16)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
