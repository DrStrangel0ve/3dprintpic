from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


SPACE_RUNTIME_ROOT = Path(
    os.getenv("THREEDPRINTPIC_RUNTIME_DIR", Path(tempfile.gettempdir()) / "3dprintpic-space")
)
HF_XET_CACHE_DIR = Path(os.getenv("HF_XET_CACHE", SPACE_RUNTIME_ROOT / "huggingface-xet"))
HF_XET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_XET_CACHE"] = str(HF_XET_CACHE_DIR)

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
        prepare_object_selection,
        select_precomputed_object,
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
        prepare_object_selection,
        select_precomputed_object,
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
.selection-hidden { display: none !important; }
.sam3-hover-overlay {
  position: absolute;
  pointer-events: none;
  z-index: 8;
}
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


SELECTION_HOVER_JS = r"""
() => {
  if (window.__sam3HoverPickerInstalled) return [];
  window.__sam3HoverPickerInstalled = true;
  document.documentElement.dataset.sam3HoverPickerInstalled = "true";
  const prefixes = ["relief", "diorama", "mesh"];
  const pickers = new Map();

  const componentInput = (id) => document.querySelector(
    `#${id} textarea, #${id} input`
  );

  const sourceImage = (root) => {
    const candidates = Array.from(root.querySelectorAll('[data-testid="image"] img, img[src]'));
    return candidates.find((image) => image.naturalWidth > 1 && image.naturalHeight > 1) || null;
  };

  const pickerFor = (prefix) => {
    if (!pickers.has(prefix)) {
      pickers.set(prefix, {
        prefix,
        manifestText: "",
        manifest: null,
        mapPixels: null,
        root: null,
        image: null,
        canvas: null,
        context: null,
        hoveredId: 0,
        selectedId: 0,
        renderedId: -1,
      });
    }
    return pickers.get(prefix);
  };

  const contentRect = (picker) => {
    const image = picker.image;
    if (!image || !image.naturalWidth || !image.naturalHeight) return null;
    const bounds = image.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return null;
    const fit = window.getComputedStyle(image).objectFit;
    const widthScale = bounds.width / image.naturalWidth;
    const heightScale = bounds.height / image.naturalHeight;
    let scale = Math.min(widthScale, heightScale);
    if (fit === "scale-down") scale = Math.min(1, scale);
    if (fit === "none") scale = 1;
    if (fit === "cover") scale = Math.max(widthScale, heightScale);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    return {
      left: bounds.left + (bounds.width - width) / 2,
      top: bounds.top + (bounds.height - height) / 2,
      width,
      height,
    };
  };

  const syncCanvasLayout = (picker) => {
    if (!picker.root || !picker.canvas || !picker.manifest) return;
    const imageRect = contentRect(picker);
    if (!imageRect) return;
    const rootRect = picker.root.getBoundingClientRect();
    picker.root.style.position = "relative";
    picker.canvas.style.left = `${imageRect.left - rootRect.left}px`;
    picker.canvas.style.top = `${imageRect.top - rootRect.top}px`;
    picker.canvas.style.width = `${imageRect.width}px`;
    picker.canvas.style.height = `${imageRect.height}px`;
    if (
      picker.canvas.width !== picker.manifest.width
      || picker.canvas.height !== picker.manifest.height
    ) {
      picker.canvas.width = picker.manifest.width;
      picker.canvas.height = picker.manifest.height;
      picker.renderedId = -1;
      renderRegion(picker, picker.hoveredId || picker.selectedId);
    }
  };

  const regionAt = (picker, clientX, clientY) => {
    if (!picker.mapPixels || !picker.manifest) return 0;
    const rect = contentRect(picker);
    if (!rect) return 0;
    const nx = (clientX - rect.left) / rect.width;
    const ny = (clientY - rect.top) / rect.height;
    if (nx < 0 || nx >= 1 || ny < 0 || ny >= 1) return 0;
    const x = Math.min(picker.manifest.width - 1, Math.floor(nx * picker.manifest.width));
    const y = Math.min(picker.manifest.height - 1, Math.floor(ny * picker.manifest.height));
    const offset = 4 * (y * picker.manifest.width + x);
    return picker.mapPixels[offset]
      + (picker.mapPixels[offset + 1] << 8)
      + (picker.mapPixels[offset + 2] << 16);
  };

  const renderRegion = (picker, regionId) => {
    if (!picker.context || !picker.mapPixels || !picker.manifest) return;
    if (picker.renderedId === regionId) return;
    picker.renderedId = regionId;
    const output = picker.context.createImageData(
      picker.manifest.width,
      picker.manifest.height
    );
    if (regionId) {
      for (let offset = 0; offset < picker.mapPixels.length; offset += 4) {
        const candidate = picker.mapPixels[offset]
          + (picker.mapPixels[offset + 1] << 8)
          + (picker.mapPixels[offset + 2] << 16);
        if (candidate === regionId) {
          output.data[offset] = 8;
          output.data[offset + 1] = 119;
          output.data[offset + 2] = 91;
          output.data[offset + 3] = 154;
        }
      }
    }
    picker.context.putImageData(output, 0, 0);
    const region = picker.manifest.regions[String(regionId)];
    picker.canvas.setAttribute("aria-label", region ? `${region.label} selection preview` : "");
  };

  const pointerPosition = (picker, event) => {
    const rect = contentRect(picker);
    if (!rect) return null;
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x >= 1 || y < 0 || y >= 1) return null;
    return {x, y};
  };

  const commitSelection = (picker, event) => {
    if (event.target.closest("button, input, .icon-button-wrapper")) return;
    const regionId = regionAt(picker, event.clientX, event.clientY);
    const position = pointerPosition(picker, event);
    if (!regionId || !position) return;
    event.preventDefault();
    event.stopPropagation();
    picker.selectedId = regionId;
    renderRegion(picker, regionId);
    const target = componentInput(`${picker.prefix}-hover-event`);
    if (!target) return;
    const payload = JSON.stringify({
      x: position.x,
      y: position.y,
      region_id: regionId,
      nonce: Date.now(),
    });
    const prototype = target.tagName === "TEXTAREA"
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value").set;
    setter.call(target, payload);
    target.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: payload,
    }));
  };

  const attachPicker = (prefix) => {
    const picker = pickerFor(prefix);
    const root = document.getElementById(`${prefix}-source`);
    if (!root) return;
    const image = sourceImage(root);
    if (!image) return;
    picker.root = root;
    picker.image = image;
    let canvas = root.querySelector("canvas.sam3-hover-overlay");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.className = "sam3-hover-overlay";
      root.appendChild(canvas);
    }
    picker.canvas = canvas;
    picker.context = canvas.getContext("2d", {willReadFrequently: false});
    if (!root.dataset.sam3HoverBound) {
      root.dataset.sam3HoverBound = "true";
      root.addEventListener("pointermove", (event) => {
        const current = pickerFor(prefix);
        const regionId = regionAt(current, event.clientX, event.clientY);
        current.hoveredId = regionId;
        renderRegion(current, regionId || current.selectedId);
        if (current.image) current.image.style.cursor = regionId ? "pointer" : "default";
      }, {passive: true});
      root.addEventListener("pointerleave", () => {
        const current = pickerFor(prefix);
        current.hoveredId = 0;
        renderRegion(current, current.selectedId);
        if (current.image) current.image.style.cursor = "default";
      }, {passive: true});
      root.addEventListener("click", (event) => commitSelection(pickerFor(prefix), event), true);
    }
    syncCanvasLayout(picker);
  };

  const loadManifest = (prefix) => {
    const picker = pickerFor(prefix);
    const input = componentInput(`${prefix}-hover-manifest`);
    const manifestText = input ? input.value : "";
    if (manifestText === picker.manifestText) {
      attachPicker(prefix);
      return;
    }
    picker.manifestText = manifestText;
    picker.manifest = null;
    picker.mapPixels = null;
    picker.hoveredId = 0;
    picker.selectedId = 0;
    picker.renderedId = -1;
    if (!manifestText) {
      if (picker.context && picker.canvas) {
        picker.context.clearRect(0, 0, picker.canvas.width, picker.canvas.height);
      }
      return;
    }
    try {
      picker.manifest = JSON.parse(manifestText);
    } catch (_error) {
      return;
    }
    const manifest = picker.manifest;
    const hitMap = new Image();
    hitMap.onload = () => {
      if (picker.manifest !== manifest || picker.manifestText !== manifestText) return;
      const scratch = document.createElement("canvas");
      scratch.width = manifest.width;
      scratch.height = manifest.height;
      const context = scratch.getContext("2d", {willReadFrequently: true});
      context.drawImage(hitMap, 0, 0, scratch.width, scratch.height);
      picker.mapPixels = context.getImageData(0, 0, scratch.width, scratch.height).data;
      attachPicker(prefix);
      renderRegion(picker, 0);
    };
    hitMap.src = picker.manifest.hit_map;
  };

  const sync = () => prefixes.forEach((prefix) => loadManifest(prefix));
  const observer = new MutationObserver(sync);
  observer.observe(document.body, {childList: true, subtree: true});
  window.addEventListener("resize", () => prefixes.forEach((prefix) => {
    syncCanvasLayout(pickerFor(prefix));
  }));
  window.setInterval(sync, 350);
  sync();
  return [];
}
"""

def _dimensions(image_path, long_edge_mm):
    return image_dimensions_mm(image_path, long_edge_mm)


def _reset_selection(image_path, scope):
    if str(scope).startswith("Select"):
        status = "Finding selectable objects with SAM 3..." if image_path else "Upload a photo to find objects."
    else:
        status = "Using the full scene."
    return "", None, None, None, status


@spaces.GPU(duration=45)
def _prepare_hover_selection(image_path, scope):
    if not str(scope).startswith("Select"):
        return "", None, None, None, "Using the full scene."
    if not image_path:
        return "", None, None, None, "Upload a photo to find objects."
    try:
        manifest, precompute_state = prepare_object_selection(image_path)
        region_count = int(precompute_state.get("region_count", 0))
        status = f"{region_count} selectable regions ready. Hover to preview and click to select."
        return manifest, precompute_state, None, None, status
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _select_from_hover_event(image_path, precompute_state, event_json):
    try:
        event = json.loads(event_json or "{}")
        result = select_precomputed_object(
            image_path,
            precompute_state,
            float(event["x"]),
            float(event["y"]),
            int(event["region_id"]),
        )
        labels = ", ".join(result.get("labels") or ["object"])
        coverage = 100.0 * float(result.get("mask_coverage", 0.0))
        return result["overlay"], result, f"Selected {labels} with SAM 3 ({coverage:.1f}% of image)."
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


@spaces.GPU(duration=110)
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


@spaces.GPU(duration=110)
def _generate_diorama_ui(image, scope, selection, max_size, scene_depth, base_thickness):
    try:
        cleanup_expired_outputs()
        return generate_diorama(image, scope, selection, max_size, scene_depth, base_thickness)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


FULL_MESH_GPU_DURATION_SECONDS = 150


@spaces.GPU(duration=FULL_MESH_GPU_DURATION_SECONDS)
def _generate_full_mesh_ui(image, scope, selection, max_dimension, seed):
    try:
        cleanup_expired_outputs()
        return generate_full_mesh(image, scope, selection, max_dimension, int(seed))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _selection_controls(prefix: str, *, selection_required: bool = False):
    selection_state = gr.State(None)
    precompute_state = gr.State(None)
    scope = gr.Radio(
        choices=["Select object"] if selection_required else ["Full scene", "Select object"],
        value="Select object" if selection_required else "Full scene",
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
    initial_status = "Upload a photo to find objects." if selection_required else "Using the full scene."
    selection_status = gr.Markdown(initial_status, elem_classes="selection-status")
    hover_manifest = gr.Textbox(
        value="",
        interactive=False,
        container=False,
        elem_id=f"{prefix}-hover-manifest",
        elem_classes="selection-hidden",
    )
    hover_event = gr.Textbox(
        value="",
        interactive=True,
        container=False,
        elem_id=f"{prefix}-hover-event",
        elem_classes="selection-hidden",
    )
    prepare_trigger = gr.Button(
        "Prepare object selection",
        elem_id=f"{prefix}-prepare-selection",
        elem_classes="selection-hidden",
    )
    prepare_event = prepare_trigger.click(
        _prepare_hover_selection,
        inputs=[source, scope],
        outputs=[
            hover_manifest,
            precompute_state,
            selection_preview,
            selection_state,
            selection_status,
        ],
        show_progress="full",
        show_progress_on=source,
        concurrency_id="gpu-work",
        concurrency_limit=1,
        trigger_mode="always_last",
    )
    hover_event.input(
        _select_from_hover_event,
        inputs=[source, precompute_state, hover_event],
        outputs=[selection_preview, selection_state, selection_status],
        show_progress="hidden",
        queue=False,
        trigger_mode="always_last",
    )
    reset_outputs = [
        hover_manifest,
        precompute_state,
        selection_preview,
        selection_state,
        selection_status,
    ]
    trigger_js = (
        "(image, scope) => { const install = "
        + SELECTION_HOVER_JS
        + "; install(); "
        + "if (image && String(scope).startsWith('Select')) { "
        + "window.setTimeout(() => { "
        + f"const root = document.getElementById('{prefix}-prepare-selection'); "
        + "const button = root && (root.matches('button') ? root : root.querySelector('button')); "
        + "if (button) button.click(); }, 100); } return [image, scope]; }"
    )
    source.change(
        _reset_selection,
        inputs=[source, scope],
        outputs=reset_outputs,
        show_progress="hidden",
        queue=False,
        js=trigger_js,
        cancels=prepare_event,
    )
    scope.change(
        _reset_selection,
        inputs=[source, scope],
        outputs=reset_outputs,
        show_progress="hidden",
        queue=False,
        js=trigger_js,
        cancels=prepare_event,
    )
    return source, scope, selection_state


theme = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="zinc",
    radius_size="sm",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

with gr.Blocks(
    title="3D Print a Picture",
    theme=theme,
    css=CSS,
) as demo:
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
                concurrency_id="gpu-work",
                concurrency_limit=1,
            )

        with gr.Tab("Scene Diorama"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tool-panel"):
                    diorama_image, diorama_scope, diorama_selection = _selection_controls("diorama")
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
                inputs=[diorama_image, diorama_scope, diorama_selection, diorama_size, diorama_depth, diorama_base],
                outputs=[
                    diorama_model,
                    diorama_stl,
                    diorama_glb,
                    diorama_preview,
                    diorama_diagnostics,
                ],
                concurrency_id="gpu-work",
                concurrency_limit=1,
            )

        with gr.Tab("Full Mesh STL"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tool-panel"):
                    mesh_image, mesh_scope, mesh_selection = _selection_controls(
                        "mesh", selection_required=True
                    )
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
                concurrency_id="gpu-work",
                concurrency_limit=1,
            )

    gr.HTML(
        f"""
        <div class="footer-note">
          Free, non-commercial research demo. Uploads and generated files remain in temporary Space storage
          and are not used for training. Project `{PROJECT_REVISION[:8]}` | TripoSG weights `{TRIPOSG_MODEL_REVISION[:8]}`.
        </div>
        """
    )

    demo.load(None, js=SELECTION_HOVER_JS, queue=False)

demo.queue(default_concurrency_limit=1, max_size=16)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
