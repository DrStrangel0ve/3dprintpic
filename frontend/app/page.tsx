/* eslint-disable @next/next/no-img-element -- Blob URLs and backend mask previews must bypass the Next image optimizer. */
'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  BadgeCheck,
  AlertTriangle,
  Box,
  Boxes,
  Braces,
  Camera,
  Check,
  CircleDot,
  Cuboid,
  Download,
  FileDown,
  Film,
  Image as ImageIcon,
  Layers3,
  Loader2,
  MousePointer2,
  Play,
  Printer,
  RefreshCcw,
  ScanLine,
  Sparkles,
  Upload,
  Wand2,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { WorkspaceColumn, WorkspaceShell } from '@/components/workspace/workspace-shell';

const StlViewer = dynamic(
  () => import('react-stl-viewer').then((module) => module.StlViewer),
  { ssr: false },
);

const DEFAULT_BACKEND_URL = 'http://localhost:8004';
const DEFAULT_VIDEO_BACKEND_URL = 'http://localhost:8005';
const CONFIGURED_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || DEFAULT_BACKEND_URL;
const CONFIGURED_VIDEO_BACKEND_URL = process.env.NEXT_PUBLIC_VIDEO_BACKEND_URL || DEFAULT_VIDEO_BACKEND_URL;
const PROCESS_IMAGE_TIMEOUT_MS = 10 * 60 * 1000;
const IMAGE_TO_MESH_TIMEOUT_MS = 60 * 60 * 1000;
const VIDEO_TO_MESH_TIMEOUT_MS = 60 * 60 * 1000;
const FULL_MESH_RUNNER_STL_POSTPROCESS = 'trimesh-repair';
const PRODUCTION_DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Large-hf';
const PRODUCTION_PHOTO_SELECTION_MODEL = 'sam3-person-aware';
const PRODUCTION_TURNTABLE_SELECTION_MODEL = 'turntable-grabcut';
const PRODUCTION_TRACKED_VIDEO_SELECTION_MODEL = 'sam2.1-hiera-tiny-video';
const PRODUCTION_TURNTABLE_FRAME_MODEL = 'uniform-frame-sampler';
const PRODUCTION_TRACKED_FRAME_MODEL = 'sharpness-motion-selector';
const PRODUCTION_IMAGE_TO_MESH_MODEL = 'triposg';
const LIVE_VIDEO_CAMERA_MODEL = 'turntable-orbit';
const LIVE_VIDEO_RECONSTRUCTION_MODEL = 'multiview-visual-hull';
const LIVE_VIDEO_FRAME_SELECTION_MODELS = ['uniform-frame-sampler', 'sharpness-motion-selector'] as const;
const LIVE_VIDEO_SEGMENTATION_MODELS = [
  'turntable-grabcut',
  'sam2.1-hiera-tiny-video',
  'sam2.1-hiera-base-plus-video',
] as const;

function isLiveVideoSegmentationModel(modelId: string) {
  return (LIVE_VIDEO_SEGMENTATION_MODELS as readonly string[]).includes(modelId);
}

function isLiveVideoFrameSelectionModel(modelId: string) {
  return (LIVE_VIDEO_FRAME_SELECTION_MODELS as readonly string[]).includes(modelId);
}
const DEPTH_PRO_MODEL_ID = 'apple/DepthPro-hf';
const MODEL_GROUPS: ModelGroup[] = [
  'selection',
  'frame_selection',
  'camera_pose',
  'video_reconstruction',
  'image_to_mesh',
  'stl_postprocess',
];

type MediaKind = 'photo' | 'video';
type PhotoScope = 'whole-image' | 'object-selection';
type PhotoTarget = 'depth-relief' | 'scene-diorama' | 'full-mesh';
type VideoTarget = 'turntable-mesh' | 'tracked-relief';
type VideoScope = 'selected-frames' | 'everything';
type RunState = 'idle' | 'running' | 'ready' | 'blocked' | 'error';
type ReliefPolarity = 'raised-print' | 'mold';
type CatalogState = 'loading' | 'ready' | 'fallback';
type ModelGroup =
  | 'selection'
  | 'frame_selection'
  | 'camera_pose'
  | 'video_reconstruction'
  | 'image_to_mesh'
  | 'stl_postprocess';

type ModelOption = {
  id: string;
  label: string;
  model?: string;
  role?: string;
  availability?: string;
  notes?: string;
  local?: boolean;
  gpu_supported?: boolean;
};

type ModelCatalog = {
  service?: string;
  mode?: string;
  defaults: Record<ModelGroup, string>;
  groups: Record<ModelGroup, ModelOption[]>;
  metrics: string[];
  notes?: string;
};

type ProviderReadiness = {
  id: string;
  label?: string;
  status?: 'available' | 'missing' | string;
  runnable?: boolean;
  setup_errors?: string[];
  checks?: Record<string, unknown>;
  env?: {
    timeout_seconds?: number;
    provider_dir_configured?: boolean;
    provider_python_configured?: boolean;
    provider_dir_env_names?: string[];
    provider_python_env_names?: string[];
  };
};

type PlannerResponse = {
  status: string;
  run_id?: string;
  service?: string;
  execution_mode?: string;
  stages?: Array<{ id: string; model: ModelOption }>;
  metrics?: string[];
  next_backend_contract?: Record<string, string>;
  print_volume?: PrintVolumePlan;
};

type StlDiagnostics = {
  artifact_contract?: string;
  runner?: string;
  stl_exists?: boolean;
  stl_is_watertight?: boolean;
  stl_is_volume?: boolean;
  stl_is_manifold?: boolean;
  stl_winding_consistent?: boolean;
  stl_positive_volume?: boolean;
  stl_single_component?: boolean;
  stl_faces?: number;
  stl_bbox_aspect_ratio?: number | null;
  stl_nonmanifold_edge_count?: number;
  stl_degenerate_face_count?: number;
  stl_passes_hard_checks?: boolean;
  stl_failed_checks?: string[];
  relief_postprocess?: {
    emitted_printability?: {
      supported?: boolean;
      recognition_first_oversampling?: boolean;
      slope_limit_passed?: boolean;
      processing_minimum_feature_scaled_to_output_mm?: number | null;
      slope_p99_mm_per_mm?: number | null;
    };
  };
};

type RuntimeInfo = {
  torch?: string | null;
  cuda_available?: boolean;
  cuda_version?: string | null;
  device?: string;
  error?: string;
};

type BackendHealth = {
  status?: string;
  runtime?: RuntimeInfo;
  default_depth_provider?: string;
  default_depth_model?: string;
};

type StageTimings = Record<string, number>;

type DepthRunMetadata = {
  provider?: string;
  requested_model?: string;
  effective_model?: string;
  fallback_model?: string | null;
  fallback_reason?: string | null;
};

type DepthPreloadStatus = {
  model_id?: string;
  status?: 'idle' | 'missing' | 'downloading' | 'ready' | 'error' | string;
  message?: string;
  downloaded_bytes?: number;
  total_bytes?: number;
  progress_percent?: number;
  complete?: boolean;
  missing_files?: string[];
  incomplete_file_count?: number;
  error?: string | null;
};

type SelectionPoint = {
  id: string;
  x: number;
  y: number;
};

type SelectionResult = {
  job_id?: string;
  selected_image?: string;
  selected_image_url?: string;
  mask?: string;
  mask_url?: string;
  overlay?: string;
  overlay_url?: string;
  tint?: string;
  tint_url?: string;
  metadata_url?: string;
  points?: Array<{ x: number; y: number }>;
  model_id?: string;
  model_status?: string;
  model_error?: string | null;
  selection_labels?: string[];
  mask_pixels?: number;
  mask_coverage?: number;
};

type SelectionMask = SelectionResult & {
  id: string;
  x: number;
  y: number;
  tintUrl: string;
  maskPath: string;
};

type SelectionPrecomputeResult = {
  precompute_id?: string | null;
  model_id?: string;
  model_status?: string;
  precompute_supported?: boolean;
  message?: string;
  segment_count?: number;
  selection_labels?: string[];
  timings?: StageTimings;
};

type SelectionApplyResult = {
  file: File;
  previewUrl: string;
  selectionJobId: string;
};

type PrinterPresetId = 'bambulab-p1s' | 'custom';

type PrinterPreset = {
  id: PrinterPresetId;
  label: string;
  maxX: number;
  maxY: number;
  maxZ: number;
  nozzleDiameter: number;
};

type PrintVolumePlan = {
  preset: PrinterPresetId;
  label: string;
  max_x_mm: number;
  max_y_mm: number;
  max_z_mm: number;
  clearance_mm: number;
  usable_x_mm: number;
  usable_y_mm: number;
  usable_z_mm: number;
  max_target_dimension_mm: number;
  target_dimension_mm: number;
  print_scale_percent: number;
  max_relief_height_mm: number;
  nozzle_diameter_mm: number;
  minimum_feature_mm: number;
};

type MediaDimensions = {
  width: number;
  height: number;
};

type PipelineStep = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  detail: string;
};

type DepthModelOption = {
  id: string;
  label: string;
  tier: string;
  notes: string;
  farIsHigh: boolean;
};

const photoTargets: Array<{ value: PhotoTarget; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { value: 'depth-relief', label: '2.5D Relief STL', icon: ScanLine },
  { value: 'scene-diorama', label: 'Scene Diorama', icon: Boxes },
  { value: 'full-mesh', label: 'Full Mesh STL', icon: Cuboid },
];

const reliefDetailSamples = { min: 192, max: 900 };
const resolutionMultipliers = [1.5, 2, 3, 4];
const depthModels: DepthModelOption[] = [
  {
    id: 'depth-anything/Depth-Anything-V2-Large-hf',
    label: 'Depth Anything V2 Large',
    tier: 'Verified production',
    notes: 'Selected by the measured 30 mm relief and scene-depth regressions on the local CUDA path.',
    farIsHigh: false,
  },
];

const printerPresets: PrinterPreset[] = [
  { id: 'bambulab-p1s', label: 'Bambu Lab P1S', maxX: 256, maxY: 256, maxZ: 256, nozzleDiameter: 0.4 },
  { id: 'custom', label: 'Custom printer', maxX: 220, maxY: 220, maxZ: 220, nozzleDiameter: 0.4 },
];

const fallbackModelCatalog: ModelCatalog = {
  service: 'frontend-fallback',
  mode: 'planner-only',
  defaults: {
    selection: 'sam3-person-aware',
    frame_selection: 'uniform-frame-sampler',
    camera_pose: LIVE_VIDEO_CAMERA_MODEL,
    video_reconstruction: LIVE_VIDEO_RECONSTRUCTION_MODEL,
    image_to_mesh: 'triposg',
    stl_postprocess: 'trimesh-repair',
  },
  groups: {
    selection: [
      {
        id: 'sam3-person-aware',
        label: 'SAM 3 Person-aware',
        model: 'facebook/sam3',
        role: 'cached full-person concept masks with point-tracker fallback for other objects',
        availability: 'configured',
        notes: 'Pinned from the gated official checkpoint and measured on the shirt-omission regression.',
      },
      {
        id: 'turntable-grabcut',
        label: 'Turntable foreground',
        model: 'OpenCV GrabCut with temporal mask prior',
        role: 'automatic centered-object masks for controlled turntable videos',
        availability: 'configured',
      },
      {
        id: 'sam2.1-hiera-tiny',
        label: 'SAM 2.1 Tiny',
        model: 'facebook/sam2.1-hiera-tiny',
        role: 'whole-object point-prompted masks for still photos',
        availability: 'configured',
      },
      {
        id: 'detr-resnet-50-panoptic',
        label: 'DETR Panoptic',
        model: 'facebook/detr-resnet-50-panoptic',
        role: 'cached click-to-segment object masks for people, buildings, foliage, and scene parts',
        availability: 'configured',
      },
      {
        id: 'sam2.1-hiera-large',
        label: 'SAM 2.1 Hiera Large',
        model: 'facebook/sam2.1-hiera-large',
        role: 'promptable object and video mask propagation',
        availability: 'configured',
      },
      {
        id: 'sam2.1-hiera-tiny-video',
        label: 'SAM 2.1 Tiny Video',
        model: 'facebook/sam2.1-hiera-tiny',
        role: 'point-prompted temporal video masks',
        availability: 'setup-required',
      },
      {
        id: 'sam2.1-hiera-base-plus-video',
        label: 'SAM 2.1 Base+ Video',
        model: 'facebook/sam2.1-hiera-base-plus',
        role: 'higher-quality temporal video masks',
        availability: 'setup-required',
      },
      {
        id: 'grounding-dino-sam2',
        label: 'Grounding DINO + SAM 2.1',
        model: 'IDEA-Research/GroundingDINO + facebook/sam2.1',
        role: 'text-prompted object box plus mask',
        availability: 'adapter-planned',
      },
      {
        id: 'panoptic-detr',
        label: 'DETR Panoptic',
        model: 'facebook/detr-resnet-50-panoptic',
        role: 'click nearest panoptic segment',
        availability: 'configured',
      },
      {
        id: 'rmbg-2.0',
        label: 'RMBG 2.0',
        model: 'briaai/RMBG-2.0',
        role: 'automatic foreground matte',
        availability: 'adapter-planned',
      },
    ],
    frame_selection: [
      {
        id: 'uniform-frame-sampler',
        label: 'Uniform frame sampler',
        model: 'opencv-videoio',
        role: 'deterministic every-n-frame sampling',
        availability: 'configured',
      },
      {
        id: 'scenedetect-adaptive',
        label: 'PySceneDetect adaptive',
        model: 'scenedetect-adaptive',
        role: 'shot and motion change sampling',
        availability: 'adapter-planned',
      },
    ],
    camera_pose: [
      {
        id: LIVE_VIDEO_CAMERA_MODEL,
        label: 'Turntable orbit',
        model: 'frame-time orbit prior',
        role: 'fixed-camera rotating-object camera assignment',
        availability: 'configured',
      },
      {
        id: 'colmap-sift',
        label: 'COLMAP SIFT',
        model: 'COLMAP',
        role: 'camera matching and sparse reconstruction',
        availability: 'adapter-planned',
      },
      {
        id: 'hloc-lightglue',
        label: 'hloc + LightGlue',
        model: 'SuperPoint/DISK + LightGlue',
        role: 'learned feature matching for camera poses',
        availability: 'adapter-planned',
      },
      {
        id: 'vggt-camera',
        label: 'VGGT camera head',
        model: 'VGGT',
        role: 'feed-forward camera/depth/point prediction',
        availability: 'adapter-planned',
      },
    ],
    video_reconstruction: [
      {
        id: LIVE_VIDEO_RECONSTRUCTION_MODEL,
        label: 'Multiview visual hull',
        model: 'silhouette visual hull',
        role: 'deterministic turntable mesh from object silhouettes',
        availability: 'configured',
      },
      {
        id: 'colmap-openmvs',
        label: 'COLMAP + OpenMVS',
        model: 'COLMAP/OpenMVS',
        role: 'photogrammetry mesh',
        availability: 'adapter-planned',
      },
      {
        id: 'gaussian-splatting-mesh',
        label: 'Gaussian Splatting + mesh',
        model: '3D Gaussian Splatting + mesh extraction',
        role: 'splat reconstruction to repaired STL',
        availability: 'adapter-planned',
      },
      {
        id: 'vggt-fusion',
        label: 'VGGT fusion',
        model: 'VGGT',
        role: 'multi-frame depth/point fusion',
        availability: 'adapter-planned',
      },
      {
        id: 'dust3r-mast3r',
        label: 'DUSt3R/MASt3R',
        model: 'DUSt3R or MASt3R',
        role: 'dense correspondence and 3D point prediction',
        availability: 'adapter-planned',
      },
    ],
    image_to_mesh: [
      {
        id: 'triposg',
        label: 'TripoSG',
        model: 'TripoSG-style direct mesh',
        role: 'single image to mesh',
        availability: 'adapter-planned',
      },
      {
        id: 'hunyuan3d-shape',
        label: 'Hunyuan3D Shape',
        model: 'Hunyuan3D Shape',
        role: 'single image to shape mesh',
        availability: 'adapter-planned',
      },
      {
        id: 'triposr',
        label: 'TripoSR',
        model: 'TripoSR',
        role: 'single image sparse-view reconstruction',
        availability: 'adapter-planned',
      },
      {
        id: 'stable-fast-3d',
        label: 'Stable Fast 3D',
        model: 'SF3D',
        role: 'single image to textured mesh',
        availability: 'adapter-planned',
      },
      {
        id: 'spar3d',
        label: 'SPAR3D',
        model: 'SPAR3D',
        role: 'single image sparse 3D reconstruction',
        availability: 'adapter-planned',
      },
    ],
    stl_postprocess: [
      {
        id: 'trimesh-repair',
        label: 'Trimesh repair',
        model: 'trimesh',
        role: 'mesh cleanup and STL export',
        availability: 'configured',
      },
      {
        id: 'manifold3d',
        label: 'Manifold3D repair',
        model: 'manifold3d',
        role: 'watertight boolean/manifold conversion',
        availability: 'adapter-planned',
      },
      {
        id: 'pymeshlab-remesh',
        label: 'PyMeshLab remesh',
        model: 'pymeshlab',
        role: 'surface repair and decimation',
        availability: 'adapter-planned',
      },
    ],
  },
  metrics: [
    'watertightness',
    'manifoldness',
    'positive volume',
    'single component',
    'minimum printable thickness',
    'bbox aspect ratio',
    'surface Chamfer when ground truth exists',
  ],
};

function classNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ');
}

function mediaKindFromFile(file: File | null): MediaKind {
  if (file?.type.startsWith('video/')) return 'video';
  return 'photo';
}

function fileSizeLabel(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function secondsLabel(seconds?: number) {
  if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '-';
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
}

function timingLabel(key: string) {
  const labels: Record<string, string> = {
    depth_seconds: 'Depth',
    provider_seconds: 'Provider',
    stl_seconds: 'STL mesh',
    diagnostics_seconds: 'Checks',
    total_seconds: 'Total',
  };
  return labels[key] || key.replace(/_/g, ' ');
}

function depthPreloadLabel(status?: string) {
  if (status === 'ready') return 'Cached';
  if (status === 'downloading') return 'Downloading';
  if (status === 'error') return 'Failed';
  if (status === 'missing') return 'Not cached';
  return 'Checking';
}

function depthPreloadTone(status?: string) {
  if (status === 'ready') return 'border-emerald-700 bg-emerald-50 text-emerald-900';
  if (status === 'downloading') return 'border-blue-700 bg-blue-50 text-blue-900';
  if (status === 'error') return 'border-red-700 bg-red-50 text-red-900';
  return 'border-orange-700 bg-orange-50 text-orange-900';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function optionalString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function optionalNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function stringList(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
    : [];
}

function normalizeModelOption(value: unknown): ModelOption | null {
  if (!isRecord(value)) return null;
  const id = optionalString(value.id);
  if (!id) return null;
  return {
    id,
    label: optionalString(value.label) || id,
    model: optionalString(value.model),
    role: optionalString(value.role),
    availability: optionalString(value.availability),
    notes: optionalString(value.notes),
    local: typeof value.local === 'boolean' ? value.local : undefined,
    gpu_supported: typeof value.gpu_supported === 'boolean' ? value.gpu_supported : undefined,
  };
}

function normalizeModelOptions(value: unknown, fallback: ModelOption[]) {
  if (!Array.isArray(value)) return fallback;
  const options = value.map(normalizeModelOption).filter((option): option is ModelOption => Boolean(option));
  return options.length ? options : fallback;
}

function normalizeCatalog(data: unknown): ModelCatalog {
  const source = isRecord(data) ? data : {};
  const sourceDefaults = isRecord(source.defaults) ? source.defaults : {};
  const sourceGroups = isRecord(source.groups) ? source.groups : {};
  const defaults = {} as Record<ModelGroup, string>;
  const groups = {} as Record<ModelGroup, ModelOption[]>;

  MODEL_GROUPS.forEach((group) => {
    defaults[group] = optionalString(sourceDefaults[group]) || fallbackModelCatalog.defaults[group];
    groups[group] = normalizeModelOptions(sourceGroups[group], fallbackModelCatalog.groups[group]);
  });

  const metrics = stringList(source.metrics);
  return {
    service: optionalString(source.service) || fallbackModelCatalog.service,
    mode: optionalString(source.mode) || fallbackModelCatalog.mode,
    defaults,
    groups,
    metrics: metrics.length ? metrics : fallbackModelCatalog.metrics,
    notes: optionalString(source.notes) || fallbackModelCatalog.notes,
  };
}

function modelsFor(catalog: ModelCatalog, group: ModelGroup) {
  const options = catalog.groups?.[group];
  return Array.isArray(options) && options.length ? options : fallbackModelCatalog.groups[group];
}

function normalizeProviderReadiness(data: unknown) {
  if (!isRecord(data) || !Array.isArray(data.providers)) return {};
  return Object.fromEntries(
    data.providers.flatMap((value): Array<[string, ProviderReadiness]> => {
      if (!isRecord(value)) return [];
      const id = optionalString(value.id);
      if (!id) return [];
      const env = isRecord(value.env) ? value.env : {};
      return [[
        id,
        {
          id,
          label: optionalString(value.label),
          status: optionalString(value.status),
          runnable: typeof value.runnable === 'boolean' ? value.runnable : false,
          setup_errors: stringList(value.setup_errors),
          checks: isRecord(value.checks) ? value.checks : undefined,
          env: {
            timeout_seconds: optionalNumber(env.timeout_seconds),
            provider_dir_configured:
              typeof env.provider_dir_configured === 'boolean' ? env.provider_dir_configured : undefined,
            provider_python_configured:
              typeof env.provider_python_configured === 'boolean' ? env.provider_python_configured : undefined,
            provider_dir_env_names: stringList(env.provider_dir_env_names),
            provider_python_env_names: stringList(env.provider_python_env_names),
          },
        },
      ]];
    }),
  );
}

function normalizeSelectionResult(data: unknown): SelectionResult {
  if (!isRecord(data)) return {};
  const points = Array.isArray(data.points)
    ? data.points.flatMap((point) => {
        if (!isRecord(point)) return [];
        const x = optionalNumber(point.x);
        const y = optionalNumber(point.y);
        return x === undefined || y === undefined ? [] : [{ x, y }];
      })
    : undefined;
  return {
    job_id: optionalString(data.job_id),
    selected_image: optionalString(data.selected_image),
    selected_image_url: optionalString(data.selected_image_url),
    mask: optionalString(data.mask),
    mask_url: optionalString(data.mask_url),
    overlay: optionalString(data.overlay),
    overlay_url: optionalString(data.overlay_url),
    tint: optionalString(data.tint),
    tint_url: optionalString(data.tint_url),
    metadata_url: optionalString(data.metadata_url),
    points,
    model_id: optionalString(data.model_id),
    model_status: optionalString(data.model_status),
    model_error: optionalString(data.model_error) || null,
    selection_labels: stringList(data.selection_labels),
    mask_pixels: optionalNumber(data.mask_pixels),
    mask_coverage: optionalNumber(data.mask_coverage),
  };
}

function normalizeSelectionPrecomputeResult(data: unknown): SelectionPrecomputeResult {
  if (!isRecord(data)) return {};
  return {
    precompute_id: optionalString(data.precompute_id) || null,
    model_id: optionalString(data.model_id),
    model_status: optionalString(data.model_status),
    precompute_supported:
      typeof data.precompute_supported === 'boolean' ? data.precompute_supported : undefined,
    message: optionalString(data.message),
    segment_count: optionalNumber(data.segment_count),
    selection_labels: stringList(data.selection_labels),
    timings: isRecord(data.timings) ? (data.timings as StageTimings) : undefined,
  };
}

function resolveServiceUrl(baseUrl: string, value: unknown) {
  const path = optionalString(value);
  if (!path) return '';
  try {
    return new URL(path, `${baseUrl.replace(/\/+$/, '')}/`).toString();
  } catch {
    return '';
  }
}

function selectionModelSupportsPrecompute(modelId: string) {
  return (
    modelId === 'sam3-person-aware' ||
    modelId === 'facebook/sam3' ||
    modelId.startsWith('sam2') ||
    modelId === 'panoptic-detr' ||
    modelId === 'detr-resnet-50-panoptic' ||
    modelId.includes('detr-resnet-50-panoptic')
  );
}

function modelFor(catalog: ModelCatalog, group: ModelGroup, modelId: string) {
  return modelsFor(catalog, group).find((model) => model.id === modelId) || modelsFor(catalog, group)[0];
}

function modelLabel(catalog: ModelCatalog, group: ModelGroup, modelId: string) {
  return modelFor(catalog, group, modelId)?.label || modelId;
}

function failedChecksLabel(value: unknown) {
  if (Array.isArray(value) && value.length) return value.join(', ');
  return 'hard STL checks';
}

function providerReadinessLabel(readiness?: ProviderReadiness) {
  if (!readiness) return 'Unknown';
  return readiness.runnable ? 'Ready' : 'Setup missing';
}

function providerReadinessTone(readiness?: ProviderReadiness) {
  if (!readiness) return 'border-zinc-300 bg-zinc-50 text-zinc-700';
  return readiness.runnable
    ? 'border-emerald-700 bg-emerald-50 text-emerald-900'
    : 'border-red-700 bg-red-50 text-red-900';
}

function providerSetupMessage(readiness?: ProviderReadiness) {
  if (!readiness) return 'Provider preflight has not reported this model yet.';
  if (readiness.runnable) {
    const timeout = readiness.env?.timeout_seconds;
    return timeout ? `Configured for up to ${Math.round(timeout / 60)} minutes.` : 'Provider entrypoint is available.';
  }
  return readiness.setup_errors?.[0] || 'Provider setup is incomplete.';
}

function availabilityTone(availability?: string) {
  if (availability === 'configured') return 'border-emerald-700 bg-emerald-50 text-emerald-900';
  if (availability === 'adapter-planned') return 'border-blue-700 bg-blue-50 text-blue-900';
  return 'border-zinc-200 bg-zinc-50 text-zinc-600';
}

function availabilityLabel(availability?: string) {
  if (availability === 'adapter-planned') return 'planned';
  return availability || 'planned';
}

function StepIcon({ active, done }: { active?: boolean; done?: boolean }) {
  if (done) return <Check className="h-4 w-4 text-emerald-700" />;
  return <CircleDot className={classNames('h-4 w-4', active ? 'text-blue-700' : 'text-zinc-400')} />;
}

function workflowSteps({
  mediaKind,
  photoScope,
  photoTarget,
  videoScope,
  selectedFrameCount,
  frameStep,
  meshBackend,
  videoBackend,
}: {
  mediaKind: MediaKind;
  photoScope: PhotoScope;
  photoTarget: PhotoTarget;
  videoScope: VideoScope;
  selectedFrameCount: number;
  frameStep: number;
  meshBackend: string;
  videoBackend: string;
}): PipelineStep[] {
  if (mediaKind === 'photo' && photoTarget === 'depth-relief') {
    return [
      {
        icon: Upload,
        label: 'Import photo',
        detail: photoScope === 'object-selection' ? 'Object mask enabled' : 'Whole frame',
      },
      {
        icon: ScanLine,
        label: 'Estimate depth',
        detail: 'Local GPU depth model',
      },
      {
        icon: Layers3,
        label: 'Build relief',
        detail: 'Height field, backing plate, printable normals',
      },
      {
        icon: FileDown,
        label: 'Export STL',
        detail: 'Watertight 2.5D model',
      },
    ];
  }

  if (mediaKind === 'photo' && photoTarget === 'scene-diorama') {
    return [
      {
        icon: MousePointer2,
        label: 'Select scene layers',
        detail: photoScope === 'object-selection' ? 'Keep people, buildings, and props' : 'Use the whole frame',
      },
      {
        icon: ScanLine,
        label: 'Estimate scene depth',
        detail: 'Monocular depth with semantic ordering',
      },
      {
        icon: Boxes,
        label: 'Build diorama',
        detail: 'Layered facade geometry and hidden sides',
      },
      {
        icon: FileDown,
        label: 'Export GLB + STL',
        detail: 'Free-view scene and connected printable model',
      },
    ];
  }

  if (mediaKind === 'photo') {
    return [
      {
        icon: Upload,
        label: 'Import photo',
        detail: photoScope === 'object-selection' ? 'Selected object only' : 'Whole visible subject',
      },
      {
        icon: Cuboid,
        label: 'Generate full mesh',
        detail: meshBackend,
      },
      {
        icon: BadgeCheck,
        label: 'Repair STL',
        detail: 'Watertight, manifold, positive volume',
      },
    ];
  }

  if (videoScope === 'selected-frames') {
    return [
      {
        icon: Film,
        label: 'Import video',
        detail: `${selectedFrameCount} selected frames`,
      },
      {
        icon: MousePointer2,
        label: 'Keep full frames',
        detail: 'Camera matching keeps background/keypoints',
      },
      {
        icon: Sparkles,
        label: 'Mask selected object',
        detail: 'Masks constrain the reconstruction target',
      },
      {
        icon: Boxes,
        label: 'Fuse mesh',
        detail: `${videoBackend}, then STL repair`,
      },
    ];
  }

  return [
    {
      icon: Film,
      label: 'Import video',
      detail: `Every ${frameStep} frame${frameStep === 1 ? '' : 's'}`,
    },
    {
      icon: Camera,
      label: 'Recover cameras',
      detail: 'Global scene reconstruction',
    },
    {
      icon: Boxes,
      label: 'Generate 3D scene',
      detail: videoBackend,
    },
    {
      icon: FileDown,
      label: 'Convert to STL',
      detail: 'Mesh extraction, repair, scale',
    },
  ];
}

type ProductionModelFieldProps = {
  controlId: string;
  label: string;
  modelId: string;
  modelLabel: string;
  detail?: string;
  className?: string;
};

function ProductionModelField({
  controlId,
  label,
  modelId,
  modelLabel: displayName,
  detail = 'Evidence-backed production choice',
  className,
}: ProductionModelFieldProps) {
  return (
    <div className={classNames('min-w-0', className)}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={controlId} className="text-sm font-medium text-zinc-700">
          {label}
        </label>
        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700">
          <BadgeCheck className="h-3.5 w-3.5" />
          Production
        </span>
      </div>
      <select
        id={controlId}
        aria-label={label}
        value={modelId}
        disabled
        className="mt-2 h-10 w-full appearance-none border border-zinc-300 bg-zinc-50 px-3 pr-9 text-sm font-medium text-zinc-800 disabled:cursor-default disabled:opacity-100"
      >
        <option value={modelId}>{displayName}</option>
      </select>
      {detail && <span className="mt-1 block text-xs text-zinc-500">{detail}</span>}
    </div>
  );
}
export default function Home() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [videoBackendUrl, setVideoBackendUrl] = useState(DEFAULT_VIDEO_BACKEND_URL);
  const [backendConfigReady, setBackendConfigReady] = useState(false);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog>(fallbackModelCatalog);
  const [catalogState, setCatalogState] = useState<CatalogState>('loading');
  const [providerReadiness, setProviderReadiness] = useState<Record<string, ProviderReadiness>>({});
  const [providerReadinessState, setProviderReadinessState] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [mediaDimensions, setMediaDimensions] = useState<MediaDimensions | null>(null);
  const [mediaKind, setMediaKind] = useState<MediaKind>('photo');
  const [photoScope, setPhotoScope] = useState<PhotoScope>('whole-image');
  const [photoTarget, setPhotoTarget] = useState<PhotoTarget>('depth-relief');
  const [videoTarget, setVideoTarget] = useState<VideoTarget>('turntable-mesh');
  const [videoScope, setVideoScope] = useState<VideoScope>('selected-frames');
  const [selectedFrameCount, setSelectedFrameCount] = useState(12);
  const [frameStep, setFrameStep] = useState(8);
  const [depthModel, setDepthModel] = useState(PRODUCTION_DEPTH_MODEL);
  const [depthScale, setDepthScale] = useState(30);
  const [baseThickness, setBaseThickness] = useState(2.4);
  const [sceneDepth, setSceneDepth] = useState(64);
  const [sceneSubjectDepth, setSceneSubjectDepth] = useState(12);
  const [sceneFacadeDetail, setSceneFacadeDetail] = useState(0.8);
  const [sceneDepthCompression, setSceneDepthCompression] = useState(0.65);
  const [reliefPolarity, setReliefPolarity] = useState<ReliefPolarity>('raised-print');
  const [detailSmoothing, setDetailSmoothing] = useState(0.35);
  const [featureBoost, setFeatureBoost] = useState(0.8);
  const [printableFeatureDepth, setPrintableFeatureDepth] = useState(0.4);
  const [featureBridgeDepth, setFeatureBridgeDepth] = useState(0.8);
  const [backgroundDetailBoost, setBackgroundDetailBoost] = useState(2.4);
  const [backgroundPhotoDetail, setBackgroundPhotoDetail] = useState(0.60);
  const [trimTopBackground, setTrimTopBackground] = useState(true);
  const [reliefGamma, setReliefGamma] = useState(0.75);
  const [baseBorderPx, setBaseBorderPx] = useState(2);
  const [meshResolutionMultiplier, setMeshResolutionMultiplier] = useState(2);
  const [printerPreset, setPrinterPreset] = useState<PrinterPresetId>('bambulab-p1s');
  const [printerMaxX, setPrinterMaxX] = useState(256);
  const [printerMaxY, setPrinterMaxY] = useState(256);
  const [printerMaxZ, setPrinterMaxZ] = useState(256);
  const [printerNozzleDiameter, setPrinterNozzleDiameter] = useState(0.4);
  const [printerNozzleInput, setPrinterNozzleInput] = useState('0.4');
  const [printerClearance, setPrinterClearance] = useState(0);
  const [printScalePercent, setPrintScalePercent] = useState(100);
  const [meshBackend, setMeshBackend] = useState(fallbackModelCatalog.defaults.image_to_mesh);
  const [selectionModel, setSelectionModel] = useState(fallbackModelCatalog.defaults.selection);
  const [selectionPoints, setSelectionPoints] = useState<SelectionPoint[]>([]);
  const [selectionResult, setSelectionResult] = useState<SelectionResult | null>(null);
  const [selectionPreviewUrl, setSelectionPreviewUrl] = useState('');
  const [selectionAppliedFile, setSelectionAppliedFile] = useState<File | null>(null);
  const [selectionState, setSelectionState] = useState<'idle' | 'editing' | 'applying' | 'ready' | 'error'>('idle');
  const [hoverSelection, setHoverSelection] = useState<SelectionMask | null>(null);
  const [hoverSelectionState, setHoverSelectionState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [selectedMasks, setSelectedMasks] = useState<SelectionMask[]>([]);
  const [selectionPrecompute, setSelectionPrecompute] = useState<SelectionPrecomputeResult | null>(null);
  const [selectionPrecomputeState, setSelectionPrecomputeState] = useState<'idle' | 'loading' | 'ready' | 'unsupported' | 'error'>('idle');
  const [selectionPrecomputeError, setSelectionPrecomputeError] = useState('');
  const [frameSelectionModel, setFrameSelectionModel] = useState(fallbackModelCatalog.defaults.frame_selection);
  const [cameraPoseModel, setCameraPoseModel] = useState(fallbackModelCatalog.defaults.camera_pose);
  const [videoBackend, setVideoBackend] = useState(fallbackModelCatalog.defaults.video_reconstruction);
  const [stlPostprocessModel, setStlPostprocessModel] = useState(fallbackModelCatalog.defaults.stl_postprocess);
  const [processedSTL, setProcessedSTL] = useState('');
  const [processedSceneGLB, setProcessedSceneGLB] = useState('');
  const [diagnosticsUrl, setDiagnosticsUrl] = useState('');
  const [stlDiagnostics, setStlDiagnostics] = useState<StlDiagnostics | null>(null);
  const [backendRuntime, setBackendRuntime] = useState<RuntimeInfo | null>(null);
  const [backendRuntimeState, setBackendRuntimeState] = useState<'checking' | 'ready' | 'unavailable'>('checking');
  const [stageTimings, setStageTimings] = useState<StageTimings | null>(null);
  const [depthRunMetadata, setDepthRunMetadata] = useState<DepthRunMetadata | null>(null);
  const [depthProPreload, setDepthProPreload] = useState<DepthPreloadStatus | null>(null);
  const [depthProPreloadState, setDepthProPreloadState] = useState<'checking' | 'ready' | 'unavailable'>('checking');
  const [depthProPreloadStarting, setDepthProPreloadStarting] = useState(false);
  const [depthProPreloadPollKey, setDepthProPreloadPollKey] = useState(0);
  const [completedPreview, setCompletedPreview] = useState('');
  const [plannerResponse, setPlannerResponse] = useState<PlannerResponse | null>(null);
  const [runState, setRunState] = useState<RunState>('idle');
  const [statusText, setStatusText] = useState('Ready');
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectionImageRef = useRef<HTMLImageElement>(null);
  const hoverTimerRef = useRef<number | undefined>(undefined);
  const hoverRequestIdRef = useRef(0);
  const lastHoverPointRef = useRef<{ x: number; y: number } | null>(null);
  const precomputeRequestIdRef = useRef(0);
  const selectionSourceGenerationRef = useRef(0);

  const selectedMeshReadiness = providerReadiness[meshBackend];
  const currentPrinterPreset = printerPresets.find((preset) => preset.id === printerPreset) || printerPresets[0];
  const parsedPrinterNozzleDiameter = Number(printerNozzleInput);
  const effectivePrinterNozzleDiameter =
    Number.isFinite(parsedPrinterNozzleDiameter)
    && parsedPrinterNozzleDiameter >= 0.2
    && parsedPrinterNozzleDiameter <= 2
      ? parsedPrinterNozzleDiameter
      : printerNozzleDiameter;
  const minimumFeatureSize = Math.max(0.4, effectivePrinterNozzleDiameter * 2);

  useEffect(() => {
    setDepthModel(PRODUCTION_DEPTH_MODEL);
    setMeshBackend(PRODUCTION_IMAGE_TO_MESH_MODEL);
    setCameraPoseModel(LIVE_VIDEO_CAMERA_MODEL);
    setVideoBackend(LIVE_VIDEO_RECONSTRUCTION_MODEL);
    setStlPostprocessModel(FULL_MESH_RUNNER_STL_POSTPROCESS);

    if (mediaKind === 'photo') {
      setSelectionModel(PRODUCTION_PHOTO_SELECTION_MODEL);
      return;
    }
    if (videoTarget === 'tracked-relief') {
      setSelectionModel(PRODUCTION_TRACKED_VIDEO_SELECTION_MODEL);
      setFrameSelectionModel(PRODUCTION_TRACKED_FRAME_MODEL);
      return;
    }
    setSelectionModel(PRODUCTION_TURNTABLE_SELECTION_MODEL);
    setFrameSelectionModel(PRODUCTION_TURNTABLE_FRAME_MODEL);
  }, [mediaKind, videoTarget]);

  const printVolume = useMemo<PrintVolumePlan>(() => {
    const usableX = Math.max(1, Math.floor(printerMaxX - printerClearance * 2));
    const usableY = Math.max(1, Math.floor(printerMaxY - printerClearance * 2));
    const usableZ = Math.max(1, Math.floor(printerMaxZ - printerClearance));
    const maxTargetDimension = Math.max(1, Math.floor(Math.min(usableX, usableY)));
    const scaledTargetDimension = Math.max(1, Math.floor(maxTargetDimension * (printScalePercent / 100)));
    const targetDimension = Math.min(maxTargetDimension, Math.max(10, scaledTargetDimension));
    return {
      preset: printerPreset,
      label: currentPrinterPreset.label,
      max_x_mm: printerMaxX,
      max_y_mm: printerMaxY,
      max_z_mm: printerMaxZ,
      clearance_mm: printerClearance,
      usable_x_mm: usableX,
      usable_y_mm: usableY,
      usable_z_mm: usableZ,
      max_target_dimension_mm: maxTargetDimension,
      target_dimension_mm: targetDimension,
      print_scale_percent: printScalePercent,
      max_relief_height_mm: Math.max(1, usableZ - baseThickness),
      nozzle_diameter_mm: effectivePrinterNozzleDiameter,
      minimum_feature_mm: minimumFeatureSize,
    };
  }, [printerPreset, currentPrinterPreset.label, printerMaxX, printerMaxY, printerMaxZ, printerClearance, printScalePercent, baseThickness, effectivePrinterNozzleDiameter, minimumFeatureSize]);
  const printFootprint = useMemo<MediaDimensions | null>(() => {
    if (!mediaDimensions || mediaDimensions.width <= 0 || mediaDimensions.height <= 0) return null;
    const aspectRatio = mediaDimensions.width / mediaDimensions.height;
    if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) return null;
    if (aspectRatio >= 1) {
      return {
        width: printVolume.target_dimension_mm,
        height: printVolume.target_dimension_mm / aspectRatio,
      };
    }
    return {
      width: printVolume.target_dimension_mm * aspectRatio,
      height: printVolume.target_dimension_mm,
    };
  }, [mediaDimensions, printVolume.target_dimension_mm]);
  const effectiveReliefHeight = Math.min(depthScale, printVolume.max_relief_height_mm);
  const reliefSliderMax = Math.max(12, Math.min(40, Math.floor(printVolume.max_relief_height_mm)));
  const reliefTargetDimension = Math.min(
    reliefDetailSamples.max,
    Math.max(reliefDetailSamples.min, Math.round(printVolume.max_target_dimension_mm * meshResolutionMultiplier)),
  );
  const reliefPrintableDimension = Math.min(
    reliefTargetDimension,
    Math.max(2, Math.floor(printVolume.max_target_dimension_mm / (minimumFeatureSize / 2)) + 1),
  );
  const reliefSamplePitch = printVolume.target_dimension_mm / Math.max(1, reliefPrintableDimension - 1);
  const sceneDepthSliderMax = Math.max(24, Math.min(160, Math.floor(printVolume.usable_y_mm)));
  const effectiveSceneDepth = Math.min(sceneDepth, sceneDepthSliderMax);
  const sceneMaxSamples = Math.min(
    320,
    Math.max(96, Math.round(printVolume.target_dimension_mm / (minimumFeatureSize / 2)) + 1),
  );
  const selectedDepthModel = depthModels.find((model) => model.id === depthModel) || depthModels[0];
  const reliefInvert = reliefPolarity === 'raised-print' ? selectedDepthModel.farIsHigh : !selectedDepthModel.farIsHigh;

  const applyPrinterPreset = (presetId: PrinterPresetId) => {
    const preset = printerPresets.find((candidate) => candidate.id === presetId) || printerPresets[0];
    setPrinterPreset(preset.id);
    setPrinterMaxX(preset.maxX);
    setPrinterMaxY(preset.maxY);
    setPrinterMaxZ(preset.maxZ);
    setPrinterNozzleDiameter(preset.nozzleDiameter);
    setPrinterNozzleInput(String(preset.nozzleDiameter));
  };

  const commitPrinterNozzleDiameter = () => {
    const parsed = Number(printerNozzleInput);
    const nozzle = Number.isFinite(parsed)
      ? Math.min(2, Math.max(0.2, parsed))
      : printerNozzleDiameter;
    setPrinterNozzleDiameter(nozzle);
    setPrinterNozzleInput(String(nozzle));
    setPrinterPreset('custom');
  };

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    setBackendUrl(query.get('backendUrl') || CONFIGURED_BACKEND_URL);
    setVideoBackendUrl(query.get('videoBackendUrl') || CONFIGURED_VIDEO_BACKEND_URL);
    setBackendConfigReady(true);
  }, []);

  useEffect(() => {
    if (!backendConfigReady) return;

    let cancelled = false;
    let retryTimer: number | undefined;

    const loadBackendRuntime = async () => {
      setBackendRuntimeState('checking');
      let timeout: number | undefined;
      try {
        const controller = new AbortController();
        timeout = window.setTimeout(() => controller.abort(), 2500);
        const response = await fetch(`${backendUrl}/health`, { signal: controller.signal });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Backend health ${response.status}`);
        const data = (await response.json()) as BackendHealth;
        if (!cancelled) {
          setBackendRuntime(data.runtime || null);
          setBackendRuntimeState('ready');
        }
      } catch {
        if (timeout) window.clearTimeout(timeout);
        if (!cancelled) {
          setBackendRuntime(null);
          setBackendRuntimeState('unavailable');
          retryTimer = window.setTimeout(loadBackendRuntime, 5000);
        }
      }
    };

    loadBackendRuntime();
    return () => {
      cancelled = true;
      if (retryTimer) window.clearTimeout(retryTimer);
    };
  }, [backendUrl, backendConfigReady]);

  useEffect(() => {
    if (!backendConfigReady) return;

    let cancelled = false;
    let pollTimer: number | undefined;

    const loadDepthProPreloadStatus = async () => {
      setDepthProPreloadState('checking');
      let timeout: number | undefined;
      try {
        const controller = new AbortController();
        timeout = window.setTimeout(() => controller.abort(), 4000);
        const response = await fetch(`${backendUrl}/depth/preload/depthpro/status`, { signal: controller.signal });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Depth Pro preload ${response.status}`);
        const data = (await response.json()) as DepthPreloadStatus;
        if (!cancelled) {
          setDepthProPreload(data);
          setDepthProPreloadState('ready');
          if (data.status === 'downloading') {
            pollTimer = window.setTimeout(loadDepthProPreloadStatus, 2500);
          }
        }
      } catch {
        if (timeout) window.clearTimeout(timeout);
        if (!cancelled) {
          setDepthProPreload(null);
          setDepthProPreloadState('unavailable');
        }
      }
    };

    loadDepthProPreloadStatus();
    return () => {
      cancelled = true;
      if (pollTimer) window.clearTimeout(pollTimer);
    };
  }, [backendUrl, backendConfigReady, depthProPreloadPollKey]);

  useEffect(() => {
    let cancelled = false;

    const loadModelCatalog = async () => {
      setCatalogState('loading');
      try {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 2500);
        const response = await fetch(`${videoBackendUrl}/models`, { signal: controller.signal });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Model planner ${response.status}`);
        const data: unknown = await response.json();
        if (!cancelled) {
          setModelCatalog(normalizeCatalog(data));
          setCatalogState('ready');
        }
      } catch {
        if (!cancelled) {
          setModelCatalog(fallbackModelCatalog);
          setCatalogState('fallback');
        }
      }
    };

    loadModelCatalog();
    return () => {
      cancelled = true;
    };
  }, [videoBackendUrl]);

  useEffect(() => {
    let cancelled = false;

    const loadProviderReadiness = async () => {
      setProviderReadinessState('loading');
      try {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 2500);
        const response = await fetch(`${videoBackendUrl}/providers/image-to-mesh`, { signal: controller.signal });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Provider preflight ${response.status}`);
        const data: unknown = await response.json();
        const nextReadiness = normalizeProviderReadiness(data);
        if (!cancelled) {
          setProviderReadiness(nextReadiness);
          setProviderReadinessState('ready');
        }
      } catch {
        if (!cancelled) {
          setProviderReadiness({});
          setProviderReadinessState('unavailable');
        }
      }
    };

    loadProviderReadiness();
    return () => {
      cancelled = true;
    };
  }, [videoBackendUrl]);

  useEffect(() => {
    selectionSourceGenerationRef.current += 1;
    hoverRequestIdRef.current += 1;
    setMediaDimensions(null);
    if (!file) {
      setPreviewUrl('');
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    setMediaKind(mediaKindFromFile(file));
    setProcessedSTL('');
    setProcessedSceneGLB('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setCompletedPreview('');
    setSelectionPoints([]);
    setSelectionResult(null);
    setSelectionPreviewUrl('');
    setSelectionAppliedFile(null);
    setSelectionState('idle');
    setHoverSelection(null);
    setHoverSelectionState('idle');
    setSelectedMasks([]);
    setSelectionPrecompute(null);
    setSelectionPrecomputeState('idle');
    setSelectionPrecomputeError('');
    precomputeRequestIdRef.current += 1;
    setPlannerResponse(null);
    setRunState('idle');
    setStatusText('Ready');
    setError('');
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  useEffect(() => {
    selectionSourceGenerationRef.current += 1;
    hoverRequestIdRef.current += 1;
  }, [selectionModel]);

  const steps = useMemo(
    () =>
      workflowSteps({
        mediaKind,
        photoScope,
        photoTarget,
        videoScope,
        selectedFrameCount,
        frameStep,
        meshBackend: modelLabel(modelCatalog, 'image_to_mesh', meshBackend),
        videoBackend: modelLabel(modelCatalog, 'video_reconstruction', videoBackend),
      }),
    [
      mediaKind,
      photoScope,
      photoTarget,
      videoScope,
      selectedFrameCount,
      frameStep,
      meshBackend,
      videoBackend,
      modelCatalog,
    ],
  );

  const jobPlan = useMemo(
    () => ({
      input: {
        file_name: file?.name || '',
        media_type: mediaKind,
        bytes: file?.size || 0,
      },
      services: {
        depth_relief_backend: backendUrl,
        video_selection_planner: videoBackendUrl,
        model_catalog: modelCatalog.service,
        planner_mode: modelCatalog.mode,
      },
      print_volume: printVolume,
      models: {
        selection: selectionModel,
        frame_selection: mediaKind === 'video' ? frameSelectionModel : null,
        camera_pose: mediaKind === 'video' ? cameraPoseModel : null,
        video_reconstruction: mediaKind === 'video' ? videoBackend : null,
        image_to_mesh: mediaKind === 'photo' && photoTarget === 'full-mesh' ? meshBackend : null,
        stl_postprocess: stlPostprocessModel,
      },
      model_labels: {
        selection: modelLabel(modelCatalog, 'selection', selectionModel),
        frame_selection: modelLabel(modelCatalog, 'frame_selection', frameSelectionModel),
        camera_pose: modelLabel(modelCatalog, 'camera_pose', cameraPoseModel),
        video_reconstruction: modelLabel(modelCatalog, 'video_reconstruction', videoBackend),
        image_to_mesh: modelLabel(modelCatalog, 'image_to_mesh', meshBackend),
        stl_postprocess: modelLabel(modelCatalog, 'stl_postprocess', stlPostprocessModel),
      },
      route:
        mediaKind === 'photo'
          ? {
              scope: photoScope,
              target: photoTarget,
              selection_model: photoScope === 'object-selection' ? selectionModel : null,
              selection:
                photoScope === 'object-selection'
                  ? {
                      kept_object_count: selectedMasks.length,
                      edited_image_ready: Boolean(selectionAppliedFile),
                      preserves_original_image: photoTarget === 'scene-diorama',
                      status: selectionState,
                      hover_status: hoverSelectionState,
                      model_status: selectionResult?.model_status || null,
                    }
                  : null,
              depth: {
                provider: 'transformers',
                model: depthModel,
                model_label: selectedDepthModel.label,
                model_tier: selectedDepthModel.tier,
                depth_values: selectedDepthModel.farIsHigh ? 'farther pixels are higher' : 'nearer pixels are higher',
                relief_height_mm: depthScale,
                effective_relief_height_mm: effectiveReliefHeight,
                base_thickness_mm: baseThickness,
                target_dimension_mm: printVolume.target_dimension_mm,
                max_xy_size_mm: printVolume.target_dimension_mm,
                mesh_resolution_dimension: reliefTargetDimension,
                printable_mesh_dimension: reliefPrintableDimension,
                resolution_multiplier: meshResolutionMultiplier,
                sample_pitch_mm: Number(reliefSamplePitch.toFixed(3)),
                nozzle_diameter_mm: effectivePrinterNozzleDiameter,
                minimum_feature_mm: minimumFeatureSize,
                max_relief_slope: 2.0,
                detail_basis_mm: printVolume.max_target_dimension_mm,
                polarity: reliefPolarity,
                invert_depth: reliefInvert,
                smoothing_sigma: detailSmoothing,
                feature_boost: featureBoost,
                relief_gamma: reliefGamma,
                base_border_px: baseBorderPx,
              },
              full_mesh:
                photoTarget === 'full-mesh'
                  ? {
                      mesh_backend: meshBackend,
                      mesh_backend_label: modelLabel(modelCatalog, 'image_to_mesh', meshBackend),
                      mesh_provider_readiness: selectedMeshReadiness
                        ? {
                            status: selectedMeshReadiness.status,
                            runnable: Boolean(selectedMeshReadiness.runnable),
                            setup_errors: selectedMeshReadiness.setup_errors || [],
                          }
                        : { status: providerReadinessState, runnable: null, setup_errors: [] },
                      stl_postprocess: stlPostprocessModel,
                      output: 'watertight STL',
                    }
                  : null,
              scene_reconstruction:
                photoTarget === 'scene-diorama'
                  ? {
                      mode: 'single-photo-layered-diorama',
                      selected_layer_count: photoScope === 'object-selection' ? selectedMasks.length : 0,
                      scene_depth_mm: effectiveSceneDepth,
                      subject_depth_mm: sceneSubjectDepth,
                      facade_detail_mm: sceneFacadeDetail,
                      depth_compression: sceneDepthCompression,
                      max_samples: sceneMaxSamples,
                      outputs: ['camera-free GLB', 'connected printable STL'],
                    }
                  : null,
            }
          : {
              target: videoTarget,
              scope: videoScope,
              selected_frames: videoScope === 'selected-frames' ? selectedFrameCount : null,
              frame_step: videoScope === 'everything' ? frameStep : null,
              reconstruction_backend: videoBackend,
              reconstruction_backend_label: modelLabel(modelCatalog, 'video_reconstruction', videoBackend),
              selection_model: selectionModel,
              frame_selection_model: frameSelectionModel,
              camera_pose_model: cameraPoseModel,
              stl_postprocess: stlPostprocessModel,
              preserve_full_frames_for_camera_matching: videoScope === 'selected-frames',
              object_masks_for_training_target: videoScope === 'selected-frames',
              output: 'mesh repaired to STL',
            },
      metrics: modelCatalog.metrics,
    }),
    [
      file,
      backendUrl,
      videoBackendUrl,
      mediaKind,
      photoScope,
      photoTarget,
      depthModel,
      depthScale,
      baseThickness,
      sceneSubjectDepth,
      sceneFacadeDetail,
      sceneDepthCompression,
      meshBackend,
      selectionModel,
      selectedMasks.length,
      selectionAppliedFile,
      selectionState,
      hoverSelectionState,
      selectionResult,
      frameSelectionModel,
      cameraPoseModel,
      videoTarget,
      videoScope,
      selectedFrameCount,
      frameStep,
      videoBackend,
      stlPostprocessModel,
      modelCatalog,
      printVolume,
      effectiveReliefHeight,
      reliefTargetDimension,
      reliefPrintableDimension,
      meshResolutionMultiplier,
      reliefSamplePitch,
      effectivePrinterNozzleDiameter,
      minimumFeatureSize,
      effectiveSceneDepth,
      sceneMaxSamples,
      reliefPolarity,
      reliefInvert,
      detailSmoothing,
      featureBoost,
      reliefGamma,
      baseBorderPx,
      selectedDepthModel,
      selectedMeshReadiness,
      providerReadinessState,
    ],
  );

  const displayedPlan = useMemo(
    () => (plannerResponse ? { ...jobPlan, planner_response: plannerResponse } : jobPlan),
    [jobPlan, plannerResponse],
  );

  useEffect(() => {
    setPlannerResponse(null);
  }, [jobPlan]);

  const activeModelStack = useMemo(() => {
    const stack: Array<{ label: string; model: ModelOption }> = [];
    const depthOption: ModelOption = {
      id: selectedDepthModel.id,
      label: selectedDepthModel.label,
      model: selectedDepthModel.id,
      role: selectedDepthModel.notes,
      availability: 'configured',
    };
    if (mediaKind === 'video') {
      stack.push(
        { label: 'Frames', model: modelFor(modelCatalog, 'frame_selection', frameSelectionModel) },
        { label: 'Selection', model: modelFor(modelCatalog, 'selection', selectionModel) },
      );
      if (videoTarget === 'tracked-relief') {
        stack.push({ label: 'Depth', model: depthOption });
      } else {
        stack.push(
          { label: 'Camera', model: modelFor(modelCatalog, 'camera_pose', cameraPoseModel) },
          { label: 'Reconstruct', model: modelFor(modelCatalog, 'video_reconstruction', videoBackend) },
        );
      }
      stack.push({ label: 'Repair', model: modelFor(modelCatalog, 'stl_postprocess', stlPostprocessModel) });
      return stack;
    }

    if (photoScope === 'object-selection') {
      stack.push({ label: 'Selection', model: modelFor(modelCatalog, 'selection', selectionModel) });
    }
    if (photoTarget === 'full-mesh') {
      stack.push({ label: 'Mesh', model: modelFor(modelCatalog, 'image_to_mesh', meshBackend) });
    } else {
      stack.push({ label: 'Depth', model: depthOption });
    }
    stack.push({ label: 'Repair', model: modelFor(modelCatalog, 'stl_postprocess', stlPostprocessModel) });
    return stack;
  }, [
    mediaKind,
    photoScope,
    photoTarget,
    videoTarget,
    modelCatalog,
    frameSelectionModel,
    selectionModel,
    cameraPoseModel,
    videoBackend,
    meshBackend,
    stlPostprocessModel,
    selectedDepthModel,
  ]);

  const backendAssetUrl = (path?: string) => {
    return resolveServiceUrl(backendUrl, path);
  };

  const selectionPointFromEvent = (event: React.MouseEvent<HTMLImageElement> | React.PointerEvent<HTMLImageElement>) => {
    const imageElement = selectionImageRef.current || event.currentTarget;
    const rect = imageElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  };

  const selectionMaskFromResponse = (data: SelectionResult, point: { x: number; y: number }): SelectionMask => {
    const tintUrl = backendAssetUrl(data.tint_url);
    const maskPath = data.mask || data.mask_url || '';
    if (!tintUrl || !maskPath) throw new Error('Selection response did not include a preview mask.');
    return {
      ...data,
      id: `${Date.now()}-${Math.round(point.x * 10000)}-${Math.round(point.y * 10000)}`,
      x: point.x,
      y: point.y,
      tintUrl,
      maskPath,
    };
  };

  const maskDistance = (mask: SelectionMask | null, point: { x: number; y: number }) => {
    if (!mask) return Number.POSITIVE_INFINITY;
    return Math.hypot(mask.x - point.x, mask.y - point.y);
  };

  useEffect(() => {
    if (!file || mediaKind !== 'photo' || photoScope !== 'object-selection') {
      setSelectionPrecompute(null);
      setSelectionPrecomputeState('idle');
      setSelectionPrecomputeError('');
      precomputeRequestIdRef.current += 1;
      return;
    }

    if (!selectionModelSupportsPrecompute(selectionModel)) {
      setSelectionPrecompute(null);
      setSelectionPrecomputeState('unsupported');
      setSelectionPrecomputeError('');
      precomputeRequestIdRef.current += 1;
      return;
    }

    let cancelled = false;
    const requestId = ++precomputeRequestIdRef.current;

    const precomputeSelection = async () => {
      setSelectionPrecompute(null);
      setSelectionPrecomputeError('');
      setSelectionPrecomputeState('loading');
      setHoverSelection(null);
      setHoverSelectionState('idle');
      try {
        const formData = new FormData();
        formData.append('file', file, file.name || 'photo.jpg');
        formData.append('model_id', selectionModel);
        formData.append('device', 'auto');
        const response = await fetch(`${backendUrl}/selection/precompute`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          let message = `Selection precompute ${response.status}`;
          try {
            const details = await response.json();
            if (details?.detail) message = String(details.detail);
          } catch {
            // Keep the status-based message when the backend does not return JSON.
          }
          throw new Error(message);
        }

        const data = normalizeSelectionPrecomputeResult(await response.json());
        if (cancelled || requestId !== precomputeRequestIdRef.current) return;
        setSelectionPrecompute(data);
        setSelectionPrecomputeState(data.precompute_id ? 'ready' : 'unsupported');
      } catch (precomputeError) {
        if (cancelled || requestId !== precomputeRequestIdRef.current) return;
        setSelectionPrecompute(null);
        setSelectionPrecomputeState('error');
        setSelectionPrecomputeError(precomputeError instanceof Error ? precomputeError.message : String(precomputeError));
      }
    };

    precomputeSelection();
    return () => {
      cancelled = true;
    };
  }, [file, mediaKind, photoScope, selectionModel, backendUrl]);

  const clearObjectSelection = () => {
    if (hoverTimerRef.current) window.clearTimeout(hoverTimerRef.current);
    selectionSourceGenerationRef.current += 1;
    hoverRequestIdRef.current += 1;
    setSelectionPoints([]);
    setSelectionResult(null);
    setSelectionPreviewUrl('');
    setSelectionAppliedFile(null);
    setSelectionState('idle');
    setHoverSelection(null);
    setHoverSelectionState('idle');
    setSelectedMasks([]);
    setCompletedPreview('');
    setProcessedSTL('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setRunState('idle');
    setStatusText('Ready');
    setError('');
  };

  const invalidateSelectionResult = () => {
    selectionSourceGenerationRef.current += 1;
    hoverRequestIdRef.current += 1;
    setSelectionResult(null);
    setSelectionPreviewUrl('');
    setSelectionAppliedFile(null);
    setCompletedPreview('');
    setProcessedSTL('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setRunState('idle');
    setError('');
  };

  const requestSelectionMask = async (point: { x: number; y: number }, mode: 'hover' | 'click') => {
    if (!file) return null;
    if (mode === 'hover' && selectionPrecomputeState === 'loading') {
      setHoverSelectionState('loading');
      return null;
    }
    const sourceGeneration = selectionSourceGenerationRef.current;
    const requestId = ++hoverRequestIdRef.current;
    if (mode === 'hover') setHoverSelectionState('loading');
    try {
      const pointsPayload = JSON.stringify([{ x: point.x, y: point.y }]);
      const liveMaskRequest = () => {
        const liveFormData = new FormData();
        liveFormData.append('file', file, file.name || 'photo.jpg');
        liveFormData.append('model_id', selectionModel);
        liveFormData.append('mask_max_dimension', mode === 'hover' ? '768' : '1024');
        liveFormData.append('points_json', pointsPayload);
        return fetch(`${backendUrl}/selection/mask`, {
          method: 'POST',
          body: liveFormData,
        });
      };

      let response: Response;
      if (selectionPrecompute?.precompute_id) {
        const cachedFormData = new FormData();
        cachedFormData.append('precompute_id', selectionPrecompute.precompute_id);
        cachedFormData.append('points_json', pointsPayload);
        response = await fetch(`${backendUrl}/selection/precomputed_mask`, {
          method: 'POST',
          body: cachedFormData,
        });
        if (
          sourceGeneration !== selectionSourceGenerationRef.current
          || requestId !== hoverRequestIdRef.current
        ) return null;
        if (response.status === 404 || response.status === 410) {
          setSelectionPrecompute(null);
          setSelectionPrecomputeState('error');
          setSelectionPrecomputeError('Cached segmentation expired; using live masks');
          response = await liveMaskRequest();
        }
      } else {
        response = await liveMaskRequest();
      }

      if (
        sourceGeneration !== selectionSourceGenerationRef.current
        || requestId !== hoverRequestIdRef.current
      ) return null;
      if (!response.ok) throw new Error(`Selection preview ${response.status}`);
      const data = normalizeSelectionResult(await response.json());
      if (
        sourceGeneration !== selectionSourceGenerationRef.current
        || requestId !== hoverRequestIdRef.current
      ) return null;
      const mask = selectionMaskFromResponse(data, point);
      if (mode === 'hover') {
        setHoverSelection(mask);
        setHoverSelectionState('ready');
      }
      return mask;
    } catch (maskError) {
      if (
        sourceGeneration !== selectionSourceGenerationRef.current
        || requestId !== hoverRequestIdRef.current
      ) return null;
      if (mode === 'hover') {
        setHoverSelection(null);
        setHoverSelectionState('error');
      } else {
        setError(maskError instanceof Error ? maskError.message : String(maskError));
      }
      return null;
    }
  };

  const handleSelectionImagePointerMove = (event: React.PointerEvent<HTMLImageElement>) => {
    if (!file || photoScope !== 'object-selection' || runState === 'running' || selectionState === 'applying') return;
    const point = selectionPointFromEvent(event);
    if (!point) return;
    lastHoverPointRef.current = point;
    if (hoverSelection && maskDistance(hoverSelection, point) < 0.02) return;
    if (hoverTimerRef.current) window.clearTimeout(hoverTimerRef.current);
    if (!hoverSelection) setHoverSelectionState('loading');
    hoverTimerRef.current = window.setTimeout(() => {
      const nextPoint = lastHoverPointRef.current;
      if (nextPoint) requestSelectionMask(nextPoint, 'hover');
    }, 450);
  };

  const handleSelectionImagePointerLeave = () => {
    if (hoverTimerRef.current) window.clearTimeout(hoverTimerRef.current);
    hoverRequestIdRef.current += 1;
    setHoverSelection(null);
    setHoverSelectionState('idle');
  };

  const handleSelectionImageClick = async (event: React.MouseEvent<HTMLImageElement>) => {
    if (!file || photoScope !== 'object-selection' || runState === 'running' || selectionState === 'applying') return;
    const point = selectionPointFromEvent(event);
    if (!point) return;
    if (hoverTimerRef.current) window.clearTimeout(hoverTimerRef.current);
    const mask = await requestSelectionMask(point, 'click');
    if (!mask) return;
    invalidateSelectionResult();
    setSelectedMasks((masks) => [...masks, { ...mask, id: `kept-${mask.id}-${masks.length}` }]);
    setSelectionPoints((points) => [...points, { id: `kept-${Date.now()}-${points.length}`, x: mask.x, y: mask.y }]);
    setHoverSelection(null);
    setHoverSelectionState('idle');
    setSelectionState('editing');
    setStatusText('Object kept');
  };

  const undoSelectionPoint = () => {
    invalidateSelectionResult();
    setSelectionPoints((points) => points.slice(0, -1));
    setSelectedMasks((masks) => masks.slice(0, -1));
    setSelectionState(selectedMasks.length > 1 ? 'editing' : 'idle');
    setStatusText('Selection updated');
  };

  const applyObjectSelection = async (): Promise<SelectionApplyResult | null> => {
    if (!file) return null;
    const sourceGeneration = selectionSourceGenerationRef.current;
    if (selectionAppliedFile && selectionPreviewUrl && selectionState === 'ready') {
      if (!selectionResult?.job_id || !selectionResult.mask || !selectionResult.selected_image) {
        setSelectionState('error');
        setStatusText('Selection expired');
        setError('The composed selection is incomplete. Apply the object selection again.');
        return null;
      }
      return {
        file: selectionAppliedFile,
        previewUrl: selectionPreviewUrl,
        selectionJobId: selectionResult.job_id,
      };
    }
    if (!selectedMasks.length) {
      setSelectionState('error');
      setStatusText('Selection required');
      setError('Hover until an object lights up, click it to keep it, then apply the selection.');
      return null;
    }

    setSelectionState('applying');
    setStatusText('Applying object selection');
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file, file.name || 'photo.jpg');
      formData.append(
        'mask_paths_json',
        JSON.stringify(selectedMasks.map((mask) => mask.maskPath)),
      );
      formData.append('selection_infill_mode', 'none');

      const response = await fetch(`${backendUrl}/selection/compose`, {
        method: 'POST',
        body: formData,
      });
      if (sourceGeneration !== selectionSourceGenerationRef.current) return null;
      if (!response.ok) {
        let message = `Object selection ${response.status}`;
        try {
          const details = await response.json();
          if (details?.detail) message = String(details.detail);
        } catch {
          // Keep the status-based message when the backend does not return JSON.
        }
        throw new Error(message);
      }

      const data = normalizeSelectionResult(await response.json());
      if (sourceGeneration !== selectionSourceGenerationRef.current) return null;
      if (!data.job_id || !data.mask || !data.selected_image) {
        throw new Error('Selection response did not include a complete, reusable compose job.');
      }
      const editedUrl = backendAssetUrl(data.selected_image_url);
      if (!editedUrl) throw new Error('Selection response did not include an edited image.');

      const editedResponse = await fetch(editedUrl);
      if (sourceGeneration !== selectionSourceGenerationRef.current) return null;
      if (!editedResponse.ok) throw new Error(`Could not load edited selection image ${editedResponse.status}`);
      const editedBlob = await editedResponse.blob();
      if (sourceGeneration !== selectionSourceGenerationRef.current) return null;
      const baseName = file.name ? file.name.replace(/\.[^.]+$/, '') || 'photo' : 'photo';
      const editedFile = new File([editedBlob], `selected-${baseName}.png`, { type: editedBlob.type || 'image/png' });

      setSelectionResult(data);
      setSelectionPreviewUrl(editedUrl);
      setSelectionAppliedFile(editedFile);
      setSelectionState('ready');
      setCompletedPreview(editedUrl);
      setStatusText('Selection ready');
      return { file: editedFile, previewUrl: editedUrl, selectionJobId: data.job_id };
    } catch (selectionError) {
      if (sourceGeneration !== selectionSourceGenerationRef.current) return null;
      setSelectionState('error');
      setStatusText('Selection failed');
      setError(selectionError instanceof Error ? selectionError.message : String(selectionError));
      return null;
    }
  };

  const openFilePicker = () => {
    if (!fileInputRef.current) return;
    fileInputRef.current.value = '';
    fileInputRef.current.click();
  };

  const replaceSourceFile = (nextFile: File | null) => {
    selectionSourceGenerationRef.current += 1;
    hoverRequestIdRef.current += 1;
    setFile(nextFile);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) replaceSourceFile(droppedFile);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (nextFile) replaceSourceFile(nextFile);
  };

  const recordImageDimensions = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth: width, naturalHeight: height } = event.currentTarget;
    if (width > 0 && height > 0) setMediaDimensions({ width, height });
  };

  const recordVideoDimensions = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    const { videoWidth: width, videoHeight: height } = event.currentTarget;
    if (width > 0 && height > 0) setMediaDimensions({ width, height });
  };

  const resetFile = () => {
    replaceSourceFile(null);
    setProcessedSTL('');
    setProcessedSceneGLB('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setCompletedPreview('');
    setSelectionPoints([]);
    setSelectionResult(null);
    setSelectionPreviewUrl('');
    setSelectionAppliedFile(null);
    setSelectionState('idle');
    setHoverSelection(null);
    setHoverSelectionState('idle');
    setSelectedMasks([]);
    setSelectionPrecompute(null);
    setSelectionPrecomputeState('idle');
    setSelectionPrecomputeError('');
    precomputeRequestIdRef.current += 1;
    setPlannerResponse(null);
    setRunState('idle');
    setStatusText('Ready');
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const downloadPlan = () => {
    const blob = new Blob([JSON.stringify(displayedPlan, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'stl-pipeline-plan.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const startDepthProPreload = async () => {
    setDepthProPreloadStarting(true);
    try {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 10000);
      const response = await fetch(`${backendUrl}/depth/preload/depthpro`, {
        method: 'POST',
        signal: controller.signal,
      });
      window.clearTimeout(timeout);
      if (!response.ok) throw new Error(`Depth Pro preload ${response.status}`);
      const data = (await response.json()) as DepthPreloadStatus;
      setDepthProPreload(data);
      setDepthProPreloadState('ready');
      setDepthProPreloadPollKey((value) => value + 1);
    } catch (preloadError) {
      setDepthProPreload({
        model_id: DEPTH_PRO_MODEL_ID,
        status: 'error',
        message: preloadError instanceof Error ? preloadError.message : String(preloadError),
        error: preloadError instanceof Error ? preloadError.message : String(preloadError),
      });
      setDepthProPreloadState('ready');
    } finally {
      setDepthProPreloadStarting(false);
    }
  };

  const runPipeline = async () => {
    if (!file) {
      setRunState('blocked');
      setStatusText('Import required');
      return;
    }
    const runSourceGeneration = selectionSourceGenerationRef.current;

    setError('');
    setProcessedSTL('');
    setProcessedSceneGLB('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setCompletedPreview('');
    setPlannerResponse(null);

    if (mediaKind === 'photo' && photoTarget === 'full-mesh' && stlPostprocessModel !== FULL_MESH_RUNNER_STL_POSTPROCESS) {
      setRunState('blocked');
      setStatusText('Repair adapter not attached');
      setError(
        `${modelLabel(modelCatalog, 'stl_postprocess', stlPostprocessModel)} is still planner-only for live Full Mesh STL runs. Select Trimesh repair to run the current image-to-mesh pipeline.`,
      );
      return;
    }

    if (
      mediaKind === 'photo' &&
      photoTarget === 'full-mesh' &&
      providerReadinessState === 'ready' &&
      selectedMeshReadiness &&
      !selectedMeshReadiness.runnable
    ) {
      setRunState('blocked');
      setStatusText('Provider setup missing');
      setError(
        `${modelLabel(modelCatalog, 'image_to_mesh', meshBackend)} is not runnable: ${providerSetupMessage(selectedMeshReadiness)}`,
      );
      return;
    }

    if (mediaKind === 'video') {
      const unsupported =
        videoScope !== 'selected-frames'
          ? 'Whole-video processing is not attached yet. Use Selection for the live video runner.'
          : !isLiveVideoFrameSelectionModel(frameSelectionModel)
            ? `${modelLabel(modelCatalog, 'frame_selection', frameSelectionModel)} is planner-only. Select Uniform frame sampler or Sharpness/motion selector.`
            : videoTarget === 'tracked-relief' && !selectionModel.startsWith('sam2.1-')
              ? 'Tracked relief requires SAM 2.1 Tiny or Base+ video segmentation.'
              : !isLiveVideoSegmentationModel(selectionModel)
              ? `${modelLabel(modelCatalog, 'selection', selectionModel)} is not attached to the live video runner.`
              : videoTarget === 'turntable-mesh' && cameraPoseModel !== LIVE_VIDEO_CAMERA_MODEL
                ? `${modelLabel(modelCatalog, 'camera_pose', cameraPoseModel)} is planner-only. Select Turntable orbit for a live run.`
                : videoTarget === 'turntable-mesh' && videoBackend !== LIVE_VIDEO_RECONSTRUCTION_MODEL
                  ? `${modelLabel(modelCatalog, 'video_reconstruction', videoBackend)} is planner-only. Select Multiview visual hull for a live run.`
                  : videoTarget === 'turntable-mesh' && stlPostprocessModel !== FULL_MESH_RUNNER_STL_POSTPROCESS
                    ? `${modelLabel(modelCatalog, 'stl_postprocess', stlPostprocessModel)} is planner-only. Select Trimesh repair for a live run.`
                    : '';
      if (unsupported) {
        setRunState('blocked');
        setStatusText('Video adapter not attached');
        setError(unsupported);
        return;
      }
    }

    let pipelineInputFile = file;
    let pipelinePreviewUrl = previewUrl;
    let selectionJobId = '';
    if (
      mediaKind === 'photo'
      && photoTarget === 'scene-diorama'
      && photoScope === 'object-selection'
      && selectedMasks.length === 0
    ) {
      setRunState('blocked');
      setStatusText('Scene selection required');
      setError('Select the people, building, and any props that should appear in the reconstructed scene.');
      return;
    }
    if (mediaKind === 'photo' && photoScope === 'object-selection' && photoTarget !== 'scene-diorama') {
      const selectedInput = await applyObjectSelection();
      if (runSourceGeneration !== selectionSourceGenerationRef.current) return;
      if (!selectedInput) {
        setRunState('blocked');
        return;
      }
      pipelineInputFile = selectedInput.file;
      pipelinePreviewUrl = selectedInput.previewUrl;
      selectionJobId = selectedInput.selectionJobId;
    }

    if (mediaKind === 'photo' && photoTarget === 'scene-diorama') {
      setRunState('running');
      setStatusText('Reconstructing scene diorama');
      try {
        const healthController = new AbortController();
        const healthTimeout = window.setTimeout(() => healthController.abort(), 2000);
        const health = await fetch(`${backendUrl}/health`, { signal: healthController.signal });
        window.clearTimeout(healthTimeout);
        if (!health.ok) throw new Error(`Backend health ${health.status}`);

        const sceneMasks = photoScope === 'object-selection' ? selectedMasks : [];
        const formData = new FormData();
        formData.append('file', file, file.name || 'scene.jpg');
        formData.append('mask_paths_json', JSON.stringify(sceneMasks.map((mask) => mask.maskPath)));
        formData.append(
          'selection_labels_json',
          JSON.stringify(sceneMasks.map((mask) => mask.selection_labels || [])),
        );
        formData.append('depth_provider', 'transformers');
        formData.append('depth_model', depthModel);
        formData.append('device', 'auto');
        formData.append('max_size_mm', String(printVolume.target_dimension_mm));
        formData.append('scene_depth_mm', String(effectiveSceneDepth));
        formData.append('base_thickness_mm', String(baseThickness));
        formData.append('facade_detail_mm', String(sceneFacadeDetail));
        formData.append('subject_depth_mm', String(sceneSubjectDepth));
        formData.append('depth_compression', String(sceneDepthCompression));
        formData.append('nozzle_diameter_mm', String(effectivePrinterNozzleDiameter));
        formData.append('minimum_feature_mm', String(minimumFeatureSize));
        formData.append('max_samples', String(sceneMaxSamples));

        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), PROCESS_IMAGE_TIMEOUT_MS);
        const response = await fetch(`${backendUrl}/process_scene_diorama`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        window.clearTimeout(timeout);
        if (!response.ok) {
          let message = `Scene reconstruction ${response.status}`;
          try {
            const errorPayload: unknown = await response.json();
            if (isRecord(errorPayload) && typeof errorPayload.detail === 'string') message = errorPayload.detail;
          } catch {
            // Keep the status-based message when the backend does not return JSON.
          }
          throw new Error(message);
        }

        const rawData: unknown = await response.json();
        if (!isRecord(rawData)) throw new Error('Scene reconstruction returned an invalid response.');
        const stlUrl = resolveServiceUrl(backendUrl, rawData.stl_url);
        const sceneUrl = resolveServiceUrl(backendUrl, rawData.scene_url);
        if (!stlUrl || !sceneUrl) throw new Error('Scene reconstruction did not return both STL and GLB artifacts.');
        const diagnostics = isRecord(rawData.stl_diagnostics) ? (rawData.stl_diagnostics as StlDiagnostics) : null;
        setProcessedSTL(stlUrl);
        setProcessedSceneGLB(sceneUrl);
        setDiagnosticsUrl(resolveServiceUrl(backendUrl, rawData.diagnostics_url));
        setStlDiagnostics(diagnostics);
        if (isRecord(rawData.runtime)) {
          setBackendRuntime(rawData.runtime as RuntimeInfo);
          setBackendRuntimeState('ready');
        }
        setStageTimings(isRecord(rawData.timings) ? (rawData.timings as StageTimings) : null);
        setDepthRunMetadata(isRecord(rawData.depth_metadata) ? (rawData.depth_metadata as DepthRunMetadata) : null);
        setCompletedPreview(resolveServiceUrl(backendUrl, rawData.preview_url) || previewUrl);
        const passesHardChecks = diagnostics?.stl_passes_hard_checks !== false;
        setRunState(passesHardChecks ? 'ready' : 'blocked');
        setStatusText(passesHardChecks ? 'Scene GLB and STL ready' : 'Scene emitted; STL checks failed');
        if (!passesHardChecks) {
          setError(`Scene STL failed ${failedChecksLabel(diagnostics?.stl_failed_checks)}.`);
        }
      } catch (runError) {
        setRunState('error');
        setStatusText('Scene reconstruction failed');
        setError(runError instanceof Error ? runError.message : String(runError));
      }
      return;
    }

    if (mediaKind === 'photo' && photoTarget === 'depth-relief') {
      setRunState('running');
      setStatusText('Generating relief STL');
      try {
        const healthController = new AbortController();
        const healthTimeout = window.setTimeout(() => healthController.abort(), 2000);
        const health = await fetch(`${backendUrl}/health`, { signal: healthController.signal });
        window.clearTimeout(healthTimeout);
        if (!health.ok) throw new Error(`Backend health ${health.status}`);

        const formData = new FormData();
        formData.append('file', pipelineInputFile, pipelineInputFile.name || file.name || 'photo.jpg');
        if (selectionJobId) {
          formData.append('selection_job_id', selectionJobId);
          formData.append('selection_mode', 'context');
          formData.append('selection_subject_lock', 'true');
          formData.append('selection_emission_only', 'true');
        }
        formData.append('depth_provider', 'transformers');
        formData.append('depth_model', depthModel);
        formData.append('device', 'auto');
        formData.append(
          'depth_downsample_sharpening',
          depthModel.toLowerCase().includes('depth-anything-v2') ? '0.35' : '0',
        );
        formData.append('depth_inference_precision', 'float32');
        formData.append('target_dimension', String(reliefTargetDimension));
        formData.append('z_scale', String(effectiveReliefHeight));
        formData.append('base_thickness_mm', String(baseThickness));
        formData.append('max_xy_size', String(printVolume.target_dimension_mm));
        formData.append('detail_basis_mm', String(printVolume.max_target_dimension_mm));
        formData.append('invert', String(reliefInvert));
        formData.append('sigma', String(detailSmoothing));
        formData.append('detail_boost', String(featureBoost));
        formData.append('printable_feature_depth_mm', String(printableFeatureDepth));
        formData.append('feature_bridge_depth_mm', String(featureBridgeDepth));
        formData.append('background_detail_boost', String(backgroundDetailBoost));
        formData.append('background_photo_detail_mm', String(backgroundPhotoDetail));
        formData.append('trim_top_background', String(trimTopBackground));
        formData.append('relief_gamma', String(reliefGamma));
        formData.append('base_border_px', String(baseBorderPx));
        formData.append('detail_radius', '2.0');
        formData.append('low_percentile', '1.0');
        formData.append('high_percentile', '99.0');
        formData.append('relief_polarity', reliefPolarity);
        formData.append('mesh_resolution_multiplier', String(meshResolutionMultiplier));
        formData.append('printer_profile', printVolume.label);
        formData.append('printer_max_x_mm', String(printVolume.max_x_mm));
        formData.append('printer_max_y_mm', String(printVolume.max_y_mm));
        formData.append('printer_max_z_mm', String(printVolume.max_z_mm));
        formData.append('printer_clearance_mm', String(printVolume.clearance_mm));
        formData.append('nozzle_diameter_mm', String(effectivePrinterNozzleDiameter));
        formData.append('minimum_feature_mm', String(minimumFeatureSize));
        formData.append('max_relief_slope', '2.0');
        formData.append('print_scale_percent', String(printVolume.print_scale_percent));
        formData.append('face_refinement_mode', 'auto');
        formData.append('face_detail_strength', '1.0');
        formData.append('face_feather_ratio', '0.20');
        formData.append('face_max_correction_ratio', '0.08');

        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), PROCESS_IMAGE_TIMEOUT_MS);
        const response = await fetch(`${backendUrl}/process_image`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Process image ${response.status}`);

        const rawData: unknown = await response.json();
        if (!isRecord(rawData)) throw new Error('Process image returned an invalid response.');
        const stlModel = optionalString(rawData.stl_model);
        const stlUrl =
          resolveServiceUrl(backendUrl, rawData.stl_url) ||
          (stlModel ? resolveServiceUrl(backendUrl, `/stl_model/${encodeURIComponent(stlModel)}`) : '');
        if (!stlUrl) throw new Error('Process image did not return an STL artifact.');
        const diagnostics = isRecord(rawData.stl_diagnostics)
          ? (rawData.stl_diagnostics as StlDiagnostics)
          : null;
        setProcessedSTL(stlUrl);
        setDiagnosticsUrl(resolveServiceUrl(backendUrl, rawData.diagnostics_url));
        setStlDiagnostics(diagnostics);
        if (isRecord(rawData.runtime)) {
          setBackendRuntime(rawData.runtime as RuntimeInfo);
          setBackendRuntimeState('ready');
        }
        setStageTimings(isRecord(rawData.timings) ? (rawData.timings as StageTimings) : null);
        setDepthRunMetadata(isRecord(rawData.depth_metadata) ? (rawData.depth_metadata as DepthRunMetadata) : null);
        setCompletedPreview(resolveServiceUrl(backendUrl, rawData.completed_image_url) || pipelinePreviewUrl);
        const passesHardChecks = diagnostics?.stl_passes_hard_checks !== false;
        const scaleWarning = Boolean(
          diagnostics?.relief_postprocess?.emitted_printability?.recognition_first_oversampling,
        );
        setRunState(passesHardChecks ? 'ready' : 'blocked');
        setStatusText(
          passesHardChecks
            ? scaleWarning
              ? 'STL ready; scale warning'
              : 'STL ready'
            : 'STL emitted; checks failed',
        );
        if (!passesHardChecks) {
          setError(`Relief STL failed ${failedChecksLabel(diagnostics?.stl_failed_checks)}.`);
        }
      } catch (runError) {
        setRunState('error');
        setStatusText('Run failed');
        setError(runError instanceof Error ? runError.message : String(runError));
      }
      return;
    }

    setRunState('running');
    setStatusText(
      mediaKind === 'video'
        ? videoTarget === 'tracked-relief'
          ? 'Tracking subject for relief STL'
          : 'Running turntable video-to-STL'
        : mediaKind === 'photo' && photoTarget === 'full-mesh'
          ? 'Running image-to-mesh'
          : 'Planning model stack',
    );
    try {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 5000);
      const response = await fetch(`${videoBackendUrl}/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobPlan),
        signal: controller.signal,
      });
      window.clearTimeout(timeout);
      if (!response.ok) throw new Error(`Planner ${response.status}`);
      const data = (await response.json()) as PlannerResponse;
      setPlannerResponse(data);

      if (mediaKind === 'video') {
        if (videoTarget === 'tracked-relief') {
          const reliefFormData = new FormData();
          reliefFormData.append('file', pipelineInputFile, pipelineInputFile.name || file.name || 'video.mp4');
          reliefFormData.append('frame_selection', frameSelectionModel);
          reliefFormData.append('segmentation_provider', selectionModel);
          reliefFormData.append('segmentation_device', 'auto');
          reliefFormData.append('selected_frame_count', String(Math.max(3, Math.min(64, Math.round(selectedFrameCount) || 12))));
          reliefFormData.append('object_point_x', '0.5');
          reliefFormData.append('object_point_y', '0.4');
          reliefFormData.append('strict_segmentation', 'true');
          reliefFormData.append('frame_max_side', '960');
          reliefFormData.append('depth_model', depthModel);
          reliefFormData.append('depth_device', 'auto');
          reliefFormData.append('target_dimension', '360');
          reliefFormData.append('max_xy_size_mm', String(printVolume.target_dimension_mm));
          reliefFormData.append('minimum_feature_mm', String(minimumFeatureSize));

          const reliefController = new AbortController();
          const reliefTimeout = window.setTimeout(() => reliefController.abort(), VIDEO_TO_MESH_TIMEOUT_MS);
          const reliefResponse = await fetch(`${videoBackendUrl}/run/video-to-relief`, {
            method: 'POST',
            body: reliefFormData,
            signal: reliefController.signal,
          });
          window.clearTimeout(reliefTimeout);
          if (!reliefResponse.ok) {
            let message = `Tracked video relief ${reliefResponse.status}`;
            try {
              const errorPayload: unknown = await reliefResponse.json();
              if (isRecord(errorPayload)) {
                const detail = errorPayload.detail;
                if (typeof detail === 'string') message = detail;
                else if (isRecord(detail)) message = optionalString(detail.message) || message;
              }
            } catch {
              // Keep the status-based message when the backend does not return JSON.
            }
            throw new Error(message);
          }
          const reliefData: unknown = await reliefResponse.json();
          if (!isRecord(reliefData)) throw new Error('Tracked video relief returned an invalid response.');
          const reliefStlUrl = resolveServiceUrl(videoBackendUrl, reliefData.stl_url);
          if (!reliefStlUrl) throw new Error('Tracked video relief did not return an STL artifact.');
          setProcessedSTL(reliefStlUrl);
          setDiagnosticsUrl(resolveServiceUrl(videoBackendUrl, reliefData.diagnostics_url));
          setStlDiagnostics(isRecord(reliefData.stl_diagnostics) ? (reliefData.stl_diagnostics as StlDiagnostics) : null);
          setStageTimings(isRecord(reliefData.timings) ? (reliefData.timings as StageTimings) : null);
          setCompletedPreview(
            resolveServiceUrl(videoBackendUrl, reliefData.preview_url)
              || resolveServiceUrl(videoBackendUrl, reliefData.selected_frame_url)
              || pipelinePreviewUrl,
          );
          const passesHardChecks = Boolean(reliefData.stl_passes_hard_checks);
          setStatusText(passesHardChecks ? 'Printable tracked relief ready' : 'Tracked relief emitted; checks failed');
          setRunState(passesHardChecks ? 'ready' : 'blocked');
          if (!passesHardChecks) {
            setError(`Tracked relief failed ${failedChecksLabel(reliefData.stl_failed_checks)}.`);
          }
          return;
        }

        const formData = new FormData();
        formData.append('file', pipelineInputFile, pipelineInputFile.name || file.name || 'video.mp4');
        formData.append('provider', videoBackend);
        formData.append('frame_selection', frameSelectionModel);
        formData.append('segmentation_provider', selectionModel);
        formData.append('segmentation_device', 'auto');
        formData.append('selected_frame_count', String(Math.max(3, Math.min(64, Math.round(selectedFrameCount) || 12))));
        formData.append('object_point_x', '0.5');
        formData.append('object_point_y', '0.5');
        formData.append('strict_segmentation', 'true');
        formData.append('rotation_degrees', '360');
        formData.append('rotation_direction', 'counter-clockwise');
        formData.append('provider_device', 'cpu');
        formData.append('mesh_repair', 'printable');
        formData.append('mesh_target_max_dimension', String(printVolume.target_dimension_mm));
        formData.append('mesh_min_bbox_dimension', '12');
        formData.append('mesh_target_faces', '40000');
        formData.append('frame_max_side', '960');
        formData.append('visual_hull_resolution', '32');
        formData.append('visual_hull_grid_extent', '1.9');
        formData.append('visual_hull_ortho_scale', '2.0');
        formData.append('visual_hull_mask_dilate', '1');

        const runnerController = new AbortController();
        const runnerTimeout = window.setTimeout(() => runnerController.abort(), VIDEO_TO_MESH_TIMEOUT_MS);
        const runnerResponse = await fetch(`${videoBackendUrl}/run/video-to-mesh`, {
          method: 'POST',
          body: formData,
          signal: runnerController.signal,
        });
        window.clearTimeout(runnerTimeout);
        if (!runnerResponse.ok) {
          let message = `Video-to-mesh runner ${runnerResponse.status}`;
          try {
            const errorPayload: unknown = await runnerResponse.json();
            if (isRecord(errorPayload)) {
              const detail = errorPayload.detail;
              if (typeof detail === 'string') message = detail;
              else if (isRecord(detail)) {
                const detailMessage = optionalString(detail.message);
                if (detailMessage) message = detailMessage;
              }
            }
          } catch {
            // Keep the status-based message when the backend does not return JSON.
          }
          throw new Error(message);
        }
        const runnerData: unknown = await runnerResponse.json();
        if (!isRecord(runnerData)) throw new Error('Video-to-mesh runner returned an invalid response.');
        const runnerStlUrl = resolveServiceUrl(videoBackendUrl, runnerData.stl_url);
        if (!runnerStlUrl) throw new Error('Video-to-mesh runner did not return an STL artifact.');
        const selectedFrameUrls = Array.isArray(runnerData.selected_frame_urls)
          ? runnerData.selected_frame_urls.map((value) => resolveServiceUrl(videoBackendUrl, value)).filter(Boolean)
          : [];
        setProcessedSTL(runnerStlUrl);
        setDiagnosticsUrl(resolveServiceUrl(videoBackendUrl, runnerData.diagnostics_url));
        setStlDiagnostics(isRecord(runnerData.stl_diagnostics) ? (runnerData.stl_diagnostics as StlDiagnostics) : null);
        setStageTimings(isRecord(runnerData.timings) ? (runnerData.timings as StageTimings) : null);
        setCompletedPreview(selectedFrameUrls[0] || pipelinePreviewUrl);
        const maskQualityPassed = !isRecord(runnerData.mask_quality) || runnerData.mask_quality.passes_hard_checks !== false;
        const passesHardChecks = Boolean(runnerData.stl_passes_hard_checks) && maskQualityPassed;
        setStatusText(passesHardChecks ? 'Printable video STL ready' : 'Video STL emitted; checks failed');
        setRunState(passesHardChecks ? 'ready' : 'blocked');
        if (!passesHardChecks) {
          setError(`Video STL failed ${failedChecksLabel(runnerData.stl_failed_checks)}.`);
        }
        return;
      }

      if (mediaKind === 'photo' && photoTarget === 'full-mesh') {
        const formData = new FormData();
        formData.append('file', pipelineInputFile, pipelineInputFile.name || file.name || 'photo.jpg');
        formData.append('provider', meshBackend);
        formData.append('provider_device', 'cuda');
        formData.append('mesh_repair', 'printable');
        formData.append('mesh_target_max_dimension', String(printVolume.target_dimension_mm));
        formData.append('mesh_min_bbox_dimension', '12');
        formData.append('mesh_max_bbox_aspect_ratio', '2.25');
        formData.append('mesh_target_faces', '40000');
        formData.append('num_inference_steps', '50');
        formData.append('guidance_scale', '7.0');
        formData.append('low_vram', 'true');
        formData.append('disable_progress', 'true');

        const runnerController = new AbortController();
        const runnerTimeout = window.setTimeout(() => runnerController.abort(), IMAGE_TO_MESH_TIMEOUT_MS);
        const runnerResponse = await fetch(`${videoBackendUrl}/run/image-to-mesh`, {
          method: 'POST',
          body: formData,
          signal: runnerController.signal,
        });
        window.clearTimeout(runnerTimeout);
        if (!runnerResponse.ok) {
          let message = `Image-to-mesh runner ${runnerResponse.status}`;
          try {
            const errorPayload: unknown = await runnerResponse.json();
            if (isRecord(errorPayload)) {
              const detail = errorPayload.detail;
              if (typeof detail === 'string') message = detail;
              else if (isRecord(detail)) {
                const detailMessage = optionalString(detail.message);
                if (detailMessage) message = detailMessage;
              }
            }
          } catch {
            // Keep the status-based message when the backend does not return JSON.
          }
          throw new Error(message);
        }
        const runnerData: unknown = await runnerResponse.json();
        if (!isRecord(runnerData)) throw new Error('Image-to-mesh runner returned an invalid response.');
        const runnerStlUrl = resolveServiceUrl(videoBackendUrl, runnerData.stl_url);
        if (!runnerStlUrl) throw new Error('Image-to-mesh runner did not return an STL artifact.');
        setProcessedSTL(runnerStlUrl);
        setDiagnosticsUrl(resolveServiceUrl(videoBackendUrl, runnerData.diagnostics_url));
        setStlDiagnostics(isRecord(runnerData.stl_diagnostics) ? (runnerData.stl_diagnostics as StlDiagnostics) : null);
        setStageTimings(isRecord(runnerData.timings) ? (runnerData.timings as StageTimings) : null);
        setCompletedPreview(pipelinePreviewUrl);
        const passesHardChecks = Boolean(runnerData.stl_passes_hard_checks);
        setStatusText(passesHardChecks ? 'Printable STL ready' : 'STL emitted; checks failed');
        setRunState(passesHardChecks ? 'ready' : 'blocked');
        if (!passesHardChecks) {
          setError(`STL was emitted but failed ${failedChecksLabel(runnerData.stl_failed_checks)}.`);
        }
        return;
      } else {
        setStatusText('Planner ready');
      }

      setRunState('ready');
    } catch (planError) {
      setPlannerResponse({
        status: 'local-plan',
        service: 'frontend-fallback',
        execution_mode: 'planner-offline',
        metrics: modelCatalog.metrics,
      });
      if ((mediaKind === 'photo' && photoTarget === 'full-mesh') || mediaKind === 'video') {
        setRunState('error');
        setStatusText('Run failed');
        setError(planError instanceof Error ? planError.message : String(planError));
      } else {
        setRunState('ready');
        setStatusText('Local plan ready');
        setError(planError instanceof Error ? `Companion planner unavailable: ${planError.message}` : 'Companion planner unavailable');
      }
    }
  };

  const canRun = Boolean(file) && runState !== 'running' && selectionState !== 'applying';
  const isPhoto = mediaKind === 'photo';
  const runtimeLabel =
    backendRuntimeState === 'checking'
      ? 'Checking'
      : backendRuntimeState === 'unavailable'
        ? 'Unavailable'
        : backendRuntime?.cuda_available
          ? 'CUDA'
          : 'CPU';
  const timingEntries = stageTimings
    ? ['preparation_seconds', 'depth_seconds', 'provider_seconds', 'stl_seconds', 'diagnostics_seconds', 'total_seconds']
        .filter((key) => typeof stageTimings[key] === 'number')
        .map((key) => [key, stageTimings[key]] as const)
    : [];
  const depthProStatus = depthProPreload?.status || (depthProPreloadState === 'checking' ? 'checking' : 'missing');
  const depthProProgress = Math.max(0, Math.min(100, depthProPreload?.progress_percent || 0));
  const depthProBytes =
    typeof depthProPreload?.downloaded_bytes === 'number' && typeof depthProPreload?.total_bytes === 'number'
      ? `${fileSizeLabel(depthProPreload.downloaded_bytes)} / ${fileSizeLabel(depthProPreload.total_bytes)}`
      : '';
  const depthProIsBusy = depthProPreloadStarting || depthProStatus === 'downloading';
  const selectionStatusLabel =
    selectionState === 'applying'
      ? 'Applying'
      : selectionState === 'ready'
        ? 'Selection ready'
        : selectionState === 'error'
          ? 'Selection error'
          : selectionPrecomputeState === 'loading'
            ? 'Preparing objects'
            : hoverSelectionState === 'loading'
            ? 'Finding object'
            : selectedMasks.length
              ? `${selectedMasks.length} kept`
              : selectionPrecompute?.precompute_id
                ? 'Hover to preview'
                : selectionPrecomputeState === 'unsupported'
                  ? 'Live hover'
                  : selectionPrecomputeState === 'error'
                    ? 'Live fallback'
                    : 'Hover to preview';
  const selectionCoverageLabel =
    typeof selectionResult?.mask_coverage === 'number'
      ? `${Math.round(selectionResult.mask_coverage * 1000) / 10}% kept`
      : `${selectedMasks.length} object${selectedMasks.length === 1 ? '' : 's'}`;

  return (
    <main className="workspace-canvas min-h-dvh overflow-x-hidden text-zinc-950">
      <WorkspaceShell>
        <WorkspaceColumn label="Input and run plan" name="input">
          <div className="workspace-panel p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-md border border-blue-200 bg-blue-50 text-blue-700">
                  <Cuboid className="h-5 w-5" strokeWidth={1.8} />
                </div>
                <div className="min-w-0">
                  <h1 className="truncate text-lg font-semibold">Photo/Video to STL</h1>
                  <p className="truncate text-sm text-zinc-500">STL-first reconstruction workspace</p>
                </div>
              </div>
              <Button variant="outline" size="icon" onClick={resetFile} title="Reset workspace">
                <RefreshCcw className="h-4 w-4" />
              </Button>
            </div>

            <div
              className="workspace-dropzone group relative flex h-[250px] items-center justify-center overflow-hidden border border-dashed border-zinc-300"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <>
                  {mediaKind === 'video' ? (
                    <video
                      src={previewUrl}
                      className="h-full w-full object-cover"
                      muted
                      playsInline
                      controls
                      onLoadedMetadata={recordVideoDimensions}
                    />
                  ) : (
                    <img
                      src={previewUrl}
                      alt=""
                      className="h-full w-full object-cover"
                      onLoad={recordImageDimensions}
                    />
                  )}
                  <button
                    type="button"
                    className="absolute left-2 top-2 grid h-8 w-8 place-items-center bg-white text-zinc-900 shadow-sm"
                    onClick={openFilePicker}
                    title="Change file"
                  >
                    <Upload className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    className="absolute right-2 top-2 grid h-8 w-8 place-items-center bg-zinc-950 text-white"
                    onClick={(event) => {
                      event.preventDefault();
                      resetFile();
                    }}
                    title="Remove file"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="flex h-full w-full flex-col items-center justify-center gap-3 text-center text-zinc-600"
                  onClick={openFilePicker}
                >
                  <Upload className="h-10 w-10 text-blue-700" />
                  <div className="text-sm font-medium">Import image or video</div>
                </button>
              )}
            </div>
            <input ref={fileInputRef} type="file" className="hidden" accept="image/*,video/*" onChange={handleFileInput} />

            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <button
                type="button"
                onClick={() => setMediaKind('photo')}
                className={classNames(
                  'workspace-segment flex h-10 items-center justify-center gap-2 border',
                  mediaKind === 'photo' ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-zinc-200 bg-white',
                )}
              >
                <ImageIcon className="h-4 w-4" />
                Photo
              </button>
              <button
                type="button"
                onClick={() => setMediaKind('video')}
                className={classNames(
                  'workspace-segment flex h-10 items-center justify-center gap-2 border',
                  mediaKind === 'video' ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-zinc-200 bg-white',
                )}
              >
                <Film className="h-4 w-4" />
                Video
              </button>
            </div>

            {file && (
              <div className="mt-3 grid grid-cols-2 gap-2 border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
                <div className="truncate">
                  <span className="block text-zinc-400">File</span>
                  {file.name}
                </div>
                <div>
                  <span className="block text-zinc-400">Size</span>
                  {fileSizeLabel(file.size)}
                </div>
              </div>
            )}
          </div>

          <details className="workspace-panel group">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-4 [&::-webkit-details-marker]:hidden">
              <span className="flex items-center gap-2">
                <Braces className="h-5 w-5 text-emerald-700" />
                <span className="font-semibold">Run Plan</span>
              </span>
              <span className="text-xs font-medium text-zinc-500 group-open:hidden">Show JSON</span>
              <span className="hidden text-xs font-medium text-zinc-500 group-open:inline">Hide JSON</span>
            </summary>
            <div className="border-t border-zinc-200 p-3">
              <pre className="max-h-[330px] overflow-auto bg-zinc-950 p-3 text-xs leading-5 text-zinc-100">
                {JSON.stringify(displayedPlan, null, 2)}
              </pre>
            </div>
          </details>
        </WorkspaceColumn>

        <WorkspaceColumn label="Geometry route" name="geometry">
          <div className="workspace-panel p-4">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Layers3 className="h-5 w-5 text-orange-700" />
                <h2 className="font-semibold">Geometry Route</h2>
              </div>
              <span
                className={classNames(
                  'border px-2 py-1 text-xs font-medium',
                  catalogState === 'ready' && 'border-emerald-700 bg-emerald-50 text-emerald-900',
                  catalogState === 'loading' && 'border-blue-700 bg-blue-50 text-blue-900',
                  catalogState === 'fallback' && 'border-orange-700 bg-orange-50 text-orange-900',
                )}
              >
                {catalogState === 'ready' ? 'Model service' : catalogState === 'loading' ? 'Loading models' : 'Fallback models'}
              </span>
            </div>

            {isPhoto ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setPhotoScope('whole-image')}
                    className={classNames(
                      'min-h-[52px] border px-3 text-sm',
                      photoScope === 'whole-image' ? 'border-emerald-700 bg-emerald-50 text-emerald-900' : 'border-zinc-200',
                    )}
                  >
                    Whole image
                  </button>
                  <button
                    type="button"
                    onClick={() => setPhotoScope('object-selection')}
                    className={classNames(
                      'min-h-[52px] border px-3 text-sm',
                      photoScope === 'object-selection' ? 'border-emerald-700 bg-emerald-50 text-emerald-900' : 'border-zinc-200',
                    )}
                  >
                    Object selection
                  </button>
                </div>

                {photoScope === 'object-selection' && (
                  <div className="space-y-3">
                    <ProductionModelField
                      controlId="photo-selection-model"
                      label="Selection model"
                      modelId={selectionModel}
                      modelLabel={modelLabel(modelCatalog, 'selection', selectionModel)}
                      detail="Selected for complete person and garment masks from one still-photo click."
                    />

                    <div className="border border-zinc-200 bg-zinc-50 p-3">
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-800">
                          <MousePointer2 className="h-4 w-4 text-emerald-700" />
                          Keep Objects
                        </div>
                        <span
                          className={classNames(
                            'border px-2 py-1 text-xs font-medium',
                            selectionState === 'ready' && 'border-emerald-700 bg-emerald-50 text-emerald-900',
                            selectionState === 'applying' && 'border-blue-700 bg-blue-50 text-blue-900',
                            selectionState === 'error' && 'border-red-700 bg-red-50 text-red-900',
                            (selectionState === 'idle' || selectionState === 'editing') && 'border-zinc-300 bg-white text-zinc-700',
                          )}
                        >
                          {selectionStatusLabel}
                        </span>
                      </div>

                      {previewUrl ? (
                        <div className="relative overflow-hidden border border-zinc-200 bg-white">
                          <img
                            ref={selectionImageRef}
                            src={previewUrl}
                            alt=""
                            className={classNames(
                              'block w-full cursor-crosshair select-none transition-opacity',
                              selectedMasks.length || hoverSelection ? 'opacity-55' : 'opacity-75',
                            )}
                            onPointerMove={handleSelectionImagePointerMove}
                            onPointerLeave={handleSelectionImagePointerLeave}
                            onClick={handleSelectionImageClick}
                            draggable={false}
                          />
                          {selectedMasks.map((mask) => (
                            <img
                              key={mask.id}
                              src={mask.tintUrl}
                              alt=""
                              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-100"
                            />
                          ))}
                          {hoverSelection?.tintUrl && (
                            <img
                              src={hoverSelection.tintUrl}
                              alt=""
                              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-50"
                            />
                          )}
                        </div>
                      ) : (
                        <div className="border border-dashed border-zinc-300 bg-white p-4 text-center text-sm text-zinc-500">
                          Import a photo to select objects.
                        </div>
                      )}

                      <div className="mt-2 grid grid-cols-3 gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          className="h-10"
                          onClick={undoSelectionPoint}
                          disabled={!selectedMasks.length || selectionState === 'applying'}
                        >
                          Undo
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          className="h-10"
                          onClick={clearObjectSelection}
                          disabled={(!selectedMasks.length && !selectionPreviewUrl) || selectionState === 'applying'}
                        >
                          Clear
                        </Button>
                        <Button
                          type="button"
                          className="h-10 gap-2"
                          onClick={applyObjectSelection}
                          disabled={!selectedMasks.length || selectionState === 'applying' || !file}
                        >
                          {selectionState === 'applying' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                          Apply
                        </Button>
                      </div>

                      {selectionPreviewUrl && (
                        <div className="mt-3 border border-emerald-200 bg-white p-2">
                          <div className="mb-2 flex items-center justify-between gap-2 text-xs">
                            <span className="font-semibold text-emerald-900">Selection preview</span>
                            <span className="text-zinc-500">{selectionCoverageLabel}</span>
                          </div>
                          <img src={selectionPreviewUrl} alt="" className="block w-full border border-zinc-200" />
                        </div>
                      )}

                      {selectionResult?.model_status && (
                        <div className="mt-2 text-xs text-zinc-500">
                          {selectionResult.model_status}
                          {selectionResult.model_error ? ` fallback: ${selectionResult.model_error}` : ''}
                        </div>
                      )}
                      {selectionPrecomputeState !== 'idle' && (
                        <div className="mt-2 text-xs text-zinc-500">
                          {selectionPrecomputeState === 'loading'
                            ? 'Preparing segmentation map'
                            : selectionPrecompute?.precompute_id
                              ? `Precomputed ${selectionPrecompute.segment_count || 0} segments`
                              : selectionPrecomputeState === 'unsupported'
                                ? selectionPrecompute?.message || 'Live hover masks'
                                : `Precompute unavailable: ${selectionPrecomputeError || 'using live hover masks'}`}
                        </div>
                      )}
                      {(hoverSelection?.selection_labels?.length || selectedMasks.length > 0) && (
                        <div className="mt-2 text-xs text-zinc-500">
                          {(hoverSelection?.selection_labels?.length
                            ? hoverSelection.selection_labels
                            : selectedMasks.flatMap((mask) => mask.selection_labels || [])
                          )
                            .filter((label, index, labels) => label && labels.indexOf(label) === index)
                            .join(', ') || 'object'}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
                  {photoTargets.map((target) => {
                    const Icon = target.icon;
                    return (
                      <button
                        key={target.value}
                        type="button"
                        onClick={() => {
                          setPhotoTarget(target.value);
                          if (target.value === 'scene-diorama') setPhotoScope('object-selection');
                        }}
                        className={classNames(
                          'flex min-h-[72px] items-center gap-3 border px-3 text-left',
                          photoTarget === target.value ? 'border-blue-700 bg-blue-50 text-blue-900' : 'border-zinc-200',
                        )}
                      >
                        <Icon className="h-5 w-5 shrink-0" />
                        <span className="text-sm font-medium">{target.label}</span>
                      </button>
                    );
                  })}
                </div>

                {photoTarget === 'depth-relief' ? (
                  <div className="space-y-4">
                    <ProductionModelField
                      controlId="relief-depth-model"
                      label="Depth model"
                      modelId={depthModel}
                      modelLabel={selectedDepthModel.label}
                      detail={selectedDepthModel.notes}
                    />

                    {selectedDepthModel.id === DEPTH_PRO_MODEL_ID && (
                      <div className="border border-zinc-200 bg-zinc-50 p-3 text-xs">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span className="font-semibold text-zinc-800">Depth Pro Cache</span>
                          <span className={classNames('border px-2 py-1 font-medium', depthPreloadTone(depthProStatus))}>
                            {depthPreloadLabel(depthProStatus)}
                          </span>
                        </div>
                        <div className="h-2 overflow-hidden bg-zinc-200">
                          <div className="h-full bg-blue-700" style={{ width: `${depthProProgress}%` }} />
                        </div>
                        <div className="mt-2 flex items-center justify-between gap-2 text-zinc-600">
                          <span>{depthProBytes || depthProPreload?.message || 'Checking cache'}</span>
                          <span>{depthProProgress.toFixed(1)}%</span>
                        </div>
                        {depthProPreload?.error && <div className="mt-2 text-red-700">{depthProPreload.error}</div>}
                        <Button
                          type="button"
                          variant="outline"
                          className="mt-3 h-10 w-full gap-2"
                          onClick={startDepthProPreload}
                          disabled={depthProIsBusy || depthProStatus === 'ready'}
                        >
                          {depthProIsBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                          {depthProStatus === 'ready' ? 'Cached' : depthProIsBusy ? 'Preloading' : 'Preload Depth Pro'}
                        </Button>
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => setReliefPolarity('raised-print')}
                        className={classNames(
                          'min-h-[52px] border px-3 text-sm',
                          reliefPolarity === 'raised-print' ? 'border-emerald-700 bg-emerald-50 text-emerald-900' : 'border-zinc-200',
                        )}
                      >
                        Raised print
                      </button>
                      <button
                        type="button"
                        onClick={() => setReliefPolarity('mold')}
                        className={classNames(
                          'min-h-[52px] border px-3 text-sm',
                          reliefPolarity === 'mold' ? 'border-orange-700 bg-orange-50 text-orange-900' : 'border-zinc-200',
                        )}
                      >
                        Mold
                      </button>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2">
                      <label className="text-sm font-medium text-zinc-700">
                        Relief height
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="2"
                          max={reliefSliderMax}
                          value={Math.min(depthScale, reliefSliderMax)}
                          onChange={(event) => setDepthScale(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">
                          {depthScale} mm{effectiveReliefHeight < depthScale ? `, using ${effectiveReliefHeight.toFixed(1)} mm` : ''}
                        </span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Base thickness
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="1"
                          max="8"
                          step="0.2"
                          value={baseThickness}
                          onChange={(event) => setBaseThickness(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{baseThickness.toFixed(1)} mm</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Detail smoothing
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={detailSmoothing}
                          onChange={(event) => setDetailSmoothing(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{detailSmoothing.toFixed(1)} sigma</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Feature boost
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="3"
                          step="0.1"
                          value={featureBoost}
                          onChange={(event) => setFeatureBoost(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{featureBoost.toFixed(1)}x local detail</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Feature depth
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={printableFeatureDepth}
                          onChange={(event) => setPrintableFeatureDepth(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{printableFeatureDepth.toFixed(2)} mm signed detail</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Feature attachment
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={featureBridgeDepth}
                          onChange={(event) => setFeatureBridgeDepth(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{featureBridgeDepth.toFixed(1)} mm bridge cap</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Background detail
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="1"
                          max="3"
                          step="0.1"
                          value={backgroundDetailBoost}
                          onChange={(event) => setBackgroundDetailBoost(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{backgroundDetailBoost.toFixed(1)}x outside faces</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Photo texture
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="0.6"
                          step="0.02"
                          value={backgroundPhotoDetail}
                          onChange={(event) => setBackgroundPhotoDetail(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{backgroundPhotoDetail.toFixed(2)} mm image relief</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Relief curve
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0.45"
                          max="1.4"
                          step="0.05"
                          value={reliefGamma}
                          onChange={(event) => setReliefGamma(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{reliefGamma.toFixed(2)} gamma</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Mesh detail
                        <select
                          value={meshResolutionMultiplier}
                          onChange={(event) => setMeshResolutionMultiplier(Number(event.target.value))}
                          className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                        >
                          {resolutionMultipliers.map((multiplier) => (
                            <option key={multiplier} value={multiplier}>
                              {multiplier}x ({Math.min(
                                reliefDetailSamples.max,
                                Math.max(reliefDetailSamples.min, Math.round(printVolume.max_target_dimension_mm * multiplier)),
                              )} samples)
                            </option>
                          ))}
                        </select>
                        <span className="text-xs text-zinc-500">
                          {reliefTargetDimension} depth / {reliefPrintableDimension} STL samples, {reliefSamplePitch.toFixed(2)} mm/sample
                        </span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Crisp border
                        <input
                          className="mt-2 h-10 w-full border border-zinc-300 px-3"
                          type="number"
                          min="0"
                          max="12"
                          value={baseBorderPx}
                          onChange={(event) => setBaseBorderPx(Number(event.target.value))}
                        />
                      </label>
                      <label className="flex items-center gap-3 text-sm font-medium text-zinc-700">
                        <input
                          className="h-4 w-4 accent-blue-700"
                          type="checkbox"
                          checked={trimTopBackground}
                          onChange={(event) => setTrimTopBackground(event.target.checked)}
                        />
                        Trim empty sky
                      </label>
                    </div>
                  </div>
                ) : photoTarget === 'scene-diorama' ? (
                  <div className="space-y-4">
                    <ProductionModelField
                      controlId="scene-depth-model"
                      label="Scene depth model"
                      modelId={depthModel}
                      modelLabel={selectedDepthModel.label}
                      detail="The same verified depth field keeps subject and background geometry on one measured scale."
                    />

                    <div className="grid gap-4 md:grid-cols-2">
                      <label className="text-sm font-medium text-zinc-700">
                        Scene depth
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="24"
                          max={sceneDepthSliderMax}
                          step="2"
                          value={effectiveSceneDepth}
                          onChange={(event) => setSceneDepth(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{effectiveSceneDepth.toFixed(0)} mm</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Subject volume
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="4"
                          max="30"
                          step="1"
                          value={sceneSubjectDepth}
                          onChange={(event) => setSceneSubjectDepth(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{sceneSubjectDepth.toFixed(0)} mm</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Facade detail
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={sceneFacadeDetail}
                          onChange={(event) => setSceneFacadeDetail(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{sceneFacadeDetail.toFixed(1)} mm</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Depth separation
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={sceneDepthCompression}
                          onChange={(event) => setSceneDepthCompression(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{sceneDepthCompression.toFixed(2)}</span>
                      </label>
                      <label className="text-sm font-medium text-zinc-700">
                        Base thickness
                        <input
                          className="mt-2 w-full accent-blue-700"
                          type="range"
                          min="1"
                          max="8"
                          step="0.2"
                          value={baseThickness}
                          onChange={(event) => setBaseThickness(Number(event.target.value))}
                        />
                        <span className="text-xs text-zinc-500">{baseThickness.toFixed(1)} mm</span>
                      </label>
                      <div className="border border-zinc-200 bg-zinc-50 p-3 text-sm">
                        <div className="text-xs font-medium text-zinc-500">Scene sampling</div>
                        <div className="mt-1 font-semibold text-zinc-900">{sceneMaxSamples} samples</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="min-w-0">
                      <ProductionModelField
                        controlId="full-mesh-model"
                        label="Mesh model"
                        modelId={meshBackend}
                        modelLabel={modelLabel(modelCatalog, 'image_to_mesh', meshBackend)}
                        detail="Promoted after a paired 10-object STL-quality run with all printability gates passing."
                      />
                      <span
                        className={classNames(
                          'mt-2 inline-flex border px-2 py-1 text-xs font-medium',
                          providerReadinessState === 'loading' && 'border-blue-700 bg-blue-50 text-blue-900',
                          providerReadinessState === 'unavailable' && 'border-zinc-300 bg-zinc-50 text-zinc-700',
                          providerReadinessState === 'ready' && providerReadinessTone(selectedMeshReadiness),
                        )}
                      >
                        {providerReadinessState === 'loading'
                          ? 'Checking'
                          : providerReadinessState === 'unavailable'
                            ? 'Unknown'
                            : providerReadinessLabel(selectedMeshReadiness)}
                      </span>
                      <span className="mt-1 block text-xs text-zinc-500">{providerSetupMessage(selectedMeshReadiness)}</span>
                    </div>
                    <ProductionModelField
                      controlId="photo-stl-repair"
                      label="STL repair"
                      modelId={stlPostprocessModel}
                      modelLabel={modelLabel(modelCatalog, 'stl_postprocess', stlPostprocessModel)}
                      detail="Fixed to the repair path used by the watertightness and manifoldness gates."
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2" aria-label="Video output mode">
                  <button
                    type="button"
                    onClick={() => {
                      setVideoTarget('turntable-mesh');
                      setSelectionModel('turntable-grabcut');
                    }}
                    className={classNames(
                      'min-h-[64px] border px-3 text-sm',
                      videoTarget === 'turntable-mesh' ? 'border-blue-700 bg-blue-50 text-blue-900' : 'border-zinc-200',
                    )}
                  >
                    Turntable mesh
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setVideoTarget('tracked-relief');
                      setVideoScope('selected-frames');
                      setFrameSelectionModel('sharpness-motion-selector');
                      setSelectionModel('sam2.1-hiera-tiny-video');
                    }}
                    className={classNames(
                      'min-h-[64px] border px-3 text-sm',
                      videoTarget === 'tracked-relief' ? 'border-blue-700 bg-blue-50 text-blue-900' : 'border-zinc-200',
                    )}
                  >
                    Tracked relief
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setVideoScope('selected-frames')}
                    className={classNames(
                      'min-h-[64px] border px-3 text-sm',
                      videoScope === 'selected-frames' ? 'border-blue-700 bg-blue-50 text-blue-900' : 'border-zinc-200',
                    )}
                  >
                    Selection
                  </button>
                  <button
                    type="button"
                    onClick={() => setVideoScope('everything')}
                    className={classNames(
                      'min-h-[64px] border px-3 text-sm',
                      videoScope === 'everything' ? 'border-blue-700 bg-blue-50 text-blue-900' : 'border-zinc-200',
                    )}
                  >
                    Everything
                  </button>
                </div>

                <div className="grid gap-3">
                  <label className="text-sm font-medium text-zinc-700">
                    {videoScope === 'selected-frames' ? 'Selected frames' : 'Frame step'}
                    <input
                      className="mt-2 h-10 w-full border border-zinc-300 px-3"
                      type="number"
                      min={videoScope === 'selected-frames' ? 3 : 1}
                      max={videoScope === 'selected-frames' ? 64 : 240}
                      value={videoScope === 'selected-frames' ? selectedFrameCount : frameStep}
                      onChange={(event) =>
                        videoScope === 'selected-frames'
                          ? setSelectedFrameCount(Number(event.target.value))
                          : setFrameStep(Number(event.target.value))
                      }
                    />
                  </label>
                  <ProductionModelField
                    controlId="video-frame-selector"
                    label="Frame selector"
                    modelId={frameSelectionModel}
                    modelLabel={modelLabel(modelCatalog, 'frame_selection', frameSelectionModel)}
                    detail={videoTarget === 'tracked-relief' ? 'Sharpness and motion coverage for the best relief frame.' : 'Uniform angular coverage for a controlled turntable clip.'}
                  />
                  <ProductionModelField
                    controlId="video-selection-model"
                    label="Selection model"
                    modelId={selectionModel}
                    modelLabel={modelLabel(modelCatalog, 'selection', selectionModel)}
                    detail={videoTarget === 'tracked-relief' ? 'Temporal SAM propagation for the selected subject.' : 'Deterministic foreground masks for fixed-camera turntable video.'}
                  />
                  <ProductionModelField
                    controlId="video-camera-model"
                    label="Camera/pose"
                    modelId={cameraPoseModel}
                    modelLabel={modelLabel(modelCatalog, 'camera_pose', cameraPoseModel)}
                    detail="Deterministic orbit assignment for the supported fixed-camera capture."
                  />
                  <ProductionModelField
                    controlId="video-reconstruction-model"
                    label="Reconstruction"
                    modelId={videoBackend}
                    modelLabel={modelLabel(modelCatalog, 'video_reconstruction', videoBackend)}
                    detail="The measured multiview lane that emitted printable STLs across its held-out slice."
                  />
                  <ProductionModelField
                    controlId="video-stl-repair"
                    label="STL repair"
                    modelId={stlPostprocessModel}
                    modelLabel={modelLabel(modelCatalog, 'stl_postprocess', stlPostprocessModel)}
                    detail="Shared printability repair and diagnostics path."
                  />
                </div>

                <div className="grid gap-2 md:grid-cols-2">
                  <div className="border border-zinc-200 bg-zinc-50 p-3 text-sm">
                    <div className="mb-1 font-medium">Camera/keypoint stream</div>
                    <div className="text-zinc-600">
                      {videoTarget === 'tracked-relief'
                        ? 'Perspective motion measured from tracked masks'
                        : videoScope === 'selected-frames'
                          ? 'Uncropped frames retained'
                          : 'All sampled frames retained'}
                    </div>
                  </div>
                  <div className="border border-zinc-200 bg-zinc-50 p-3 text-sm">
                    <div className="mb-1 font-medium">Training target stream</div>
                    <div className="text-zinc-600">
                      {videoTarget === 'tracked-relief'
                        ? 'Best visible frame converted to printable relief'
                        : videoScope === 'selected-frames'
                          ? 'Object masks only'
                          : 'Full scene/object mesh'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-4 border-t border-zinc-200 pt-4">
              <div className="mb-2 text-sm font-semibold">Active Model Stack</div>
              <div className="grid gap-2">
                {activeModelStack.map(({ label, model }) => (
                  <article key={`${label}-${model.id}`} className="min-w-0 border border-zinc-200 bg-zinc-50 p-3 text-sm">
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-xs font-medium uppercase text-zinc-500">{label}</div>
                        <div className="break-words font-semibold text-zinc-900">{model.label}</div>
                      </div>
                      <span className={classNames('shrink-0 border px-2 py-1 text-[11px] font-medium', availabilityTone(model.availability))}>
                        {availabilityLabel(model.availability)}
                      </span>
                    </div>
                    <div className="break-words text-xs text-zinc-600">{model.role || model.model || model.id}</div>
                  </article>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-3">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const done = runState === 'ready' || (runState === 'running' && index < 1);
              const active = runState === 'running' && index === 1;
              return (
                <article key={`${step.label}-${index}`} className="workspace-step min-h-[128px] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <Icon className="h-5 w-5 text-zinc-800" />
                    <StepIcon active={active} done={done} />
                  </div>
                  <div className="text-sm font-semibold">{step.label}</div>
                  <div className="mt-1 text-sm text-zinc-600">{step.detail}</div>
                </article>
              );
            })}
          </div>
        </WorkspaceColumn>

        <WorkspaceColumn label="STL output and printer settings" name="output">
          <div className="workspace-panel p-4">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Box className="h-5 w-5 text-blue-700" />
                <h2 className="font-semibold">STL Output</h2>
              </div>
              <span
                className={classNames(
                  'border px-2 py-1 text-xs font-medium',
                  runState === 'ready' && 'border-emerald-700 bg-emerald-50 text-emerald-900',
                  runState === 'running' && 'border-blue-700 bg-blue-50 text-blue-900',
                  runState === 'blocked' && 'border-orange-700 bg-orange-50 text-orange-900',
                  runState === 'error' && 'border-red-700 bg-red-50 text-red-900',
                  runState === 'idle' && 'border-zinc-200 bg-zinc-50 text-zinc-600',
                )}
              >
                {statusText}
              </span>
            </div>

            <div className="relative flex aspect-square items-center justify-center overflow-hidden border border-zinc-200 bg-zinc-100">
              {processedSTL ? (
                <div className="absolute inset-0" data-testid="stl-orbit-viewer">
                  <StlViewer
                    key={processedSTL}
                    style={{ width: '100%', height: '100%' }}
                    url={processedSTL}
                    orbitControls
                    shadows
                    cameraProps={{ initialPosition: { latitude: Math.PI / 8, longitude: 0, distance: 1 } }}
                    modelProps={{ color: photoTarget === 'scene-diorama' ? '#d8574b' : '#737b85' }}
                    floorProps={{ gridWidth: 8, gridLength: 8 }}
                    canvasId="stl-output-canvas"
                  />
                </div>
              ) : (
                <>
                  {completedPreview || previewUrl ? (
                    <img src={completedPreview || previewUrl} alt="" className="absolute inset-0 h-full w-full object-cover opacity-35" />
                  ) : null}
                  <div className="relative grid h-44 w-44 place-items-center border border-zinc-300 bg-white/80">
                    {runState === 'running' ? (
                      <Loader2 className="h-14 w-14 animate-spin text-blue-700" />
                    ) : (
                      <div className="relative h-24 w-24">
                        <div className="absolute left-3 top-4 h-16 w-20 skew-x-[-12deg] border border-zinc-900 bg-emerald-100" />
                        <div className="absolute left-7 top-1 h-16 w-16 rotate-45 border border-zinc-900 bg-blue-100" />
                        <div className="absolute bottom-0 left-0 h-5 w-24 border border-zinc-900 bg-orange-100" />
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>

            {error && <div className="mt-3 border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div>}

            {depthRunMetadata?.fallback_reason && (
              <div className="mt-3 border border-orange-200 bg-orange-50 p-3 text-xs text-orange-900">
                <div className="font-semibold">Depth fallback used</div>
                <div className="mt-1">{depthRunMetadata.fallback_reason}</div>
                {depthRunMetadata.effective_model && (
                  <div className="mt-1 text-orange-800">Effective model: {depthRunMetadata.effective_model}</div>
                )}
              </div>
            )}

            <div className="mt-3 border border-zinc-200 bg-zinc-50 p-3 text-xs">
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="font-semibold text-zinc-800">Backend Runtime</span>
                <span
                  className={classNames(
                    'border px-2 py-1 font-medium',
                    backendRuntime?.cuda_available && backendRuntimeState === 'ready' && 'border-emerald-700 bg-emerald-50 text-emerald-900',
                    !backendRuntime?.cuda_available && backendRuntimeState === 'ready' && 'border-orange-700 bg-orange-50 text-orange-900',
                    backendRuntimeState === 'checking' && 'border-blue-700 bg-blue-50 text-blue-900',
                    backendRuntimeState === 'unavailable' && 'border-red-700 bg-red-50 text-red-900',
                  )}
                >
                  {runtimeLabel}
                </span>
              </div>
              <div className="break-words font-medium text-zinc-900">{backendRuntime?.device || 'Waiting for backend health'}</div>
              <div className="mt-1 text-zinc-500">
                Torch {backendRuntime?.torch || '-'} / CUDA {backendRuntime?.cuda_version || '-'}
              </div>
            </div>

            {mediaKind === 'photo' && photoTarget === 'depth-relief' && (
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Depth</div>
                  <div className="mt-1 font-semibold">{depthModels.find((model) => model.id === depthModel)?.label.replace('Depth Anything ', '')}</div>
                </div>
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Polarity</div>
                  <div className="mt-1 font-semibold">{reliefPolarity === 'raised-print' ? 'Raised' : 'Mold'}</div>
                </div>
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Smooth</div>
                  <div className="mt-1 font-semibold">{detailSmoothing.toFixed(1)}</div>
                </div>
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Samples</div>
                  <div className="mt-1 font-semibold">{reliefTargetDimension}</div>
                </div>
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Boost</div>
                  <div className="mt-1 font-semibold">{featureBoost.toFixed(1)}</div>
                </div>
                <div className="border border-zinc-200 bg-zinc-50 p-2">
                  <div className="font-medium text-zinc-500">Border</div>
                  <div className="mt-1 font-semibold">{baseBorderPx}px</div>
                </div>
              </div>
            )}

            {timingEntries.length > 0 && (
              <div className="mt-3 border border-zinc-200 bg-white p-3">
                <div className="mb-2 text-sm font-semibold">Stage Timings</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {timingEntries.map(([key, seconds]) => (
                    <div key={key} className="flex items-center justify-between border border-zinc-200 bg-zinc-50 px-2 py-1.5">
                      <span className="text-zinc-600">{timingLabel(key)}</span>
                      <span className="font-semibold">{secondsLabel(seconds)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Button onClick={runPipeline} disabled={!canRun} className="h-11 gap-2">
                {runState === 'running' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Run
              </Button>
              <Button variant="outline" onClick={downloadPlan} className="h-11 gap-2">
                <Download className="h-4 w-4" />
                Plan
              </Button>
            </div>

            {processedSTL && (
              <a
                href={processedSTL}
                download="model.stl"
                className="mt-2 flex h-11 items-center justify-center gap-2 border border-emerald-700 bg-emerald-50 text-sm font-medium text-emerald-900"
              >
                <FileDown className="h-4 w-4" />
                Download STL
              </a>
            )}

            {processedSceneGLB && (
              <a
                href={processedSceneGLB}
                download="scene.glb"
                className="mt-2 flex h-11 items-center justify-center gap-2 border border-blue-700 bg-blue-50 text-sm font-medium text-blue-900"
              >
                <Box className="h-4 w-4" />
                Download Scene GLB
              </a>
            )}

            {stlDiagnostics && (
              <div className="mt-3 border border-zinc-200 bg-zinc-50 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="text-sm font-semibold">Diagnostics</div>
                  {diagnosticsUrl && (
                    <a href={diagnosticsUrl} className="text-xs font-medium text-blue-700">
                      JSON
                    </a>
                  )}
                </div>
                {stlDiagnostics.relief_postprocess?.emitted_printability?.recognition_first_oversampling && (
                  <div className="mb-2 flex gap-2 border border-amber-300 bg-amber-50 px-2.5 py-2 text-xs text-amber-950">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>
                      Full detail is retained at this size. Some slopes or features are finer than the configured nozzle; verify the result in your slicer.
                    </span>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {([
                    ['Watertight', stlDiagnostics.stl_is_watertight],
                    ['Volume', stlDiagnostics.stl_is_volume],
                    ['Manifold', stlDiagnostics.stl_is_manifold],
                    ['Winding', stlDiagnostics.stl_winding_consistent],
                    ['Positive', stlDiagnostics.stl_positive_volume],
                    ['Single body', stlDiagnostics.stl_single_component],
                  ] as Array<[string, boolean | undefined]>).map(([label, value]) => (
                    <div key={String(label)} className="flex items-center justify-between border border-zinc-200 bg-white px-2 py-1.5">
                      <span className="text-zinc-600">{label}</span>
                      <span className={classNames('font-semibold', value ? 'text-emerald-700' : 'text-red-700')}>
                        {value ? 'Pass' : 'Fail'}
                      </span>
                    </div>
                  ))}
                  <div className="flex items-center justify-between border border-zinc-200 bg-white px-2 py-1.5">
                    <span className="text-zinc-600">Faces</span>
                    <span className="font-semibold">{stlDiagnostics.stl_faces?.toLocaleString() || '-'}</span>
                  </div>
                  <div className="flex items-center justify-between border border-zinc-200 bg-white px-2 py-1.5">
                    <span className="text-zinc-600">Aspect</span>
                    <span className="font-semibold">
                      {typeof stlDiagnostics.stl_bbox_aspect_ratio === 'number' ? stlDiagnostics.stl_bbox_aspect_ratio.toFixed(2) : '-'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="workspace-panel p-4">
            <div className="mb-3 flex items-center gap-2">
              <Printer className="h-5 w-5 text-orange-700" />
              <h2 className="font-semibold">Printer Volume</h2>
            </div>

            <label className="block text-sm font-medium text-zinc-700">
              Preset
              <select
                value={printerPreset}
                onChange={(event) => applyPrinterPreset(event.target.value as PrinterPresetId)}
                className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
              >
                {printerPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                { label: 'X', value: printerMaxX, setValue: setPrinterMaxX },
                { label: 'Y', value: printerMaxY, setValue: setPrinterMaxY },
                { label: 'Z', value: printerMaxZ, setValue: setPrinterMaxZ },
              ].map((axis) => (
                <label key={axis.label} className="text-sm font-medium text-zinc-700">
                  {axis.label}
                  <input
                    className="mt-2 h-10 w-full border border-zinc-300 px-2"
                    type="number"
                    min="1"
                    max="1000"
                    value={axis.value}
                    onChange={(event) => {
                      setPrinterPreset('custom');
                      axis.setValue(Number(event.target.value));
                    }}
                  />
                </label>
              ))}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <label className="text-sm font-medium text-zinc-700">
                Clearance
                <input
                  className="mt-2 h-10 w-full border border-zinc-300 px-3"
                  type="number"
                  min="0"
                  max="40"
                  step="0.5"
                  value={printerClearance}
                  onChange={(event) => setPrinterClearance(Number(event.target.value))}
                />
              </label>
              <label className="text-sm font-medium text-zinc-700">
                Nozzle
                <input
                  className="mt-2 h-10 w-full border border-zinc-300 px-3"
                  type="number"
                  min="0.2"
                  max="2"
                  step="0.05"
                  value={printerNozzleInput}
                  onChange={(event) => {
                    setPrinterPreset('custom');
                    setPrinterNozzleInput(event.target.value);
                  }}
                  onBlur={commitPrinterNozzleDiameter}
                />
              </label>
            </div>

            <label className="mt-3 block text-sm font-medium text-zinc-700">
              Print size
              <input
                className="mt-2 w-full accent-blue-700"
                type="range"
                min="10"
                max="100"
                step="5"
                value={printScalePercent}
                onChange={(event) => setPrintScalePercent(Number(event.target.value))}
              />
              <span className="text-xs text-zinc-500">
                {printScalePercent}% / {printFootprint
                  ? `${printFootprint.width.toFixed(1)} x ${printFootprint.height.toFixed(1)} mm`
                  : `${printVolume.target_dimension_mm} mm max`}
              </span>
            </label>

            <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">STL X</div>
                <div className="font-semibold" data-testid="print-size-x">
                  {printFootprint ? `${printFootprint.width.toFixed(1)} mm` : '--'}
                </div>
              </div>
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">STL Y</div>
                <div className="font-semibold" data-testid="print-size-y">
                  {printFootprint ? `${printFootprint.height.toFixed(1)} mm` : '--'}
                </div>
              </div>
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">Max relief Z</div>
                <div className="font-semibold">{printVolume.max_relief_height_mm.toFixed(1)} mm</div>
              </div>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">Nozzle</div>
                <div className="font-semibold">{effectivePrinterNozzleDiameter.toFixed(2)} mm</div>
              </div>
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">Min feature</div>
                <div className="font-semibold">{minimumFeatureSize.toFixed(2)} mm</div>
              </div>
            </div>

            <div className="mt-2 text-xs text-zinc-500">
              Build volume {printVolume.max_x_mm} x {printVolume.max_y_mm} x {printVolume.max_z_mm} mm
            </div>
          </div>

          <div className="workspace-panel p-4">
            <div className="mb-3 flex items-center gap-2">
              <BadgeCheck className="h-5 w-5 text-emerald-700" />
              <h2 className="font-semibold">Promotion Gates</h2>
            </div>
            <div className="space-y-2 text-sm">
              {['Watertight', 'Manifold', 'Positive volume', 'Single component', 'Printable scale', 'Fits printer volume'].map((gate) => (
                <div key={gate} className="flex items-center justify-between border border-zinc-200 px-3 py-2">
                  <span>{gate}</span>
                  <Check className="h-4 w-4 text-emerald-700" />
                </div>
              ))}
            </div>
          </div>
        </WorkspaceColumn>
      </WorkspaceShell>
    </main>
  );
}
