'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  BadgeCheck,
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

const DEFAULT_BACKEND_URL = 'http://localhost:8004';
const DEFAULT_VIDEO_BACKEND_URL = 'http://localhost:8005';
const CONFIGURED_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || DEFAULT_BACKEND_URL;
const CONFIGURED_VIDEO_BACKEND_URL = process.env.NEXT_PUBLIC_VIDEO_BACKEND_URL || DEFAULT_VIDEO_BACKEND_URL;
const PROCESS_IMAGE_TIMEOUT_MS = 10 * 60 * 1000;
const IMAGE_TO_MESH_TIMEOUT_MS = 60 * 60 * 1000;
const FULL_MESH_RUNNER_STL_POSTPROCESS = 'trimesh-repair';
const DEPTH_PRO_MODEL_ID = 'apple/DepthPro-hf';

type MediaKind = 'photo' | 'video';
type PhotoScope = 'whole-image' | 'object-selection';
type PhotoTarget = 'depth-relief' | 'full-mesh';
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

type ProviderPreflightResponse = {
  providers?: ProviderReadiness[];
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

type PrinterPresetId = 'bambulab-p1s' | 'custom';

type PrinterPreset = {
  id: PrinterPresetId;
  label: string;
  maxX: number;
  maxY: number;
  maxZ: number;
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
  { value: 'full-mesh', label: 'Full Mesh STL', icon: Cuboid },
];

const inpaintBackends = ['Mirror prior', 'SDXL inpaint', 'FLUX Fill', 'Qwen Image Edit'];
const resolutionMultipliers = [1, 1.5, 2, 3];
const depthModels: DepthModelOption[] = [
  {
    id: 'depth-anything/Depth-Anything-V2-Large-hf',
    label: 'Depth Anything V2 Large',
    tier: 'Best verified',
    notes: 'Best CUDA-backed option verified locally for relief STL generation.',
    farIsHigh: false,
  },
  {
    id: DEPTH_PRO_MODEL_ID,
    label: 'Apple Depth Pro',
    tier: 'Experimental detail',
    notes: 'Sharp metric-depth candidate; large first download, use after preload.',
    farIsHigh: true,
  },
  {
    id: 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf',
    label: 'DA V2 Metric Indoor Large',
    tier: 'Metric indoor',
    notes: 'Good for room, person, furniture, and object photos.',
    farIsHigh: true,
  },
  {
    id: 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf',
    label: 'DA V2 Metric Outdoor Large',
    tier: 'Metric outdoor',
    notes: 'Good for larger outdoor scenes.',
    farIsHigh: true,
  },
  {
    id: 'depth-anything/Depth-Anything-V2-Base-hf',
    label: 'Depth Anything V2 Base',
    tier: 'Balanced',
    notes: 'Good default for iteration speed and quality.',
    farIsHigh: false,
  },
  {
    id: 'depth-anything/Depth-Anything-V2-Small-hf',
    label: 'Depth Anything V2 Small',
    tier: 'Fast',
    notes: 'Fastest option for quick previews.',
    farIsHigh: false,
  },
];

const printerPresets: PrinterPreset[] = [
  { id: 'bambulab-p1s', label: 'Bambu Lab P1S', maxX: 256, maxY: 256, maxZ: 256 },
  { id: 'custom', label: 'Custom printer', maxX: 220, maxY: 220, maxZ: 220 },
];

const fallbackModelCatalog: ModelCatalog = {
  service: 'frontend-fallback',
  mode: 'planner-only',
  defaults: {
    selection: 'sam2.1-hiera-large',
    frame_selection: 'uniform-frame-sampler',
    camera_pose: 'hloc-lightglue',
    video_reconstruction: 'colmap-openmvs',
    image_to_mesh: 'triposg',
    stl_postprocess: 'trimesh-repair',
  },
  groups: {
    selection: [
      {
        id: 'sam2.1-hiera-large',
        label: 'SAM 2.1 Hiera Large',
        model: 'facebook/sam2.1-hiera-large',
        role: 'promptable object and video mask propagation',
        availability: 'configured',
      },
      {
        id: 'grounding-dino-sam2',
        label: 'Grounding DINO + SAM 2.1',
        model: 'IDEA-Research/GroundingDINO + facebook/sam2.1',
        role: 'text-prompted object box plus mask',
        availability: 'adapter-planned',
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
    completion_seconds: 'Complete',
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

function normalizeCatalog(data: Partial<ModelCatalog>): ModelCatalog {
  return {
    service: data.service || fallbackModelCatalog.service,
    mode: data.mode || fallbackModelCatalog.mode,
    defaults: { ...fallbackModelCatalog.defaults, ...(data.defaults || {}) },
    groups: {
      selection: data.groups?.selection?.length ? data.groups.selection : fallbackModelCatalog.groups.selection,
      frame_selection: data.groups?.frame_selection?.length ? data.groups.frame_selection : fallbackModelCatalog.groups.frame_selection,
      camera_pose: data.groups?.camera_pose?.length ? data.groups.camera_pose : fallbackModelCatalog.groups.camera_pose,
      video_reconstruction: data.groups?.video_reconstruction?.length
        ? data.groups.video_reconstruction
        : fallbackModelCatalog.groups.video_reconstruction,
      image_to_mesh: data.groups?.image_to_mesh?.length ? data.groups.image_to_mesh : fallbackModelCatalog.groups.image_to_mesh,
      stl_postprocess: data.groups?.stl_postprocess?.length ? data.groups.stl_postprocess : fallbackModelCatalog.groups.stl_postprocess,
    },
    metrics: data.metrics?.length ? data.metrics : fallbackModelCatalog.metrics,
    notes: data.notes || fallbackModelCatalog.notes,
  };
}

function modelsFor(catalog: ModelCatalog, group: ModelGroup) {
  return catalog.groups[group]?.length ? catalog.groups[group] : fallbackModelCatalog.groups[group];
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
  inpaintBackend,
  meshBackend,
  videoBackend,
}: {
  mediaKind: MediaKind;
  photoScope: PhotoScope;
  photoTarget: PhotoTarget;
  videoScope: VideoScope;
  selectedFrameCount: number;
  frameStep: number;
  inpaintBackend: string;
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

  if (mediaKind === 'photo') {
    return [
      {
        icon: Upload,
        label: 'Import photo',
        detail: photoScope === 'object-selection' ? 'Selected object only' : 'Whole visible subject',
      },
      {
        icon: Wand2,
        label: 'Complete hidden side',
        detail: inpaintBackend,
      },
      {
        icon: Cuboid,
        label: 'Image to mesh',
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
  const [mediaKind, setMediaKind] = useState<MediaKind>('photo');
  const [photoScope, setPhotoScope] = useState<PhotoScope>('whole-image');
  const [photoTarget, setPhotoTarget] = useState<PhotoTarget>('depth-relief');
  const [videoScope, setVideoScope] = useState<VideoScope>('selected-frames');
  const [selectedFrameCount, setSelectedFrameCount] = useState(12);
  const [frameStep, setFrameStep] = useState(8);
  const [depthModel, setDepthModel] = useState('depth-anything/Depth-Anything-V2-Large-hf');
  const [depthScale, setDepthScale] = useState(42);
  const [baseThickness, setBaseThickness] = useState(2.4);
  const [reliefPolarity, setReliefPolarity] = useState<ReliefPolarity>('raised-print');
  const [detailSmoothing, setDetailSmoothing] = useState(0.6);
  const [featureBoost, setFeatureBoost] = useState(1.4);
  const [reliefGamma, setReliefGamma] = useState(0.75);
  const [baseBorderPx, setBaseBorderPx] = useState(2);
  const [meshResolutionMultiplier, setMeshResolutionMultiplier] = useState(2);
  const [printerPreset, setPrinterPreset] = useState<PrinterPresetId>('bambulab-p1s');
  const [printerMaxX, setPrinterMaxX] = useState(256);
  const [printerMaxY, setPrinterMaxY] = useState(256);
  const [printerMaxZ, setPrinterMaxZ] = useState(256);
  const [printerClearance, setPrinterClearance] = useState(0);
  const [printScalePercent, setPrintScalePercent] = useState(100);
  const [meshBackend, setMeshBackend] = useState(fallbackModelCatalog.defaults.image_to_mesh);
  const [inpaintBackend, setInpaintBackend] = useState(inpaintBackends[0]);
  const [selectionModel, setSelectionModel] = useState(fallbackModelCatalog.defaults.selection);
  const [frameSelectionModel, setFrameSelectionModel] = useState(fallbackModelCatalog.defaults.frame_selection);
  const [cameraPoseModel, setCameraPoseModel] = useState(fallbackModelCatalog.defaults.camera_pose);
  const [videoBackend, setVideoBackend] = useState(fallbackModelCatalog.defaults.video_reconstruction);
  const [stlPostprocessModel, setStlPostprocessModel] = useState(fallbackModelCatalog.defaults.stl_postprocess);
  const [processedSTL, setProcessedSTL] = useState('');
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

  const selectionModels = modelsFor(modelCatalog, 'selection');
  const frameSelectionModels = modelsFor(modelCatalog, 'frame_selection');
  const cameraPoseModels = modelsFor(modelCatalog, 'camera_pose');
  const videoModels = modelsFor(modelCatalog, 'video_reconstruction');
  const meshModels = modelsFor(modelCatalog, 'image_to_mesh');
  const stlPostprocessModels = modelsFor(modelCatalog, 'stl_postprocess');
  const selectedMeshReadiness = providerReadiness[meshBackend];
  const currentPrinterPreset = printerPresets.find((preset) => preset.id === printerPreset) || printerPresets[0];
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
    };
  }, [printerPreset, currentPrinterPreset.label, printerMaxX, printerMaxY, printerMaxZ, printerClearance, printScalePercent, baseThickness]);
  const effectiveReliefHeight = Math.min(depthScale, printVolume.max_relief_height_mm);
  const reliefSliderMax = Math.max(12, Math.min(96, Math.floor(printVolume.max_relief_height_mm)));
  const reliefTargetDimension = Math.max(64, Math.round(printVolume.target_dimension_mm * meshResolutionMultiplier));
  const selectedDepthModel = depthModels.find((model) => model.id === depthModel) || depthModels[0];
  const reliefInvert = reliefPolarity === 'raised-print' ? selectedDepthModel.farIsHigh : !selectedDepthModel.farIsHigh;

  const applyPrinterPreset = (presetId: PrinterPresetId) => {
    const preset = printerPresets.find((candidate) => candidate.id === presetId) || printerPresets[0];
    setPrinterPreset(preset.id);
    setPrinterMaxX(preset.maxX);
    setPrinterMaxY(preset.maxY);
    setPrinterMaxZ(preset.maxZ);
  };

  useEffect(() => {
    setBackendUrl(CONFIGURED_BACKEND_URL);
    setVideoBackendUrl(CONFIGURED_VIDEO_BACKEND_URL);
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
        const data = (await response.json()) as Partial<ModelCatalog>;
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
        const data = (await response.json()) as ProviderPreflightResponse;
        const nextReadiness = Object.fromEntries((data.providers || []).map((provider) => [provider.id, provider]));
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
    if (!file) {
      setPreviewUrl('');
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    setMediaKind(mediaKindFromFile(file));
    setProcessedSTL('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setCompletedPreview('');
    setPlannerResponse(null);
    setRunState('idle');
    setStatusText('Ready');
    setError('');
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const steps = useMemo(
    () =>
      workflowSteps({
        mediaKind,
        photoScope,
        photoTarget,
        videoScope,
        selectedFrameCount,
        frameStep,
        inpaintBackend,
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
      inpaintBackend,
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
                resolution_multiplier: meshResolutionMultiplier,
                polarity: reliefPolarity,
                invert_depth: reliefInvert,
                smoothing_sigma: detailSmoothing,
                feature_boost: featureBoost,
                relief_gamma: reliefGamma,
                base_border_px: baseBorderPx,
              },
              completion:
                photoTarget === 'full-mesh'
                  ? {
                      inpaint_backend: inpaintBackend,
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
            }
          : {
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
      inpaintBackend,
      meshBackend,
      selectionModel,
      frameSelectionModel,
      cameraPoseModel,
      videoScope,
      selectedFrameCount,
      frameStep,
      videoBackend,
      stlPostprocessModel,
      modelCatalog,
      printVolume,
      effectiveReliefHeight,
      reliefTargetDimension,
      meshResolutionMultiplier,
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
    const stack: Array<{ group: ModelGroup; label: string; model: ModelOption }> = [];
    if (mediaKind === 'video') {
      stack.push(
        { group: 'frame_selection', label: 'Frames', model: modelFor(modelCatalog, 'frame_selection', frameSelectionModel) },
        { group: 'selection', label: 'Selection', model: modelFor(modelCatalog, 'selection', selectionModel) },
        { group: 'camera_pose', label: 'Camera', model: modelFor(modelCatalog, 'camera_pose', cameraPoseModel) },
        { group: 'video_reconstruction', label: 'Reconstruct', model: modelFor(modelCatalog, 'video_reconstruction', videoBackend) },
        { group: 'stl_postprocess', label: 'Repair', model: modelFor(modelCatalog, 'stl_postprocess', stlPostprocessModel) },
      );
      return stack;
    }

    if (photoScope === 'object-selection') {
      stack.push({ group: 'selection', label: 'Selection', model: modelFor(modelCatalog, 'selection', selectionModel) });
    }
    if (photoTarget === 'full-mesh') {
      stack.push(
        { group: 'image_to_mesh', label: 'Mesh', model: modelFor(modelCatalog, 'image_to_mesh', meshBackend) },
        { group: 'stl_postprocess', label: 'Repair', model: modelFor(modelCatalog, 'stl_postprocess', stlPostprocessModel) },
      );
    } else {
      stack.push({ group: 'stl_postprocess', label: 'Repair', model: modelFor(modelCatalog, 'stl_postprocess', stlPostprocessModel) });
    }
    return stack;
  }, [
    mediaKind,
    photoScope,
    photoTarget,
    modelCatalog,
    frameSelectionModel,
    selectionModel,
    cameraPoseModel,
    videoBackend,
    meshBackend,
    stlPostprocessModel,
  ]);

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) setFile(droppedFile);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (nextFile) setFile(nextFile);
  };

  const resetFile = () => {
    setFile(null);
    setProcessedSTL('');
    setDiagnosticsUrl('');
    setStlDiagnostics(null);
    setStageTimings(null);
    setDepthRunMetadata(null);
    setCompletedPreview('');
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

    setError('');
    setProcessedSTL('');
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
        formData.append('file', file, file.name || 'photo.jpg');
        formData.append('depth_provider', 'transformers');
        formData.append('depth_model', depthModel);
        formData.append('device', 'auto');
        formData.append('completion_mode', 'none');
        formData.append('target_dimension', String(reliefTargetDimension));
        formData.append('z_scale', String(effectiveReliefHeight));
        formData.append('max_xy_size', String(printVolume.target_dimension_mm));
        formData.append('invert', String(reliefInvert));
        formData.append('sigma', String(detailSmoothing));
        formData.append('detail_boost', String(featureBoost));
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
        formData.append('print_scale_percent', String(printVolume.print_scale_percent));

        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), PROCESS_IMAGE_TIMEOUT_MS);
        const response = await fetch(`${backendUrl}/process_image`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        window.clearTimeout(timeout);
        if (!response.ok) throw new Error(`Process image ${response.status}`);

        const data = await response.json();
        const stlUrl = data.stl_url ? `${backendUrl}${data.stl_url}` : `${backendUrl}/stl_model/${data.stl_model}`;
        setProcessedSTL(stlUrl);
        setDiagnosticsUrl(data.diagnostics_url ? `${backendUrl}${data.diagnostics_url}` : '');
        setStlDiagnostics(data.stl_diagnostics || null);
        if (data.runtime) {
          setBackendRuntime(data.runtime);
          setBackendRuntimeState('ready');
        }
        setStageTimings(data.timings || null);
        setDepthRunMetadata(data.depth_metadata || null);
        setCompletedPreview(data.completed_image_url ? `${backendUrl}${data.completed_image_url}` : previewUrl);
        setRunState('ready');
        setStatusText('STL ready');
      } catch (runError) {
        setRunState('error');
        setStatusText('Run failed');
        setError(runError instanceof Error ? runError.message : String(runError));
      }
      return;
    }

    setRunState('running');
    setStatusText(mediaKind === 'photo' && photoTarget === 'full-mesh' ? 'Running image-to-mesh' : 'Planning model stack');
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

      if (mediaKind === 'photo' && photoTarget === 'full-mesh') {
        const formData = new FormData();
        formData.append('file', file, file.name || 'photo.jpg');
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
        if (!runnerResponse.ok) throw new Error(`Image-to-mesh runner ${runnerResponse.status}`);
        const runnerData = await runnerResponse.json();
        setProcessedSTL(runnerData.stl_url ? `${videoBackendUrl}${runnerData.stl_url}` : '');
        setDiagnosticsUrl(runnerData.diagnostics_url ? `${videoBackendUrl}${runnerData.diagnostics_url}` : '');
        setStlDiagnostics(runnerData.stl_diagnostics || null);
        setStageTimings(runnerData.timings || null);
        setCompletedPreview(previewUrl);
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
      if (mediaKind === 'photo' && photoTarget === 'full-mesh') {
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

  const canRun = Boolean(file) && runState !== 'running';
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
    ? ['completion_seconds', 'depth_seconds', 'provider_seconds', 'stl_seconds', 'diagnostics_seconds', 'total_seconds']
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

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-950">
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-4 px-4 py-4 lg:h-screen lg:flex-row lg:overflow-hidden">
        <section className="flex min-h-0 flex-1 flex-col gap-4 lg:max-w-[430px]">
          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h1 className="text-xl font-semibold">Photo/Video to STL</h1>
                <p className="text-sm text-zinc-500">STL-first reconstruction workspace</p>
              </div>
              <Button variant="outline" size="icon" onClick={resetFile} title="Reset workspace">
                <RefreshCcw className="h-4 w-4" />
              </Button>
            </div>

            <label
              className="group relative flex h-[250px] cursor-pointer items-center justify-center overflow-hidden border border-dashed border-zinc-300 bg-zinc-100"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <>
                  {mediaKind === 'video' ? (
                    <video src={previewUrl} className="h-full w-full object-cover" muted playsInline controls />
                  ) : (
                    <img src={previewUrl} alt="" className="h-full w-full object-cover" />
                  )}
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
                <div className="flex flex-col items-center gap-3 text-center text-zinc-600">
                  <Upload className="h-10 w-10 text-blue-700" />
                  <div className="text-sm font-medium">Import image or video</div>
                </div>
              )}
              <input ref={fileInputRef} type="file" className="hidden" accept="image/*,video/*" onChange={handleFileInput} />
            </label>

            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <button
                type="button"
                onClick={() => setMediaKind('photo')}
                className={classNames(
                  'flex h-10 items-center justify-center gap-2 border',
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
                  'flex h-10 items-center justify-center gap-2 border',
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

          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center gap-2">
              <Braces className="h-5 w-5 text-emerald-700" />
              <h2 className="font-semibold">Run Plan</h2>
            </div>
            <pre className="max-h-[330px] overflow-auto bg-zinc-950 p-3 text-xs leading-5 text-zinc-100">
              {JSON.stringify(displayedPlan, null, 2)}
            </pre>
          </div>
        </section>

        <section className="flex min-h-0 flex-[1.35] flex-col gap-4 lg:overflow-auto">
          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
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
                  <label className="block text-sm font-medium text-zinc-700">
                    Selection model
                    <select
                      value={selectionModel}
                      onChange={(event) => setSelectionModel(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {selectionModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}

                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                  {photoTargets.map((target) => {
                    const Icon = target.icon;
                    return (
                      <button
                        key={target.value}
                        type="button"
                        onClick={() => setPhotoTarget(target.value)}
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
                    <label className="block text-sm font-medium text-zinc-700">
                      Depth model
                      <select
                        value={depthModel}
                        onChange={(event) => setDepthModel(event.target.value)}
                        className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                      >
                        {depthModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.label} - {model.tier}
                          </option>
                        ))}
                      </select>
                      <span className="mt-1 block text-xs text-zinc-500">{selectedDepthModel.notes}</span>
                    </label>

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
                          min="12"
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
                              {multiplier}x ({Math.round(printVolume.target_dimension_mm * multiplier)} samples)
                            </option>
                          ))}
                        </select>
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
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-3 md:grid-cols-2">
                    <label className="text-sm font-medium text-zinc-700">
                      Inpainting
                      <select
                        value={inpaintBackend}
                        onChange={(event) => setInpaintBackend(event.target.value)}
                        className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                      >
                        {inpaintBackends.map((backend) => (
                          <option key={backend}>{backend}</option>
                        ))}
                      </select>
                    </label>
                    <label className="text-sm font-medium text-zinc-700">
                      Mesh model
                      <select
                        value={meshBackend}
                        onChange={(event) => setMeshBackend(event.target.value)}
                        className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                      >
                        {meshModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.label}
                            {providerReadiness[model.id] ? ` - ${providerReadinessLabel(providerReadiness[model.id])}` : ''}
                          </option>
                        ))}
                      </select>
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
                    </label>
                    <label className="text-sm font-medium text-zinc-700 md:col-span-2">
                      STL repair
                      <select
                        value={stlPostprocessModel}
                        onChange={(event) => setStlPostprocessModel(event.target.value)}
                        className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                      >
                        {stlPostprocessModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
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

                <div className="grid gap-3 md:grid-cols-2">
                  <label className="text-sm font-medium text-zinc-700">
                    {videoScope === 'selected-frames' ? 'Selected frames' : 'Frame step'}
                    <input
                      className="mt-2 h-10 w-full border border-zinc-300 px-3"
                      type="number"
                      min="1"
                      max="240"
                      value={videoScope === 'selected-frames' ? selectedFrameCount : frameStep}
                      onChange={(event) =>
                        videoScope === 'selected-frames'
                          ? setSelectedFrameCount(Number(event.target.value))
                          : setFrameStep(Number(event.target.value))
                      }
                    />
                  </label>
                  <label className="text-sm font-medium text-zinc-700">
                    Frame selector
                    <select
                      value={frameSelectionModel}
                      onChange={(event) => setFrameSelectionModel(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {frameSelectionModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm font-medium text-zinc-700">
                    Selection model
                    <select
                      value={selectionModel}
                      onChange={(event) => setSelectionModel(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {selectionModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm font-medium text-zinc-700">
                    Camera/pose
                    <select
                      value={cameraPoseModel}
                      onChange={(event) => setCameraPoseModel(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {cameraPoseModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm font-medium text-zinc-700">
                    Reconstruction
                    <select
                      value={videoBackend}
                      onChange={(event) => setVideoBackend(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {videoModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm font-medium text-zinc-700">
                    STL repair
                    <select
                      value={stlPostprocessModel}
                      onChange={(event) => setStlPostprocessModel(event.target.value)}
                      className="mt-2 h-10 w-full border border-zinc-300 bg-white px-3"
                    >
                      {stlPostprocessModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="grid gap-2 md:grid-cols-2">
                  <div className="border border-zinc-200 bg-zinc-50 p-3 text-sm">
                    <div className="mb-1 font-medium">Camera/keypoint stream</div>
                    <div className="text-zinc-600">
                      {videoScope === 'selected-frames' ? 'Uncropped frames retained' : 'All sampled frames retained'}
                    </div>
                  </div>
                  <div className="border border-zinc-200 bg-zinc-50 p-3 text-sm">
                    <div className="mb-1 font-medium">Training target stream</div>
                    <div className="text-zinc-600">
                      {videoScope === 'selected-frames' ? 'Object masks only' : 'Full scene/object mesh'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-4 border-t border-zinc-200 pt-4">
              <div className="mb-2 text-sm font-semibold">Active Model Stack</div>
              <div className="grid gap-2 md:grid-cols-2">
                {activeModelStack.map(({ group, label, model }) => (
                  <article key={`${group}-${model.id}`} className="min-w-0 border border-zinc-200 bg-zinc-50 p-3 text-sm">
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

          <div className="grid gap-4 xl:grid-cols-4">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const done = runState === 'ready' || (runState === 'running' && index < 1);
              const active = runState === 'running' && index === 1;
              return (
                <article key={`${step.label}-${index}`} className="min-h-[128px] border border-zinc-200 bg-white p-4 shadow-sm">
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
        </section>

        <section className="flex min-h-0 flex-1 flex-col gap-4 lg:max-w-[390px]">
          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
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

          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
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

            <label className="mt-3 block text-sm font-medium text-zinc-700">
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
                {printScalePercent}% / {printVolume.target_dimension_mm} mm XY
              </span>
            </label>

            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">STL XY</div>
                <div className="font-semibold">{printVolume.target_dimension_mm} mm</div>
              </div>
              <div className="border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-xs font-medium uppercase text-zinc-500">Max relief Z</div>
                <div className="font-semibold">{printVolume.max_relief_height_mm.toFixed(1)} mm</div>
              </div>
            </div>

            <div className="mt-2 text-xs text-zinc-500">
              Build volume {printVolume.max_x_mm} x {printVolume.max_y_mm} x {printVolume.max_z_mm} mm
            </div>
          </div>

          <div className="border border-zinc-200 bg-white p-4 shadow-sm">
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
        </section>
      </div>
    </main>
  );
}
