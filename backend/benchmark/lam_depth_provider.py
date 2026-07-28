"""Run pinned LAM-20K camera-depth inference in an isolated Windows runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


LAM_SOURCE_URL = "https://github.com/3DAIGC/LAM.git"
LAM_SOURCE_REVISION = "339573649dd93df4cba8093a964e85a80d1b61f3"
LAM_MODEL_ID = "3DAIGC/LAM-20K"
LAM_MODEL_REVISION = "a710cd3c40c86ffe3fc572e895ed12f3ead47289"
LAM_MODEL_RELATIVE_PATH = Path(
    "model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
    "model.safetensors"
)
LAM_MODEL_SIZE_BYTES = 2_356_556_212
LAM_MODEL_SHA256 = (
    "f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925"
)

# The official one-click archive predates the current source pin. Its exact
# Windows compatibility patch is accepted only inside the research runtime.
LAM_WINDOWS_BUNDLE_URL = (
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/"
    "aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip"
)
LAM_WINDOWS_BUNDLE_SIZE_BYTES = 8_993_781_762
LAM_WINDOWS_BUNDLE_SHA256 = (
    "81ca564f84df14db995868685af42d612b51b3c82204d2f6fcd366d45a4a1121"
)
LAM_WINDOWS_SOURCE_REVISION = "4368fb6a5d5a930af6b1af79ef76392f8a6c4278"
LAM_WINDOWS_DIFF_SHA256 = (
    "c49ff720e4f6756e818430da13637aae299264b170c022868d40c4c7366a858d"
)
LAM_WINDOWS_DIRTY_PATHS = (
    "app_lam.py",
    "configs/vhap_tracking/base_tracking_config.yaml",
    "external/landmark_detection/FaceBoxesV2/utils/build.py",
    "external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.c",
    "external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx",
    "external/landmark_detection/FaceBoxesV2/utils/nms_wrapper.py",
    "external/vgghead_detector/VGGDetector.py",
    "tools/flame_tracking_single_image.py",
)
LAM_REQUIRED_FILES = (
    ".glut/python.exe",
    "configs/inference/lam-20k-8gpu.yaml",
    "lam/models/modeling_lam.py",
    "lam/models/rendering/gs_renderer.py",
    "lam/runners/infer/head_utils.py",
    "model_zoo/flame_tracking_models/68_keypoints_model.pkl",
    "model_zoo/flame_tracking_models/FaceBoxesV2.pth",
    "model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
    "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
    "tools/flame_tracking_single_image.py",
)
LAM_ASSET_SPECS = {
    "model_zoo/flame_tracking_models/68_keypoints_model.pkl": (
        245_650_943,
        "0f8a1532772f3baaba042117ca1399bad8d4e3c959bed6a991eaaae246352b86",
    ),
    "model_zoo/flame_tracking_models/FaceBoxesV2.pth": (
        4_153_573,
        "aae07fec4b62ac655508c06336662538803407852312ca5009fd93fb487d8cd7",
    ),
    "model_zoo/flame_tracking_models/matting/stylematte_synth.pt": (
        140_040_541,
        "5ce985571e909b6677d7d25e560216fa3f620e5cd337a8382ee0799c6d9af16c",
    ),
    "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd": (
        417_153_502,
        "18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74",
    ),
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl": (
        53_023_716,
        "8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0",
    ),
}
LAM_ASSET_PROVENANCE = {
    "model_zoo/flame_tracking_models/68_keypoints_model.pkl": {
        "bundle_source": LAM_WINDOWS_BUNDLE_URL,
        "upstream_terms": "not separately established",
        "production_eligible": False,
    },
    "model_zoo/flame_tracking_models/FaceBoxesV2.pth": {
        "bundle_source": LAM_WINDOWS_BUNDLE_URL,
        "upstream_reference": "https://github.com/yangfly/FaceBoxesV2",
        "upstream_terms": "checkpoint terms not separately established",
        "production_eligible": False,
    },
    "model_zoo/flame_tracking_models/matting/stylematte_synth.pt": {
        "bundle_source": LAM_WINDOWS_BUNDLE_URL,
        "upstream_terms": "checkpoint terms not separately established",
        "production_eligible": False,
    },
    "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd": {
        "bundle_source": LAM_WINDOWS_BUNDLE_URL,
        "upstream_reference": "https://github.com/NVlabs/VGGHeads",
        "upstream_terms": "checkpoint terms not separately established",
        "production_eligible": False,
    },
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl": {
        "bundle_source": LAM_WINDOWS_BUNDLE_URL,
        "upstream_reference": "https://flame.is.tue.mpg.de/",
        "upstream_terms": "registered FLAME research terms",
        "production_eligible": False,
    },
}
LAM_UNTRACKED_RUNTIME_SPECS = {
    "external/landmark_detection/FaceBoxesV2/utils/nms/"
    "cpu_nms.cp310-win_amd64.pyd": (
        84_480,
        "514051cd11969c1ea1b014666a1f246ea0d6be16ee2c04a2409adcb4b02c9a61",
    ),
    "glut.py": (
        62_114,
        "6e50dd4f1e2fa3a3bdc97e71d29b28f54694eba98a502dcd01ef071ea7a5b65f",
    ),
    "pyarmor_runtime_000000/__init__.py": (
        103,
        "ba1a5b4ac7e0f41c9d8e0b3ad2b21548cf883f76831a85984852df54c5ff32e9",
    ),
    "pyarmor_runtime_000000/pyarmor_runtime.pyd": (
        634_368,
        "dc582149656718394f7e260dadbeb67f3cabd3327a5ba70ab385fa3eb98a9371",
    ),
}
LAM_RUNTIME_VERSION_SPECS = {
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "numpy": "1.23.0",
    "Pillow": "10.4.0",
    "safetensors": "0.5.3",
    "omegaconf": "2.3.0",
    "huggingface-hub": "0.23.2",
    "transformers": "4.41.2",
    "accelerate": "1.6.0",
    "scipy": "1.13.1",
}
LAM_RUNTIME_BINARY_SPECS = {
    ".glut/python.exe": (
        103_192,
        "3cce33d75d6fdae4e004d0bdf149320b3147482a9caf370079dcb9c191a1b260",
    ),
    ".glut/python310.dll": (
        4_458_776,
        "14b06796f288bc6599e458fb23a944ab0c843e9868058f02a91d4606533505ed",
    ),
    ".glut/Lib/site-packages/numpy/core/"
    "_multiarray_umath.cp310-win_amd64.pyd": (
        2_681_856,
        "a07740d006895c0b56771889f2d8b438640c0d86f6c4e5056285412189c657e3",
    ),
    ".glut/Lib/site-packages/PIL/_imaging.cp310-win_amd64.pyd": (
        2_341_888,
        "433681b1e01606b1730d2a7f7811fab558e4e8c0fa91d403cfd8d7cc1a2b6a09",
    ),
    ".glut/Lib/site-packages/safetensors/_safetensors_rust.pyd": (
        687_616,
        "4cb45b0e12a81dba2341a8dbf575537e789e9268870953eb449c919f442da6ac",
    ),
    ".glut/Lib/site-packages/cv2/cv2.pyd": (
        74_698_752,
        "cb94fb2b0224b1ea1dffef2e69ed56467df2c57f00b8ce88c89ff76a08b39d00",
    ),
    ".glut/Lib/site-packages/torch/_C.cp310-win_amd64.pyd": (
        10_752,
        "60793d0c5abd5d327918a158611513059330602f0cb2cffdcd956a1474118b28",
    ),
    ".glut/Lib/site-packages/torch/lib/torch_cpu.dll": (
        252_805_632,
        "e393927b88c5e788a3cd5c826761194393f9e7b9d656c14ea550aec3c2d2844d",
    ),
    ".glut/Lib/site-packages/torch/lib/torch_cuda.dll": (
        1_417_319_424,
        "d098cee29e4d08bf72584ed19cb87d7dfe1b65d91412f9e371da7d963e456618",
    ),
    ".glut/Lib/site-packages/torchvision/_C.pyd": (
        8_944_640,
        "f3ae5ab6b9ba371ef167ddc578037348123dcbc8d356afcac7f48843583e4606",
    ),
}
CUDA_REDISTRIBUTABLE_SPECS = {
    "cuda_nvcc-windows-x86_64-12.8.61-archive.zip": (
        120_682_751,
        "d0158c4dd6565c0c9bca50d2e0bd5fd944d40333725a1c6875ef1840b719abfc",
    ),
    "cuda_cudart-windows-x86_64-12.8.57-archive.zip": (
        3_034_859,
        "2c7aa62a195d79229d4381c8bd0174a30502cf3d8124c6e94ee50a7fc8a1e9f4",
    ),
    "cuda_cccl-windows-x86_64-12.8.55-archive.zip": (
        2_911_311,
        "e218372c742d1ff2df9fbef82803e36c4fb05cffb51e9f123b380ad9c51e6965",
    ),
    "libcublas-windows-x86_64-12.8.3.14-archive.zip": (
        574_528_660,
        "a2f990cf61f0086d942632f2455727240baa9378c2e9fa2bdda56ef81f6cf8ad",
    ),
    "libcusolver-windows-x86_64-11.7.2.55-archive.zip": (
        246_448_372,
        "8478e9dbb6606dfaa7aa08c3f60372d93eedc721c701adb46eb1e4a3d1ef3854",
    ),
    "libcusparse-windows-x86_64-12.5.7.53-archive.zip": (
        285_389_488,
        "eb6d62a449390c91b22a72df5facdae8d6121736931dba75cfd1d2a9063af49d",
    ),
}
MSVC_VERSION = "14.44.35207"
WINDOWS_SDK_VERSION = "10.0.26100.0"
NVIDIA_REDISTRIBUTABLE_INDEX_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/"
    "redistrib_12.8.0.json"
)
DIFF_GAUSSIAN_SOURCE_URL = (
    "https://github.com/ashawkey/diff-gaussian-rasterization.git"
)
DIFF_GAUSSIAN_SOURCE_REVISION = (
    "8829d14f814fccdaf840b7b0f3021a616583c0a1"
)
DIFF_GAUSSIAN_GLM_REVISION = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
DIFF_GAUSSIAN_EXTENSION_SIZE_BYTES = 1_138_688
DIFF_GAUSSIAN_EXTENSION_SHA256 = (
    "2b67a0b15d1956715c071416313a2eade8c98212371261dd1c257b65c8edffaa"
)
LAM_RENDER_SIZE = 512
LAM_TRACKER_SIZE = 1024


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        stderr=subprocess.PIPE,
    )


def _file_records(root: Path, specs: dict[str, tuple[int, str]]) -> dict:
    records = {}
    for relative, (expected_size, expected_sha256) in specs.items():
        path = root / relative
        exists = path.is_file()
        size = int(path.stat().st_size) if exists else None
        sha256 = _sha256(path) if exists else None
        records[relative] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size,
            "expected_size_bytes": expected_size,
            "sha256": sha256,
            "expected_sha256": expected_sha256,
            "pinned": bool(size == expected_size and sha256 == expected_sha256),
        }
    return records


def _toolchain_layout(runtime_root: Path) -> dict[str, Path]:
    research_root = runtime_root.parents[2]
    sdk_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    sdk_root /= "Windows Kits/10"
    msvc_installed = (
        research_root / "VSBuildTools/VC/Tools/MSVC" / MSVC_VERSION
    )
    msvc_standalone = (
        research_root / "MSVCStandalone/VC/Tools/MSVC" / MSVC_VERSION
    )
    store_lib = (
        research_root
        / "VSExtract/Microsoft.VC.14.44.17.14.CRT.x64.Store.base/"
        "Contents/VC/Tools/MSVC"
        / MSVC_VERSION
        / "lib/x64"
    )
    desktop_lib = (
        research_root
        / "VSExtract/Microsoft.VC.14.44.17.14.CRT.x64.Desktop.base/"
        "Contents/VC/Tools/MSVC"
        / MSVC_VERSION
        / "lib/x64"
    )
    return {
        "research_root": research_root,
        "cuda": research_root / "CUDA128/toolkit",
        "cuda_downloads": research_root / "CUDA128/downloads",
        "msvc_installed": msvc_installed,
        "msvc_standalone": msvc_standalone,
        "msvc_bin": msvc_standalone / "bin/Hostx64/x64",
        "msvc_include": msvc_installed / "include",
        "msvc_store_lib": store_lib,
        "msvc_desktop_lib": desktop_lib,
        "vcvars64": research_root
        / "MSVCStandalone/VC/Auxiliary/Build/vcvars64.bat",
        "sdk": sdk_root,
        "torch_extensions": research_root / "LAM/torch_extensions_cuda128",
        "diff_gaussian_source": research_root
        / "LAM/diff-gaussian-rasterization",
    }


def _toolchain_file_specs(runtime_root: Path) -> dict[str, tuple[Path, int, str]]:
    layout = _toolchain_layout(runtime_root)
    sdk = layout["sdk"]
    sdk_version = WINDOWS_SDK_VERSION
    return {
        "cl.exe": (
            layout["msvc_bin"] / "cl.exe",
            677_736,
            "88c8344236a27a6e727e0a8edc49aaa2690bdc7a9464b9d18cc7abe70a9f1c0d",
        ),
        "link.exe": (
            layout["msvc_bin"] / "link.exe",
            3_252_576,
            "ca11e6c45debd34bf652dfe984c5360a531a005ed78bf72852330c9c2590cf0d",
        ),
        "vcvars64.bat": (
            layout["vcvars64"],
            99,
            "4b03600d3593415aa4eb59161a8d816556c7afb46326fb2b24df496e2ea4351a",
        ),
        "nvcc.exe": (
            layout["cuda"] / "bin/nvcc.exe",
            17_720_320,
            "bb26f719f90e6fbc86e8c125f407552ca57a0051974d81cf7d290b07e12a516a",
        ),
        "cuda.h": (
            layout["cuda"] / "include/cuda.h",
            1_183_268,
            "167e383a3791226c5e809dfd6a9614b29b1e11f4d6346c29a921943a71e3e566",
        ),
        "cudart.lib": (
            layout["cuda"] / "lib/x64/cudart.lib",
            117_462,
            "5dff3a32a498a7d8c3cb8c129bb7e01b0df4f0c21962c4e879f3870e886a42e4",
        ),
        "cublas_v2.h": (
            layout["cuda"] / "include/cublas_v2.h",
            15_938,
            "c00d426fbc7aa24c10702492c0df2530fcf45786fc3e78832a1dccb3fba2c4ee",
        ),
        "cublasLt.h": (
            layout["cuda"] / "include/cublasLt.h",
            104_962,
            "58c6326b7f5e98aacb84e4c770291a7e2365b0708cf829d633576b94a930e127",
        ),
        "cusparse.h": (
            layout["cuda"] / "include/cusparse.h",
            302_926,
            "a7750da76ba8d91b4f5135a0ffd482d04027b33b4573aeca9ad0969828ee70f8",
        ),
        "cusolverDn.h": (
            layout["cuda"] / "include/cusolverDn.h",
            157_038,
            "20b911a09c718b88e33188d51e2e0e8d88a7ec47b3ec6c9c197048a09e2ae2e3",
        ),
        "cublas.lib": (
            layout["cuda"] / "lib/x64/cublas.lib",
            154_888,
            "eff27870d15e889ef5687a2f18a18c2b41c5f994e7f381f2ec3939b6c06e4e6d",
        ),
        "cusparse.lib": (
            layout["cuda"] / "lib/x64/cusparse.lib",
            108_364,
            "7149422963c971d505832f27603bd8aaac0359e2187dc77c39737129d2a4a681",
        ),
        "cusolver.lib": (
            layout["cuda"] / "lib/x64/cusolver.lib",
            225_608,
            "c365c770b36f321e80974935359fe7342061b943d9824b4b8ef418883dcd020a",
        ),
        "windows.h": (
            sdk / "Include" / sdk_version / "um/windows.h",
            7_511,
            "b337d661d03a4abefb7b86a2742ce1ad5d19b57cd8b858bd13e7bbcc1dbeeaaa",
        ),
        "kernel32.lib": (
            sdk / "Lib" / sdk_version / "um/x64/kernel32.lib",
            311_908,
            "68566785e534f5d5692a3e79d19be4a4ba664db1ad4f178e88c4ec7c7da4b5c9",
        ),
        "ucrt.lib": (
            sdk / "Lib" / sdk_version / "ucrt/x64/ucrt.lib",
            285_588,
            "7ef4eac926bf597d2f243f16cdfed7e0db22cb3ca34a1d7e088a84c994a03d66",
        ),
    }


def _runtime_environment(runtime_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    layout = _toolchain_layout(runtime_root)
    sdk = layout["sdk"]
    sdk_version = WINDOWS_SDK_VERSION
    path_entries = (
        layout["cuda"] / "bin",
        layout["msvc_bin"],
        sdk / "bin" / sdk_version / "x64",
        runtime_root / ".glut" / "Lib" / "site-packages" / "torch" / "lib",
        runtime_root / ".glut" / "Scripts",
        runtime_root / ".glut" / "ffmpeg" / "bin",
        runtime_root / "blender",
    )
    environment["PATH"] = os.pathsep.join(
        [*(str(path) for path in path_entries), environment.get("PATH", "")]
    )
    environment["HF_HOME"] = str(runtime_root / "models")
    environment["TORCH_HOME"] = str(runtime_root / "models")
    environment["MODELSCOPE_CACHE"] = str(runtime_root)
    environment["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["CUDA_HOME"] = str(layout["cuda"])
    environment["CUDA_PATH"] = str(layout["cuda"])
    environment["INCLUDE"] = os.pathsep.join(
        str(path)
        for path in (
            layout["msvc_include"],
            sdk / "Include" / sdk_version / "ucrt",
            sdk / "Include" / sdk_version / "shared",
            sdk / "Include" / sdk_version / "um",
            sdk / "Include" / sdk_version / "winrt",
            sdk / "Include" / sdk_version / "cppwinrt",
            layout["cuda"] / "include",
        )
    )
    library_paths = (
        layout["msvc_store_lib"],
        layout["msvc_desktop_lib"],
        sdk / "Lib" / sdk_version / "um/x64",
        sdk / "Lib" / sdk_version / "ucrt/x64",
        layout["cuda"] / "lib/x64",
    )
    environment["LIB"] = os.pathsep.join(str(path) for path in library_paths)
    environment["LIBPATH"] = environment["LIB"]
    environment["DISTUTILS_USE_SDK"] = "1"
    environment["MSSdk"] = "1"
    environment["VSCMD_ARG_TGT_ARCH"] = "x64"
    environment["VSCMD_ARG_HOST_ARCH"] = "x64"
    environment["TORCH_EXTENSIONS_DIR"] = str(layout["torch_extensions"])
    environment["PYTHONPATH"] = str(layout["diff_gaussian_source"])
    environment["PYTHONHOME"] = ""
    return environment


def _install_eager_torch_compile(torch_module) -> dict:
    """Keep official decorators but force their documented eager path."""

    original_compile = torch_module.compile

    def eager_compile(model=None, *args, **kwargs):
        if args:
            raise TypeError("torch.compile accepts only one positional argument")
        kwargs["disable"] = True
        return original_compile(model, **kwargs)

    torch_module.compile = eager_compile
    return {
        "method": "torch.compile-disable-true",
        "reason": "official Windows bundle has no Triton runtime",
        "numerical_mode": "eager",
        "source_modified": False,
    }


def lam_runtime_preflight(runtime_root: str | Path) -> dict:
    """Verify source, weights, toolchain, and native CUDA rasterizers."""

    runtime_root = Path(runtime_root).resolve()
    python_path = runtime_root / ".glut" / "python.exe"
    model_path = runtime_root / LAM_MODEL_RELATIVE_PATH
    missing_files = [
        relative
        for relative in LAM_REQUIRED_FILES
        if not (runtime_root / relative).is_file()
    ]
    source_revision = None
    dirty_paths: list[str] = []
    untracked_importable_paths: list[str] = []
    diff_sha256 = None
    git_error = None
    try:
        source_revision = _git_bytes(runtime_root, "rev-parse", "HEAD").decode(
            "utf-8"
        ).strip()
        status = _git_bytes(
            runtime_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ).decode("utf-8")
        dirty_paths = sorted(
            line[3:].strip().replace("\\", "/")
            for line in status.splitlines()
            if line.strip()
        )
        diff = _git_bytes(
            runtime_root,
            "diff",
            "HEAD",
            "--no-ext-diff",
            "--binary",
        )
        diff_sha256 = hashlib.sha256(diff).hexdigest()
        untracked = _git_bytes(
            runtime_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).decode("utf-8", errors="surrogateescape")
        ignored_runtime_prefixes = (
            ".glut/",
            "assets/sample_oac/",
            "blender/",
            "models/",
        )
        importable_suffixes = (".py", ".pyd")
        untracked_importable_paths = sorted(
            path.replace("\\", "/")
            for path in untracked.split("\0")
            if path
            and not path.replace("\\", "/").startswith(
                ignored_runtime_prefixes
            )
            and path.lower().endswith(importable_suffixes)
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        git_error = f"{type(exc).__name__}: {exc}"

    model_exists = model_path.is_file()
    model_size = int(model_path.stat().st_size) if model_exists else None
    model_sha256 = _sha256(model_path) if model_exists else None
    asset_records = _file_records(runtime_root, LAM_ASSET_SPECS)
    for name, record in asset_records.items():
        record["provenance"] = LAM_ASSET_PROVENANCE.get(name)
    untracked_runtime_records = _file_records(
        runtime_root,
        LAM_UNTRACKED_RUNTIME_SPECS,
    )
    runtime_binary_records = _file_records(
        runtime_root,
        LAM_RUNTIME_BINARY_SPECS,
    )
    layout = _toolchain_layout(runtime_root)
    cuda_archive_records = _file_records(
        layout["cuda_downloads"],
        CUDA_REDISTRIBUTABLE_SPECS,
    )
    toolchain_records = {}
    for name, (path, expected_size, expected_sha256) in (
        _toolchain_file_specs(runtime_root).items()
    ):
        exists = path.is_file()
        size = int(path.stat().st_size) if exists else None
        sha256 = _sha256(path) if exists else None
        toolchain_records[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size,
            "expected_size_bytes": expected_size,
            "sha256": sha256,
            "expected_sha256": expected_sha256,
            "pinned": bool(
                size == expected_size and sha256 == expected_sha256
            ),
        }
    diff_gaussian_root = layout["diff_gaussian_source"]
    diff_gaussian_revision = None
    diff_gaussian_status = None
    diff_gaussian_glm_revision = None
    diff_gaussian_error = None
    try:
        diff_gaussian_revision = _git_bytes(
            diff_gaussian_root,
            "rev-parse",
            "HEAD",
        ).decode("utf-8").strip()
        diff_gaussian_status = _git_bytes(
            diff_gaussian_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ).decode("utf-8").strip()
        submodule_status = _git_bytes(
            diff_gaussian_root,
            "submodule",
            "status",
            "third_party/glm",
        ).decode("utf-8").strip()
        if submodule_status:
            diff_gaussian_glm_revision = submodule_status.split()[0].lstrip(
                "+-U"
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        diff_gaussian_error = f"{type(exc).__name__}: {exc}"
    diff_gaussian_extension = (
        diff_gaussian_root
        / "diff_gaussian_rasterization/_C.cp310-win_amd64.pyd"
    )
    diff_gaussian_extension_exists = diff_gaussian_extension.is_file()
    diff_gaussian_extension_size = (
        int(diff_gaussian_extension.stat().st_size)
        if diff_gaussian_extension_exists
        else None
    )
    diff_gaussian_extension_sha256 = (
        _sha256(diff_gaussian_extension)
        if diff_gaussian_extension_exists
        else None
    )
    runtime_probe = None
    runtime_probe_error = None
    if python_path.is_file():
        probe = (
            "import importlib.metadata as md,json,shutil,subprocess,sys,torch; "
            f"names={json.dumps(list(LAM_RUNTIME_VERSION_SPECS))}; "
            "import diff_gaussian_rasterization as d; "
            "import nvdiffrast.torch as dr; "
            "ctx=dr.RasterizeCudaContext(); "
            "dev='cuda'; eye=torch.eye(4,device=dev); "
            "means=torch.tensor([[0.,0.,.5]],device=dev); "
            "means2=torch.zeros_like(means); "
            "op=torch.ones((1,1),device=dev)*.8; "
            "scales=torch.ones((1,3),device=dev)*.02; "
            "rot=torch.tensor([[1.,0.,0.,0.]],device=dev); "
            "colors=torch.tensor([[1.,0.,0.]],device=dev); "
            "settings=d.GaussianRasterizationSettings(16,16,1.,1.,"
            "torch.zeros(3,device=dev),1.,eye,eye,0,"
            "torch.zeros(3,device=dev),False,False); "
            "gout=d.GaussianRasterizer(settings)(means,means2,op,"
            "colors_precomp=colors,scales=scales,rotations=rot); "
            "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
            "'cuda_build':torch.version.cuda,'cuda_available':"
            "torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else None,'vram_bytes':"
            "torch.cuda.get_device_properties(0).total_memory if "
            "torch.cuda.is_available() else None,'gaussian_rasterizer':d.__file__,"
            "'package_versions':{name:md.version(name) for name in names},"
            "'gaussian_extension':d._C.__file__,'gaussian_smoke_shapes':"
            "[list(x.shape) for x in gout],'gaussian_smoke_finite':"
            "all(bool(torch.isfinite(x).all()) for x in gout),"
            "'nvdiffrast_context':type(ctx).__name__,'cl':shutil.which('cl.exe'),"
            "'link':shutil.which('link.exe'),'nvcc':shutil.which('nvcc.exe'),"
            "'nvcc_version':subprocess.check_output([shutil.which('nvcc.exe'),"
            "'--version'],text=True).strip()}))"
        )
        try:
            completed = subprocess.run(
                (str(python_path), "-c", probe),
                cwd=runtime_root,
                env=_runtime_environment(runtime_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            runtime_probe = json.loads(completed.stdout.splitlines()[-1])
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as exc:
            runtime_probe_error = f"{type(exc).__name__}: {exc}"

    checks = {
        "runtime_exists": runtime_root.is_dir(),
        "required_files_complete": not missing_files,
        "source_revision_pinned": source_revision == LAM_WINDOWS_SOURCE_REVISION,
        "windows_patch_paths_pinned": dirty_paths
        == sorted(LAM_WINDOWS_DIRTY_PATHS),
        "windows_patch_hash_pinned": diff_sha256 == LAM_WINDOWS_DIFF_SHA256,
        "untracked_importable_paths_pinned": untracked_importable_paths
        == sorted(LAM_UNTRACKED_RUNTIME_SPECS),
        "untracked_runtime_files_pinned": all(
            record["pinned"] for record in untracked_runtime_records.values()
        ),
        "runtime_binaries_pinned": all(
            record["pinned"] for record in runtime_binary_records.values()
        ),
        "model_exists": model_exists,
        "model_size_pinned": model_size == LAM_MODEL_SIZE_BYTES,
        "model_hash_pinned": model_sha256 == LAM_MODEL_SHA256,
        "inference_assets_pinned": all(
            record["pinned"] for record in asset_records.values()
        ),
        "inference_asset_provenance_fail_closed": bool(
            set(LAM_ASSET_PROVENANCE) == set(LAM_ASSET_SPECS)
            and all(
                record.get("production_eligible") is False
                and record.get("bundle_source") == LAM_WINDOWS_BUNDLE_URL
                and record.get("upstream_terms")
                for record in LAM_ASSET_PROVENANCE.values()
            )
        ),
        "cuda_redistributables_pinned": all(
            record["pinned"] for record in cuda_archive_records.values()
        ),
        "toolchain_files_pinned": all(
            record["pinned"] for record in toolchain_records.values()
        ),
        "diff_gaussian_source_pinned": diff_gaussian_revision
        == DIFF_GAUSSIAN_SOURCE_REVISION,
        "diff_gaussian_source_clean": diff_gaussian_status == "",
        "diff_gaussian_glm_pinned": diff_gaussian_glm_revision
        == DIFF_GAUSSIAN_GLM_REVISION,
        "diff_gaussian_extension_pinned": bool(
            diff_gaussian_extension_size
            == DIFF_GAUSSIAN_EXTENSION_SIZE_BYTES
            and diff_gaussian_extension_sha256
            == DIFF_GAUSSIAN_EXTENSION_SHA256
        ),
        "runtime_probe_passed": runtime_probe is not None,
        "runtime_python_pinned": bool(
            runtime_probe
            and str(runtime_probe.get("python", "")).startswith("3.10.11 ")
        ),
        "runtime_package_versions_pinned": bool(
            runtime_probe
            and runtime_probe.get("package_versions")
            == LAM_RUNTIME_VERSION_SPECS
        ),
        "torch_cuda_build_pinned": bool(
            runtime_probe
            and runtime_probe.get("torch") == LAM_RUNTIME_VERSION_SPECS["torch"]
            and runtime_probe.get("cuda_build") == "12.8"
        ),
        "cuda_available": bool(runtime_probe and runtime_probe["cuda_available"]),
        "native_rasterizer_loaded": bool(
            runtime_probe
            and runtime_probe.get("nvdiffrast_context")
            == "RasterizeCudaContext"
        ),
        "gaussian_rasterizer_smoke_passed": bool(
            runtime_probe
            and runtime_probe.get("gaussian_smoke_finite")
            and runtime_probe.get("gaussian_smoke_shapes")
            == [[3, 16, 16], [1], [1, 16, 16], [1, 16, 16]]
        ),
    }
    return {
        "schema_version": 2,
        "provider": "lam-20k-camera-depth",
        "runtime_root": str(runtime_root),
        "source": {
            "url": LAM_SOURCE_URL,
            "current_reference_revision": LAM_SOURCE_REVISION,
            "bundle_revision": source_revision,
            "expected_bundle_revision": LAM_WINDOWS_SOURCE_REVISION,
            "dirty_paths": dirty_paths,
            "expected_dirty_paths": list(LAM_WINDOWS_DIRTY_PATHS),
            "diff_sha256": diff_sha256,
            "expected_diff_sha256": LAM_WINDOWS_DIFF_SHA256,
            "untracked_importable_paths": untracked_importable_paths,
            "expected_untracked_importable_paths": sorted(
                LAM_UNTRACKED_RUNTIME_SPECS
            ),
            "error": git_error,
        },
        "bundle": {
            "url": LAM_WINDOWS_BUNDLE_URL,
            "size_bytes": LAM_WINDOWS_BUNDLE_SIZE_BYTES,
            "sha256": LAM_WINDOWS_BUNDLE_SHA256,
        },
        "model": {
            "id": LAM_MODEL_ID,
            "revision": LAM_MODEL_REVISION,
            "path": str(model_path),
            "size_bytes": model_size,
            "expected_size_bytes": LAM_MODEL_SIZE_BYTES,
            "sha256": model_sha256,
            "expected_sha256": LAM_MODEL_SHA256,
        },
        "assets": asset_records,
        "untracked_runtime_files": untracked_runtime_records,
        "runtime_binaries": runtime_binary_records,
        "runtime_package_versions": {
            "actual": (
                runtime_probe.get("package_versions")
                if runtime_probe
                else None
            ),
            "expected": LAM_RUNTIME_VERSION_SPECS,
        },
        "toolchain": {
            "nvidia_redistributable_index": NVIDIA_REDISTRIBUTABLE_INDEX_URL,
            "cuda_archives": cuda_archive_records,
            "files": toolchain_records,
            "windows_sdk_version": WINDOWS_SDK_VERSION,
            "msvc_version": MSVC_VERSION,
            "torch_extensions_dir": str(layout["torch_extensions"]),
            "python_path": str(diff_gaussian_root),
            "diff_gaussian_rasterizer": {
                "source_url": DIFF_GAUSSIAN_SOURCE_URL,
                "source_revision": diff_gaussian_revision,
                "expected_source_revision": DIFF_GAUSSIAN_SOURCE_REVISION,
                "tracked_status": diff_gaussian_status,
                "glm_revision": diff_gaussian_glm_revision,
                "expected_glm_revision": DIFF_GAUSSIAN_GLM_REVISION,
                "extension_path": str(diff_gaussian_extension),
                "extension_size_bytes": diff_gaussian_extension_size,
                "expected_extension_size_bytes": (
                    DIFF_GAUSSIAN_EXTENSION_SIZE_BYTES
                ),
                "extension_sha256": diff_gaussian_extension_sha256,
                "expected_extension_sha256": (
                    DIFF_GAUSSIAN_EXTENSION_SHA256
                ),
                "error": diff_gaussian_error,
                "reason": (
                    "official prebuilt requested 156.02 GiB for a bounded "
                    "20,018-Gaussian render"
                ),
            },
        },
        "missing_files": missing_files,
        "runtime": runtime_probe,
        "runtime_error": runtime_probe_error,
        "license": {
            "source_code": "Apache-2.0",
            "primary_model_weights": "CC BY-NC 4.0",
            "auxiliary_assets": "mixed or separately unestablished terms",
            "all_asset_terms_established": False,
            "auxiliary_asset_manifest": LAM_ASSET_PROVENANCE,
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def source_to_render_maps(
    source_shape: tuple[int, int],
    tracker_crop_xyxy: tuple[int, int, int, int] | list[int],
    preprocessor_crop_xywh: tuple[int, int, int, int] | list[int],
    *,
    tracker_size: int = LAM_TRACKER_SIZE,
    render_size: int = LAM_RENDER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Map original source pixel centers into the native LAM render."""

    height, width = (int(value) for value in source_shape)
    left, top, right, bottom = (
        int(value) for value in tracker_crop_xyxy
    )
    offset_x, offset_y, crop_width, crop_height = (
        int(value) for value in preprocessor_crop_xywh
    )
    if height < 2 or width < 2:
        raise ValueError("LAM source image is too small")
    if right <= left or bottom <= top:
        raise ValueError("LAM tracker crop is empty")
    if crop_width <= 1 or crop_height <= 1:
        raise ValueError("LAM preprocessor crop is empty")
    if tracker_size < 2 or render_size < 2:
        raise ValueError("LAM render dimensions must be at least two pixels")

    source_y, source_x = np.mgrid[0:height, 0:width].astype(np.float32)
    tracker_x = (
        (source_x - left + 0.5) * tracker_size / (right - left) - 0.5
    )
    tracker_y = (
        (source_y - top + 0.5) * tracker_size / (bottom - top) - 0.5
    )
    render_x = (
        (tracker_x - offset_x + 0.5) * render_size / crop_width - 0.5
    )
    render_y = (
        (tracker_y - offset_y + 0.5) * render_size / crop_height - 0.5
    )
    return render_x.astype(np.float32), render_y.astype(np.float32)


def warp_lam_depth_to_source(
    render_depth_weighted: np.ndarray,
    render_mask: np.ndarray,
    source_shape: tuple[int, int],
    tracker_crop_xyxy: tuple[int, int, int, int] | list[int],
    preprocessor_crop_xywh: tuple[int, int, int, int] | list[int],
    *,
    minimum_mask: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp weighted depth and alpha, then recover source camera depth."""

    import cv2

    render_depth_weighted = np.asarray(
        render_depth_weighted,
        dtype=np.float32,
    ).squeeze()
    render_mask = np.asarray(render_mask, dtype=np.float32).squeeze()
    if render_depth_weighted.shape != (LAM_RENDER_SIZE, LAM_RENDER_SIZE):
        raise ValueError(
            f"Unexpected LAM depth shape: {render_depth_weighted.shape}"
        )
    if render_mask.shape != render_depth_weighted.shape:
        raise ValueError("LAM depth and mask shapes differ")
    if not math.isfinite(float(minimum_mask)) or not 0 <= minimum_mask <= 1:
        raise ValueError("LAM minimum mask must be between zero and one")
    map_x, map_y = source_to_render_maps(
        source_shape,
        tracker_crop_xyxy,
        preprocessor_crop_xywh,
    )
    source_depth_weighted = cv2.remap(
        render_depth_weighted,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    source_mask = cv2.remap(
        render_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    finite = (
        np.isfinite(source_depth_weighted)
        & np.isfinite(source_mask)
        & (source_mask >= minimum_mask)
        & (source_depth_weighted > 0)
    )
    source_depth = np.full(source_mask.shape, np.nan, dtype=np.float32)
    source_depth[finite] = (
        source_depth_weighted[finite] / source_mask[finite]
    )
    source_mask[~np.isfinite(source_mask)] = 0.0
    return source_depth.astype(np.float32), source_mask.astype(np.float32)


def normalize_lam_comp_depth(
    weighted_depth: np.ndarray,
    render_mask: np.ndarray,
    *,
    minimum_mask: float = 0.01,
) -> np.ndarray:
    """Convert native alpha-weighted comp_depth to expected camera depth."""

    weighted_depth = np.asarray(weighted_depth, dtype=np.float32).squeeze()
    render_mask = np.asarray(render_mask, dtype=np.float32).squeeze()
    if weighted_depth.shape != (LAM_RENDER_SIZE, LAM_RENDER_SIZE):
        raise ValueError(f"Unexpected LAM depth shape: {weighted_depth.shape}")
    if render_mask.shape != weighted_depth.shape:
        raise ValueError("LAM weighted depth and mask shapes differ")
    if not math.isfinite(float(minimum_mask)) or not 0 < minimum_mask <= 1:
        raise ValueError("LAM normalization mask must be in (0, 1]")
    if not np.all(np.isfinite(render_mask)):
        raise ValueError("LAM render mask contains non-finite values")
    if float(np.min(render_mask)) < -1e-5 or float(np.max(render_mask)) > 1.00001:
        raise ValueError("LAM render mask is outside [0, 1]")
    valid = (
        np.isfinite(weighted_depth)
        & (weighted_depth > 0)
        & (render_mask >= minimum_mask)
    )
    normalized = np.full(weighted_depth.shape, np.nan, dtype=np.float32)
    normalized[valid] = weighted_depth[valid] / render_mask[valid]
    return normalized


def validate_lam_camera_depth(
    render_depth: np.ndarray,
    render_mask: np.ndarray,
    source_depth: np.ndarray,
    source_mask: np.ndarray,
    *,
    selection_mask: np.ndarray | None = None,
) -> dict:
    """Fail closed on empty, non-finite, or degenerate native camera depth."""

    render_depth = np.asarray(render_depth, dtype=np.float32).squeeze()
    render_mask = np.asarray(render_mask, dtype=np.float32).squeeze()
    source_depth = np.asarray(source_depth, dtype=np.float32).squeeze()
    source_mask = np.asarray(source_mask, dtype=np.float32).squeeze()
    if render_depth.shape != (LAM_RENDER_SIZE, LAM_RENDER_SIZE):
        raise ValueError(f"Unexpected LAM depth shape: {render_depth.shape}")
    if render_mask.shape != render_depth.shape:
        raise ValueError("LAM render depth and mask shapes differ")
    if source_depth.ndim != 2 or source_mask.shape != source_depth.shape:
        raise ValueError("LAM source depth and mask shapes differ")
    for name, mask in (("render", render_mask), ("source", source_mask)):
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"LAM {name} mask contains non-finite values")
        if float(np.min(mask)) < -1e-5 or float(np.max(mask)) > 1.00001:
            raise ValueError(f"LAM {name} mask is outside [0, 1]")

    render_valid = (
        (render_mask >= 0.01)
        & np.isfinite(render_depth)
        & (render_depth > 0)
    )
    source_valid = (
        (source_mask >= 0.01)
        & np.isfinite(source_depth)
        & (source_depth > 0)
    )
    if int(np.count_nonzero(render_valid)) < 256:
        raise ValueError("LAM render depth has too few valid pixels")
    if int(np.count_nonzero(source_valid)) < 64:
        raise ValueError("LAM source depth has too few valid pixels")
    render_values = render_depth[render_valid]
    source_values = source_depth[source_valid]
    render_p05, render_p95 = np.percentile(render_values, (5.0, 95.0))
    source_p05, source_p95 = np.percentile(source_values, (5.0, 95.0))
    if not float(render_p95 - render_p05) > 1e-5:
        raise ValueError("LAM render depth is degenerate")
    if not float(source_p95 - source_p05) > 1e-5:
        raise ValueError("LAM source depth is degenerate")

    selection_overlap = None
    if selection_mask is not None:
        selection = np.asarray(selection_mask).squeeze()
        if selection.shape != source_depth.shape:
            raise ValueError("LAM selection mask shape differs from source")
        selected = np.isfinite(selection) & (selection > 0.5)
        selected_count = int(np.count_nonzero(selected))
        if selected_count < 64:
            raise ValueError("LAM selection mask has too few selected pixels")
        selection_overlap = float(
            np.count_nonzero(source_valid & selected) / selected_count
        )
        if selection_overlap < 0.05:
            raise ValueError("LAM depth misses the selected face")

    return {
        "render_valid_pixels": int(np.count_nonzero(render_valid)),
        "render_coverage_ratio": float(np.mean(render_valid)),
        "render_depth_p05": float(render_p05),
        "render_depth_p50": float(np.median(render_values)),
        "render_depth_p95": float(render_p95),
        "source_valid_pixels": int(np.count_nonzero(source_valid)),
        "source_coverage_ratio": float(np.mean(source_valid)),
        "source_depth_p05": float(source_p05),
        "source_depth_p50": float(np.median(source_values)),
        "source_depth_p95": float(source_p95),
        "selection_overlap_ratio": selection_overlap,
    }


def run_lam_camera_depth(
    runtime_root: str | Path,
    source_path: str | Path,
    output_dir: str | Path,
    *,
    selection_mask_path: str | Path | None = None,
    expected_source_sha256: str | None = None,
    expected_selection_mask_sha256: str | None = None,
    timeout_seconds: float = 3600.0,
) -> dict:
    """Run the isolated worker after a fail-closed host preflight."""

    runtime_root = Path(runtime_root).resolve()
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = _sha256(source_path)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise RuntimeError("LAM source image hash changed")
    selection_mask = (
        Path(selection_mask_path).resolve()
        if selection_mask_path is not None
        else None
    )
    if selection_mask is not None:
        if not selection_mask.is_file():
            raise FileNotFoundError(selection_mask)
        selection_mask_sha256 = _sha256(selection_mask)
        if (
            expected_selection_mask_sha256
            and selection_mask_sha256 != expected_selection_mask_sha256
        ):
            raise RuntimeError("LAM selection mask hash changed")
    preflight = lam_runtime_preflight(runtime_root)
    if not preflight["runnable"]:
        failed = [
            name for name, passed in preflight["checks"].items() if not passed
        ]
        raise RuntimeError("LAM preflight failed: " + ", ".join(failed))
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "lam_preflight.json"
    _write_json(preflight_path, preflight)
    preflight_sha256 = _sha256(preflight_path)
    command = (
        str(runtime_root / ".glut" / "python.exe"),
        str(Path(__file__).resolve()),
        "--runtime-worker",
        "--runtime-root",
        str(runtime_root),
        "--source-path",
        str(source_path),
        "--output-dir",
        str(output_dir),
        "--expected-source-sha256",
        source_sha256,
        "--preflight-path",
        str(preflight_path),
        "--expected-preflight-sha256",
        preflight_sha256,
    )
    if selection_mask is not None:
        command += (
            "--selection-mask-path",
            str(selection_mask),
            "--expected-selection-mask-sha256",
            _sha256(selection_mask),
        )
    completed = subprocess.run(
        command,
        cwd=runtime_root,
        env=_runtime_environment(runtime_root),
        check=False,
        text=True,
        capture_output=True,
        timeout=float(timeout_seconds),
    )
    (output_dir / "worker_stdout.txt").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_dir / "worker_stderr.txt").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"LAM worker failed with exit code {completed.returncode}; "
            f"see {output_dir}"
        )
    metadata_path = output_dir / "provider_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("LAM worker emitted no metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_artifacts = {
        "render_depth_weighted": (LAM_RENDER_SIZE, LAM_RENDER_SIZE),
        "render_depth": (LAM_RENDER_SIZE, LAM_RENDER_SIZE),
        "render_mask": (LAM_RENDER_SIZE, LAM_RENDER_SIZE),
        "source_depth": None,
        "source_mask": None,
    }
    arrays = {}
    for name, expected_shape in expected_artifacts.items():
        record = metadata.get("artifacts", {}).get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"LAM worker omitted {name} artifact metadata")
        artifact = (output_dir / str(record.get("file", ""))).resolve()
        if artifact.parent != output_dir or not artifact.is_file():
            raise RuntimeError(f"LAM worker emitted invalid {name} artifact path")
        if _sha256(artifact) != record.get("sha256"):
            raise RuntimeError(f"LAM worker {name} artifact hash changed")
        arrays[name] = np.load(artifact, allow_pickle=False)
        if expected_shape is not None and arrays[name].shape != expected_shape:
            raise RuntimeError(f"LAM worker {name} artifact shape changed")
    if arrays["source_depth"].shape != arrays["source_mask"].shape:
        raise RuntimeError("LAM worker source artifact shapes differ")
    selection_array = None
    if selection_mask is not None:
        from PIL import Image

        with Image.open(selection_mask) as loaded:
            selection_array = np.asarray(loaded.convert("L"), dtype=np.float32)
        selection_array /= 255.0
    host_depth_validation = validate_lam_camera_depth(
        arrays["render_depth"],
        arrays["render_mask"],
        arrays["source_depth"],
        arrays["source_mask"],
        selection_mask=selection_array,
    )
    for camera_name, expected_shape in (
        ("render_c2w", (4, 4)),
        ("render_intrinsics", (4, 4)),
    ):
        camera_value = np.asarray(
            metadata.get("camera", {}).get(camera_name),
            dtype=np.float64,
        )
        if (
            camera_value.shape != expected_shape
            or not np.all(np.isfinite(camera_value))
        ):
            raise RuntimeError(f"LAM worker emitted invalid {camera_name}")
    metadata["preflight_file"] = {
        "file": preflight_path.name,
        "sha256": preflight_sha256,
    }
    metadata["host_depth_validation"] = host_depth_validation
    _write_json(metadata_path, metadata)
    return metadata


def _depth_preview(path: Path, depth: np.ndarray) -> None:
    from PIL import Image

    finite = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(finite):
        low, high = np.percentile(depth[finite], (2.0, 98.0))
        span = max(float(high - low), 1e-8)
        normalized = np.clip((depth - low) / span, 0.0, 1.0)
        preview[finite] = np.rint(normalized[finite] * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def _runtime_worker(args: argparse.Namespace) -> dict:
    runtime_root = Path(args.runtime_root).resolve()
    source_path = Path(args.source_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if Path.cwd().resolve() != runtime_root:
        raise RuntimeError("LAM worker must run from the pinned runtime root")
    if Path(sys.executable).resolve() != (
        runtime_root / ".glut" / "python.exe"
    ).resolve():
        raise RuntimeError("LAM worker must use the pinned portable Python")
    if not args.expected_source_sha256:
        raise RuntimeError("LAM worker requires a pinned source hash")
    if _sha256(source_path) != args.expected_source_sha256:
        raise RuntimeError("LAM worker source hash changed")
    preflight_path = Path(args.preflight_path).resolve()
    if preflight_path.parent != output_dir or not preflight_path.is_file():
        raise RuntimeError("LAM worker preflight path is outside the output")
    if not args.expected_preflight_sha256:
        raise RuntimeError("LAM worker requires a pinned preflight hash")
    if _sha256(preflight_path) != args.expected_preflight_sha256:
        raise RuntimeError("LAM worker preflight manifest hash changed")
    host_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest_runtime_root = host_preflight.get("runtime_root")
    if (
        not host_preflight.get("runnable")
        or not manifest_runtime_root
        or Path(manifest_runtime_root).resolve() != runtime_root
    ):
        raise RuntimeError("LAM worker received an invalid preflight manifest")
    worker_preflight = lam_runtime_preflight(runtime_root)
    if not worker_preflight["runnable"]:
        failed = [
            name
            for name, passed in worker_preflight["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "LAM worker-side preflight failed: " + ", ".join(failed)
        )
    if _sha256(runtime_root / LAM_MODEL_RELATIVE_PATH) != LAM_MODEL_SHA256:
        raise RuntimeError("LAM worker model hash changed after preflight")
    asset_records = _file_records(runtime_root, LAM_ASSET_SPECS)
    if not all(record["pinned"] for record in asset_records.values()):
        raise RuntimeError("LAM worker asset hash changed after preflight")
    repo_root = Path(__file__).resolve().parents[2]
    excluded_paths = {str(Path(__file__).resolve().parent), str(repo_root)}
    sys.path[:] = [path for path in sys.path if path not in excluded_paths]
    while str(runtime_root) in sys.path:
        sys.path.remove(str(runtime_root))
    sys.path.insert(0, str(runtime_root))

    from omegaconf import OmegaConf
    from PIL import Image
    from safetensors.torch import load_file
    import torch
    import torchvision

    torch_compile = _install_eager_torch_compile(torch)
    import diff_gaussian_rasterization as diff_gaussian

    approved_diff_gaussian_root = _toolchain_layout(runtime_root)[
        "diff_gaussian_source"
    ].resolve()
    diff_gaussian_module_path = Path(diff_gaussian.__file__).resolve()
    diff_gaussian_extension_path = Path(diff_gaussian._C.__file__).resolve()
    if not diff_gaussian_module_path.is_relative_to(approved_diff_gaussian_root):
        raise RuntimeError("LAM loaded an unapproved Gaussian rasterizer package")
    if (
        not diff_gaussian_extension_path.is_relative_to(
            approved_diff_gaussian_root
        )
        or diff_gaussian_extension_path.stat().st_size
        != DIFF_GAUSSIAN_EXTENSION_SIZE_BYTES
        or _sha256(diff_gaussian_extension_path)
        != DIFF_GAUSSIAN_EXTENSION_SHA256
    ):
        raise RuntimeError("LAM loaded an unapproved Gaussian rasterizer binary")
    from lam.models import ModelLAM
    from lam.runners.infer import head_utils
    from tools.flame_tracking_single_image import (
        FlameTrackingSingleImage,
        expand_bbox,
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stages: dict[str, float] = {}

    tracker_started = time.perf_counter()
    tracker = FlameTrackingSingleImage(
        output_dir=str(output_dir / "tracking"),
        alignment_model_path=str(
            runtime_root
            / "model_zoo/flame_tracking_models/68_keypoints_model.pkl"
        ),
        vgghead_model_path=str(
            runtime_root
            / "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd"
        ),
        human_matting_path=str(
            runtime_root
            / "model_zoo/flame_tracking_models/matting/stylematte_synth.pt"
        ),
        facebox_model_path=str(
            runtime_root
            / "model_zoo/flame_tracking_models/FaceBoxesV2.pth"
        ),
        detect_iris_landmarks=False,
        args=argparse.Namespace(
            output_dir=str(output_dir / "tracking"),
            config_name="alignment",
            blender_path=None,
        ),
    )
    frame = torchvision.io.read_image(str(source_path))[:3, ...]
    _, detected_bbox, detector_record = tracker.vgghead_encoder(frame, 0)
    if detected_bbox is None:
        raise RuntimeError("LAM tracker found no head")
    tracker_crop = (
        expand_bbox(detected_bbox, scale=1.65)
        .long()
        .detach()
        .cpu()
        .tolist()
    )
    tracker_crop = [int(value) for value in tracker_crop]
    tracker_debug = {
        "detector_record": str(detector_record),
        "tracker_crop_xyxy": tracker_crop,
    }
    selection_mask_path = (
        Path(args.selection_mask_path).resolve()
        if args.selection_mask_path
        else None
    )
    if selection_mask_path is not None:
        if not args.expected_selection_mask_sha256:
            raise RuntimeError("LAM worker requires a pinned selection hash")
        if _sha256(selection_mask_path) != args.expected_selection_mask_sha256:
            raise RuntimeError("LAM worker selection mask hash changed")
        selection = torchvision.io.read_image(str(selection_mask_path))[:1]
        if tuple(selection.shape[-2:]) != tuple(frame.shape[-2:]):
            raise RuntimeError("LAM selection mask dimensions differ from source")
        selection = torchvision.transforms.functional.crop(
            selection,
            top=tracker_crop[1],
            left=tracker_crop[0],
            height=tracker_crop[3] - tracker_crop[1],
            width=tracker_crop[2] - tracker_crop[0],
        )
        selection = torchvision.transforms.functional.resize(
            selection,
            (LAM_TRACKER_SIZE, LAM_TRACKER_SIZE),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ).float() / 255.0
        selection = selection[0].clamp(0.0, 1.0)
        selection_coverage = float(torch.mean(selection).cpu())
        if not 0.01 <= selection_coverage <= 0.99:
            raise RuntimeError("LAM tracker-crop selection coverage is invalid")

        class _SelectionMaskMatting:
            def __init__(self, mask):
                self.mask = mask

            def __call__(
                self,
                image,
                *,
                return_type,
                background_rgb,
            ):
                if return_type != "matting":
                    raise ValueError("LAM selection matting requires matting mode")
                mask = self.mask.to(image.device, image.dtype)
                composite = (
                    image * mask.unsqueeze(0)
                    + float(background_rgb) * (1.0 - mask.unsqueeze(0))
                )
                return composite, mask

        tracker.matting_engine = _SelectionMaskMatting(selection)
        tracker_debug["matting"] = {
            "method": "pinned-selection-mask-override",
            "path": str(selection_mask_path),
            "sha256": _sha256(selection_mask_path),
            "coverage": selection_coverage,
            "reason": "official StyleMatte blanked the exact small-face crop",
        }
    else:
        tracker_debug["matting"] = {"method": "official-stylematte"}
    _write_json(output_dir / "tracker_debug.json", tracker_debug)
    stages["tracker_load_seconds"] = time.perf_counter() - tracker_started

    stage = time.perf_counter()
    if tracker.preprocess(str(source_path)) != 0:
        raise RuntimeError("LAM tracker preprocessing failed")
    stages["tracker_preprocess_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    if tracker.optimize() != 0:
        raise RuntimeError("LAM tracker optimization failed")
    stages["tracker_optimize_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    export_code, export_dir_text = tracker.export()
    if export_code != 0:
        raise RuntimeError("LAM tracker export failed")
    export_dir = Path(export_dir_text).resolve()
    stages["tracker_export_seconds"] = time.perf_counter() - stage
    tracker_peak_vram = int(torch.cuda.max_memory_allocated())
    del tracker, frame, detected_bbox
    gc.collect()
    torch.cuda.empty_cache()

    exported_image = export_dir / "images" / "00000_00.png"
    exported_mask = export_dir / "fg_masks" / "00000_00.png"
    if not exported_image.is_file() or not exported_mask.is_file():
        raise RuntimeError("LAM tracker export is incomplete")
    with Image.open(exported_image) as loaded:
        export_rgb = np.asarray(loaded.convert("RGB"))
    with Image.open(exported_mask) as loaded:
        export_mask = (np.asarray(loaded.convert("L")) > 180).astype(np.float32)
    if export_rgb.shape[:2] != export_mask.shape:
        raise RuntimeError("LAM exported image and mask shapes differ")
    if int(np.count_nonzero(export_mask)) < 64:
        raise RuntimeError("LAM exported mask is empty")

    stage = time.perf_counter()
    crop_capture: dict[str, int] = {}
    original_center_crop = head_utils.center_crop_according_to_mask

    def _capture_center_crop(image_value, mask_value, aspect, enlarge):
        result = original_center_crop(image_value, mask_value, aspect, enlarge)
        if crop_capture:
            raise RuntimeError("LAM preprocessing performed multiple crops")
        cropped_image, cropped_mask, offset_x, offset_y = result
        if cropped_mask.shape != cropped_image.shape[:2]:
            raise RuntimeError("LAM preprocessor crop mask changed shape")
        crop_capture.update(
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            width=int(cropped_image.shape[1]),
            height=int(cropped_image.shape[0]),
        )
        return result

    head_utils.center_crop_according_to_mask = _capture_center_crop
    try:
        image, _, _, shape_param = head_utils.preprocess_image(
            str(exported_image),
            mask_path=str(exported_mask),
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=LAM_RENDER_SIZE,
            multiply=14,
            need_mask=True,
            get_shape_param=True,
        )
    finally:
        head_utils.center_crop_according_to_mask = original_center_crop
    if not crop_capture:
        raise RuntimeError("LAM preprocessing did not expose its crop affine")
    preprocessor_crop = [
        crop_capture["offset_x"],
        crop_capture["offset_y"],
        crop_capture["width"],
        crop_capture["height"],
    ]
    expected_input_size, _, _ = head_utils.calc_new_tgt_size_by_aspect(
        (crop_capture["height"], crop_capture["width"]),
        1.0,
        LAM_RENDER_SIZE,
        14,
    )
    if tuple(image.shape[-2:]) != tuple(expected_input_size):
        raise RuntimeError("LAM preprocessor tensor size changed")
    motion = head_utils.prepare_motion_seqs(
        str(export_dir / "flame_param"),
        None,
        save_root=str(output_dir / "motion"),
        fps=30,
        bg_color=1.0,
        aspect_standard=1.0,
        enlarge_ratio=[1.0, 1.0],
        render_image_res=LAM_RENDER_SIZE,
        multiply=16,
        need_mask=False,
        vis_motion=False,
        shape_param=shape_param,
        test_sample=False,
        cross_id=False,
        src_driven=[source_path.stem, source_path.stem],
    )
    motion["flame_params"]["betas"] = shape_param.unsqueeze(0)
    stages["input_prepare_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    cfg = OmegaConf.load(runtime_root / "configs/inference/lam-20k-8gpu.yaml")
    model = ModelLAM(**cfg.model)
    checkpoint = load_file(
        str(runtime_root / LAM_MODEL_RELATIVE_PATH),
        device="cpu",
    )
    state_dict = model.state_dict()
    unexpected = []
    mismatched = []
    for name, value in checkpoint.items():
        if name not in state_dict:
            unexpected.append(name)
        elif state_dict[name].shape != value.shape:
            mismatched.append(name)
        else:
            state_dict[name].copy_(value)
    missing = sorted(set(state_dict) - set(checkpoint))
    allowed_missing = ["renderer.flame_model.J_regressor_up"]
    if unexpected or mismatched or missing != allowed_missing:
        raise RuntimeError(
            "LAM checkpoint contract changed: "
            f"unexpected={len(unexpected)}, mismatched={len(mismatched)}, "
            f"missing={missing}"
        )
    checkpoint_coverage = {
        "checkpoint_tensors": len(checkpoint),
        "model_state_tensors": len(state_dict),
        "matched_tensors": len(checkpoint) - len(unexpected) - len(mismatched),
        "allowed_derived_missing": allowed_missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
    del checkpoint
    model.to("cuda").eval()
    stages["model_load_seconds"] = time.perf_counter() - stage

    for tensor_name in ("render_c2ws", "render_intrs", "render_bg_colors"):
        tensor = motion[tensor_name]
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"LAM emitted invalid {tensor_name}")
    for parameter_name, tensor in motion["flame_params"].items():
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"LAM emitted invalid FLAME {parameter_name}")

    def _tensor_stats(value) -> dict:
        flattened = value.detach().float().reshape(-1)
        finite = torch.isfinite(flattened)
        finite_values = flattened[finite]
        if finite_values.numel() == 0:
            return {
                "shape": list(value.shape),
                "finite_ratio": 0.0,
                "minimum": None,
                "p01": None,
                "median": None,
                "p99": None,
                "maximum": None,
            }
        quantiles = torch.quantile(
            finite_values,
            torch.tensor(
                [0.0, 0.01, 0.5, 0.99, 1.0],
                device=finite_values.device,
            ),
        ).cpu().tolist()
        return {
            "shape": list(value.shape),
            "finite_ratio": float(torch.mean(finite.float()).cpu()),
            "minimum": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "maximum": float(quantiles[4]),
        }

    original_render_single_view = model.renderer.forward_single_view

    def _audited_render_single_view(gs, camera, background_color):
        xyz_h = torch.cat(
            (
                gs.xyz.float(),
                torch.ones_like(gs.xyz[:, :1], dtype=torch.float32),
            ),
            dim=1,
        )
        camera_xyz = xyz_h @ camera.world_view_transform.float()
        camera_z = camera_xyz[:, 2]
        point_scale = torch.max(gs.scaling.float(), dim=1).values
        focal = torch.max(
            camera.intrinsic[0, 0],
            camera.intrinsic[1, 1],
        ).float()
        positive_z = camera_z > float(camera.znear)
        estimated_radius = torch.full_like(camera_z, float("inf"))
        estimated_radius[positive_z] = (
            point_scale[positive_z] * focal / camera_z[positive_z]
        )
        finite_radius = estimated_radius[torch.isfinite(estimated_radius)]
        maximum_radius = (
            float(torch.max(finite_radius).cpu())
            if finite_radius.numel()
            else None
        )
        audit = {
            "xyz": _tensor_stats(gs.xyz),
            "scaling": _tensor_stats(gs.scaling),
            "opacity": _tensor_stats(gs.opacity),
            "rotation": _tensor_stats(gs.rotation),
            "camera_xyz": _tensor_stats(camera_xyz[:, :3]),
            "camera_z": _tensor_stats(camera_z),
            "points_behind_near_plane": int(
                torch.count_nonzero(~positive_z).cpu()
            ),
            "estimated_projected_radius_pixels": _tensor_stats(
                estimated_radius
            ),
            "maximum_finite_estimated_radius_pixels": maximum_radius,
            "fov_x_radians": float(camera.FoVx),
            "fov_y_radians": float(camera.FoVy),
            "height": int(camera.height),
            "width": int(camera.width),
            "intrinsic": camera.intrinsic.detach().float().cpu().tolist(),
            "world_view_transform": camera.world_view_transform.detach()
            .float()
            .cpu()
            .tolist(),
        }
        _write_json(output_dir / "pre_render_audit.json", audit)
        if audit["points_behind_near_plane"]:
            raise RuntimeError("LAM Gaussians cross the camera near plane")
        if maximum_radius is None or maximum_radius > 4096.0:
            raise RuntimeError("LAM projected Gaussian radius is unsafe")
        return original_render_single_view(gs, camera, background_color)

    model.renderer.forward_single_view = _audited_render_single_view

    torch.cuda.reset_peak_memory_stats()
    stage = time.perf_counter()
    with torch.no_grad():
        result = model.infer_single_view(
            image.unsqueeze(0).to("cuda", torch.float32),
            None,
            None,
            render_c2ws=motion["render_c2ws"].to("cuda"),
            render_intrs=motion["render_intrs"].to("cuda"),
            render_bg_colors=motion["render_bg_colors"].to("cuda"),
            flame_params={
                name: value.to("cuda")
                for name, value in motion["flame_params"].items()
            },
        )
    torch.cuda.synchronize()
    stages["inference_seconds"] = time.perf_counter() - stage
    inference_peak_vram = int(torch.cuda.max_memory_allocated())
    render_depth_weighted = (
        result["comp_depth"][0, ..., 0].detach().float().cpu().numpy()
    )
    render_mask = result["comp_mask"][0, ..., 0].detach().float().cpu().numpy()
    render_depth = normalize_lam_comp_depth(
        render_depth_weighted,
        render_mask,
    )
    render_rgb = result["comp_rgb"][0].detach().float().cpu().numpy()
    if render_depth.shape != (LAM_RENDER_SIZE, LAM_RENDER_SIZE):
        raise RuntimeError(f"LAM emitted unexpected depth: {render_depth.shape}")
    if render_rgb.shape != (LAM_RENDER_SIZE, LAM_RENDER_SIZE, 3):
        raise RuntimeError(f"LAM emitted unexpected RGB: {render_rgb.shape}")
    if not np.all(np.isfinite(render_rgb)):
        raise RuntimeError("LAM emitted non-finite RGB")

    with Image.open(source_path) as loaded:
        source_width, source_height = loaded.size
    source_depth, source_mask = warp_lam_depth_to_source(
        render_depth_weighted,
        render_mask,
        (source_height, source_width),
        tracker_crop,
        preprocessor_crop,
    )
    source_selection = None
    if selection_mask_path is not None:
        with Image.open(selection_mask_path) as loaded:
            source_selection = np.asarray(
                loaded.convert("L"),
                dtype=np.float32,
            )
        source_selection /= 255.0
    depth_validation = validate_lam_camera_depth(
        render_depth,
        render_mask,
        source_depth,
        source_mask,
        selection_mask=source_selection,
    )
    arrays = {
        "render_depth_weighted": (
            output_dir / "provider_depth_render_weighted.npy"
        ),
        "render_depth": output_dir / "provider_depth_render.npy",
        "render_mask": output_dir / "provider_mask_render.npy",
        "source_depth": output_dir / "provider_depth_source.npy",
        "source_mask": output_dir / "provider_mask_source.npy",
    }
    np.save(
        arrays["render_depth_weighted"],
        render_depth_weighted.astype(np.float32),
    )
    np.save(arrays["render_depth"], render_depth)
    np.save(arrays["render_mask"], render_mask.astype(np.float32))
    np.save(arrays["source_depth"], source_depth)
    np.save(arrays["source_mask"], source_mask)
    _depth_preview(output_dir / "provider_depth_render.png", render_depth)
    _depth_preview(output_dir / "provider_depth_source.png", source_depth)
    Image.fromarray(
        np.rint(np.clip(render_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    ).save(output_dir / "provider_rgb_render.png")

    finite = np.isfinite(source_depth)
    source_depth_stats = {
        "finite_pixels": int(np.count_nonzero(finite)),
        "coverage_ratio": float(np.mean(finite)),
        "minimum": float(np.min(source_depth[finite])) if np.any(finite) else None,
        "median": float(np.median(source_depth[finite])) if np.any(finite) else None,
        "maximum": float(np.max(source_depth[finite])) if np.any(finite) else None,
        "validation": depth_validation,
    }
    metadata = {
        "schema_version": 2,
        "provider": "lam-20k-camera-depth",
        "privacy": "CC0 MakeHuman synthetic identity and procedural scene only",
        "source_geometry": "evaluation-only",
        "production_changed": False,
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "shape": [source_height, source_width, 3],
        },
        "model": {
            "id": LAM_MODEL_ID,
            "revision": LAM_MODEL_REVISION,
            "size_bytes": LAM_MODEL_SIZE_BYTES,
            "sha256": _sha256(runtime_root / LAM_MODEL_RELATIVE_PATH),
            "checkpoint_coverage": checkpoint_coverage,
        },
        "license": {
            "source_code": "Apache-2.0",
            "primary_model_weights": "CC BY-NC 4.0",
            "auxiliary_assets": "mixed or separately unestablished terms",
            "all_asset_terms_established": False,
            "auxiliary_asset_manifest": LAM_ASSET_PROVENANCE,
            "production_eligible": False,
            "research_only": True,
        },
        "detector": {
            "raw_record": str(detector_record),
            "tracker_crop_xyxy": tracker_crop,
            "expand_scale": 1.65,
            "matting": tracker_debug["matting"],
        },
        "camera": {
            "native_depth_field": "alpha-weighted comp_depth",
            "depth_normalization": (
                "comp_depth / comp_mask where comp_mask >= 0.01"
            ),
            "source_depth_resampling": (
                "bilinear comp_depth and comp_mask warp, then division where "
                "warped comp_mask >= 0.01"
            ),
            "depth_semantics": (
                "alpha-normalized expected positive floating-camera-depth"
            ),
            "near_is_smaller": True,
            "render_size": LAM_RENDER_SIZE,
            "tracker_size": LAM_TRACKER_SIZE,
            "preprocessor_crop_xywh": preprocessor_crop,
            "preprocessor_input_hw": [
                int(image.shape[-2]),
                int(image.shape[-1]),
            ],
            "render_c2w": motion["render_c2ws"][0, 0].tolist(),
            "render_intrinsics": motion["render_intrs"][0, 0].tolist(),
            "inverse_warp": "half-pixel tracker and render resize",
        },
        "depth": source_depth_stats,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "torch_compile": torch_compile,
            "toolchain": worker_preflight["toolchain"],
            "tracker_peak_vram_bytes": tracker_peak_vram,
            "inference_peak_vram_bytes": inference_peak_vram,
            "stages_seconds": stages,
            "total_seconds": float(time.perf_counter() - started),
        },
        "preflight_file": {
            "file": preflight_path.name,
            "sha256": args.expected_preflight_sha256,
        },
        "artifacts": {
            name: {
                "file": path.name,
                "sha256": _sha256(path),
            }
            for name, path in arrays.items()
        },
    }
    _write_json(output_dir / "provider_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-path")
    parser.add_argument("--selection-mask-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-selection-mask-sha256")
    parser.add_argument("--preflight-path")
    parser.add_argument("--expected-preflight-sha256")
    parser.add_argument("--runtime-worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = lam_runtime_preflight(args.runtime_root)
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["runnable"]:
            raise SystemExit(2)
        return
    if args.runtime_worker:
        if not all(
            (
                args.source_path,
                args.output_dir,
                args.expected_source_sha256,
                args.preflight_path,
                args.expected_preflight_sha256,
            )
        ):
            parser.error(
                "runtime worker requires source, output, hashes, and preflight"
            )
        result = _runtime_worker(args)
        print(json.dumps(result["runtime"], indent=2, allow_nan=False))
        return
    if not args.source_path or not args.output_dir:
        parser.error("provider run requires source and output")
    result = run_lam_camera_depth(
        args.runtime_root,
        args.source_path,
        args.output_dir,
        selection_mask_path=args.selection_mask_path,
        expected_source_sha256=args.expected_source_sha256,
        expected_selection_mask_sha256=args.expected_selection_mask_sha256,
    )
    print(json.dumps(result["runtime"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
