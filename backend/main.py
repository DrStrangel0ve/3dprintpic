import os
import json
from dotenv import load_dotenv
import aiohttp
import asyncio
import re
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging
from tempfile import NamedTemporaryFile
try:
    from .pic_to_3d import (
        MODERN_INPAINT_MODELS,
        complete_image,
        depth_data_to_3d_model,
        process_image_get_depth_data,
    )
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from pic_to_3d import MODERN_INPAINT_MODELS, complete_image, process_image_get_depth_data, depth_data_to_3d_model
import numpy as np
from PIL import Image

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
DEFAULT_LOCAL_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
CORS_ORIGINS = list(
    dict.fromkeys(
        origin.strip()
        for origin in [*DEFAULT_LOCAL_ORIGINS, *os.getenv("CORS_ORIGINS", "").split(",")]
        if origin.strip()
    )
)

# Updated CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get API key and team ID from environment variables
MASV_API_KEY = os.getenv("MASV_API_KEY")
MASV_TEAM_ID = os.getenv("MASV_TEAM_ID")
RBC_ACCESS_TOKEN = os.getenv("RBC_ACCESS_TOKEN")
RBC_API_BASE_URL = "https://paywithpretendpointsapi.onrender.com/api/v1"
DEFAULT_DEPTH_PROVIDER = os.getenv("DEPTH_PROVIDER", "depth-anything-v2")
DEFAULT_DEPTH_MODEL = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()

DEPTH_MODELS = [
    {
        "id": "depth-anything/Depth-Anything-V2-Small-hf",
        "label": "Depth Anything V2 Small",
        "provider": "depth-anything-v2",
        "recommended": True,
        "notes": "Fast local default for iterative STL generation.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Base-hf",
        "label": "Depth Anything V2 Base",
        "provider": "depth-anything-v2",
        "recommended": False,
        "notes": "Better detail with a larger download and slower first run.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Large-hf",
        "label": "Depth Anything V2 Large",
        "provider": "depth-anything-v2",
        "recommended": False,
        "notes": "Highest quality Depth Anything V2 option; expensive first download.",
    },
]


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_output_file(file_path: str, allowed_suffixes: tuple[str, ...]) -> Path:
    resolved_path = (OUTPUT_DIR / file_path).resolve()
    if not is_relative_to(resolved_path, OUTPUT_DIR):
        raise HTTPException(status_code=400, detail="Invalid output file path")
    if resolved_path.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Unsupported output file type")
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return resolved_path


def output_relative_path(file_path: Path | str) -> str:
    path = Path(file_path).resolve()
    return path.relative_to(OUTPUT_DIR).as_posix()


def get_runtime_info() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        }
    except Exception as exc:
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "unknown",
            "error": str(exc),
        }

@app.post("/process_image")
async def process_image(
    file: UploadFile = File(...),
    depth_provider: str = Form(DEFAULT_DEPTH_PROVIDER),
    depth_model: str | None = Form(None),
    device: str = Form("auto"),
    target_dimension: int = Form(300),
    z_scale: float = Form(50),
    max_xy_size: float | None = Form(None),
    printer_profile: str | None = Form(None),
    printer_max_x_mm: float | None = Form(None),
    printer_max_y_mm: float | None = Form(None),
    printer_max_z_mm: float | None = Form(None),
    printer_clearance_mm: float | None = Form(None),
    relief_polarity: str = Form("raised-print"),
    mesh_resolution_multiplier: float | None = Form(None),
    invert: bool = Form(False),
    sigma: float = Form(0.6),
    relief_gamma: float = Form(0.75),
    detail_boost: float = Form(1.4),
    detail_radius: float = Form(2.0),
    low_percentile: float = Form(1.0),
    high_percentile: float = Form(99.0),
    base_border_px: int = Form(2),
    completion_mode: str = Form("none"),
    completion_provider: str = Form("mirror"),
    completion_prompt: str | None = Form(None),
    completion_model: str | None = Form(None),
    completion_lora_weights: str | None = Form(None),
    completion_lora_scale: float | None = Form(None),
    completion_steps: int = Form(24),
    completion_guidance: float | None = Form(None),
    completion_seed: int | None = Form(None),
    completion_inpaint_max_dimension: int = Form(768),
):
    logger.info(f"Received file: {file.filename}")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_suffix = Path(file.filename or "").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=upload_suffix, dir=job_dir) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the image and get depth data
        selected_model = depth_model or DEFAULT_DEPTH_MODEL
        logger.info(
            "Processing image to get depth data with provider=%s model=%s device=%s",
            depth_provider,
            selected_model,
            device,
        )
        completed_image_path, applied_completion_mode = complete_image(
            temp_file_path,
            output_dir=str(job_dir),
            mode=completion_mode,
            provider=completion_provider,
            prompt=completion_prompt,
            model_name=completion_model,
            lora_weights=completion_lora_weights,
            lora_scale=completion_lora_scale,
            device=device,
            num_inference_steps=completion_steps,
            guidance_scale=completion_guidance,
            seed=completion_seed,
            inpaint_max_dimension=completion_inpaint_max_dimension,
        )
        image_for_depth = completed_image_path

        depth_data_path = process_image_get_depth_data(
            image_for_depth,
            output_dir=str(job_dir),
            provider=depth_provider,
            model_name=selected_model,
            device=device,
        )
        logger.info(f"Depth data saved as: {depth_data_path}")
        
        # Generate 3D model
        logger.info("Generating 3D model...")
        stl_path = job_dir / "model.stl"
        depth_data_to_3d_model(
            depth_data_path,
            output_stl_path=str(stl_path),
            target_dimension=target_dimension,
            z_scale=z_scale,
            max_xy_size=max_xy_size,
            invert=invert,
            sigma=sigma,
            relief_gamma=relief_gamma,
            detail_boost=detail_boost,
            detail_radius=detail_radius,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            base_border_px=base_border_px,
        )
        logger.info(f"3D model saved as: {stl_path}")
        
        # Check if the STL file was actually created
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file was not created at {stl_path}")

        metadata = {
            "job_id": job_id,
            "source_filename": file.filename,
            "depth_provider": depth_provider,
            "depth_model": selected_model,
            "device": device,
            "target_dimension": target_dimension,
            "z_scale": z_scale,
            "max_xy_size": max_xy_size,
            "printer": {
                "profile": printer_profile,
                "max_x_mm": printer_max_x_mm,
                "max_y_mm": printer_max_y_mm,
                "max_z_mm": printer_max_z_mm,
                "clearance_mm": printer_clearance_mm,
            },
            "relief_polarity": relief_polarity,
            "mesh_resolution_multiplier": mesh_resolution_multiplier,
            "invert": invert,
            "sigma": sigma,
            "relief_gamma": relief_gamma,
            "detail_boost": detail_boost,
            "detail_radius": detail_radius,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "base_border_px": base_border_px,
            "completion_mode": completion_mode,
            "completion_provider": completion_provider,
            "completion_model": completion_model,
            "completion_lora_weights": completion_lora_weights,
            "completion_lora_scale": completion_lora_scale,
            "completion_steps": completion_steps,
            "completion_guidance": completion_guidance,
            "completion_seed": completion_seed,
            "completion_inpaint_max_dimension": completion_inpaint_max_dimension,
            "applied_completion_mode": applied_completion_mode,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(job_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        depth_relative_path = output_relative_path(depth_data_path)
        stl_relative_path = output_relative_path(stl_path)
        completed_image_relative_path = (
            output_relative_path(completed_image_path)
            if completed_image_path and applied_completion_mode
            else None
        )
        
        # Return paths to the generated files
        response = {
            **metadata,
            "depth_data": depth_relative_path,
            "depth_data_url": f"/depth_data/{depth_relative_path}",
            "stl_model": stl_relative_path,
            "stl_url": f"/stl_model/{stl_relative_path}",
        }
        if completed_image_relative_path:
            response["completed_image"] = completed_image_relative_path
            response["completed_image_url"] = f"/depth_data/{completed_image_relative_path}"
        return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@app.post("/upload_to_masv")
async def upload_to_masv_endpoint(file_name: str = Form(...)):
    logger.info(f"Received request to upload file to MASV: {file_name}")
    
    file_path = resolve_output_file(file_name, (".stl",))
    
    try:
        # Upload to MASV
        logger.info("Uploading to MASV...")
        masv_package_id = await upload_to_masv(str(file_path), file_path.name)
        logger.info(f"Uploaded to MASV. Package ID: {masv_package_id}")
        
        return {"masv_package_id": masv_package_id}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during MASV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during MASV upload: {str(e)}")

async def upload_to_masv(file_path: str, file_name: str):
    # MASV API endpoints
    create_package_url = "https://api.massive.app/v1/teams/{team_id}/packages"
    add_file_url = "https://api.massive.app/v1/packages/{package_id}/files"
    finalize_package_url = "https://api.massive.app/v1/packages/{package_id}/finalize"

    headers = {
        "X-API-KEY": MASV_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Create a package
            package_data = {
                "name": "3D Print Model",
                "description": "Uploaded 3D print model",
                "recipients": ["jennylive158@gmail.com"]
            }
            async with session.post(create_package_url.format(team_id=MASV_TEAM_ID), json=package_data, headers=headers) as response:
                response.raise_for_status()
                package = await response.json()
                package_id = package["id"]
                package_token = package["access_token"]
                logger.info(f"Package created successfully. ID: {package_id}")

            # Step 2: Add file to the package
            file_data = {
                "kind": "file",
                "name": file_name,
                "path": "",
                "last_modified": datetime.utcnow().isoformat() + "Z"
            }
            headers["X-Package-Token"] = package_token
            async with session.post(add_file_url.format(package_id=package_id), json=file_data, headers=headers) as response:
                response.raise_for_status()
                file_info = await response.json()
                file_id = file_info["file"]["id"]
                logger.info(f"File added to package successfully. File ID: {file_id}")

            # Step 3: Create the file in MASV's cloud storage
            create_blueprint = file_info["create_blueprint"]
            async with session.request(
                create_blueprint["method"],
                create_blueprint["url"],
                headers=create_blueprint.get("headers", {})
            ) as response:
                response.raise_for_status()
                create_response = await response.text()
                logger.info("File created in MASV's cloud storage")
                
                # Parse the XML response to get the UploadId
                upload_id = re.search("<UploadId>(.*?)</UploadId>", create_response).group(1)

            # Step 4: Obtain upload URLs
            chunk_size = 5 * 1024 * 1024  # 5 MB chunks
            file_size = os.path.getsize(file_path)
            chunk_count = math.ceil(file_size / chunk_size)
            
            async with session.post(
                f"{add_file_url.format(package_id=package_id)}/{file_id}",
                params={"start": 0, "count": chunk_count},
                json={"upload_id": upload_id},
                headers=headers
            ) as response:
                response.raise_for_status()
                upload_urls = await response.json()
                logger.info(f"Obtained {len(upload_urls)} upload URLs")

            # Step 5: Upload file chunks
            chunk_extras = []
            with open(file_path, 'rb') as f:
                for i, upload_info in enumerate(upload_urls, start=1):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    logger.info(f"Uploading chunk {i}")
                    async with session.request(
                        upload_info["method"],
                        upload_info["url"],
                        data=chunk
                    ) as response:
                        response.raise_for_status()
                        etag = response.headers.get("ETag")
                        chunk_extras.append({"partNumber": str(i), "etag": etag})
                    logger.info(f"Chunk {i} uploaded successfully")

            # Step 6: Finalize the file
            finalize_data = {
                "chunk_extras": chunk_extras,
                "file_extras": {"upload_id": upload_id},
                "size": file_size,
                "chunk_size": chunk_size
            }
            async with session.post(
                f"{add_file_url.format(package_id=package_id)}/{file_id}/finalize",
                json=finalize_data,
                headers=headers
            ) as response:
                response.raise_for_status()
                logger.info("File upload finalized successfully")

            # Step 7: Finalize the package
            async with session.post(finalize_package_url.format(package_id=package_id), headers=headers) as response:
                response.raise_for_status()
                logger.info("Package finalized successfully")

        return package_id
    except aiohttp.ClientResponseError as e:
        logger.error(f"MASV API error: {e.status}, message='{e.message}', url='{e.request_info.url}'")
        logger.error(f"Request headers: {e.request_info.headers}")
        logger.error(f"Response headers: {e.headers}")
        raise HTTPException(status_code=500, detail=f"MASV API error: {e.status}, {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during MASV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during MASV upload: {str(e)}")

@app.get("/depth_data/{file_path:path}")
async def get_depth_data(file_path: str):
    resolved_path = resolve_output_file(file_path, (".npy", ".png", ".webp"))
    return FileResponse(resolved_path)

@app.get("/stl_model/{file_path:path}")
async def get_stl_model(file_path: str):
    resolved_path = resolve_output_file(file_path, (".stl",))
    return FileResponse(resolved_path)

def sanitize_float(x):
    if np.isnan(x) or np.isinf(x):
        return -12345678  # or another appropriate default value
    return float(x)


def completion_provider_rows() -> list[dict]:
    notes = {
        "sdxl-inpaint": "Public SDXL inpainting checkpoint. Practical middle tier for a 12 GB GPU when run at 384-512 px with fp16 and CPU offload.",
        "dreamshaper-inpaint": "Public SD1.5-style inpainting checkpoint. Smaller fp16 footprint than SDXL and useful as the first learned baseline to beat mirror/biharmonic.",
        "amused-inpaint": "Small public masked-token inpainting model. Useful as a fast non-diffusion learned baseline against mirror/biharmonic.",
        "flux-fill": "Modern rectified-flow inpainting/outpainting model. Large and may require Hugging Face access plus CPU offload.",
        "qwen-image-inpaint": "Qwen Image model through the Diffusers inpaint pipeline. Large; preserves visible pixels after generation.",
        "qwen-image-edit": "Official Qwen Image Edit pipeline prompted to fill the blank half. Large; visible pixels are restored after generation.",
    }
    rows = [
        {
            "id": "mirror",
            "label": "Mirror prior",
            "local": True,
            "gpu_supported": False,
            "notes": "Fast geometric symmetry prior. No diffusion model.",
        },
        {
            "id": "mirror-seam-repair",
            "label": "Mirror seam repair",
            "local": True,
            "gpu_supported": False,
            "notes": "Experimental fast mirror prior with a small classical inpaint repair band along the generated seam. Benchmark before making it the default.",
        }
    ]
    for provider_id, provider in MODERN_INPAINT_MODELS.items():
        rows.append(
            {
                "id": provider_id,
                "label": provider["label"],
                "model": provider["model"],
                "local": True,
                "gpu_supported": True,
                "notes": notes.get(provider_id, "Modern diffusion-based completion provider."),
            }
        )
    return rows

@app.get("/depth_data_downsampled/{file_path:path}")
async def get_depth_data_downsampled(file_path: str):
    resolved_path = resolve_output_file(file_path, (".npy",))
    
    try:
        # Load the depth data
        depth_data = np.load(resolved_path)
        
        # Get original dimensions
        original_height, original_width = depth_data.shape
        
        # Calculate the scaling factor
        max_dimension = max(original_height, original_width)
        scale_factor = max(1, int(max_dimension / 100))
        
        # Downsample the depth data by skipping pixels
        downsampled_data = depth_data[::scale_factor, ::scale_factor]
        
        # Get new dimensions
        new_height, new_width = downsampled_data.shape
        
        # Sanitize and convert to list for JSON serialization
        downsampled_list = [[sanitize_float(x) for x in row] for row in downsampled_data]
        
        return JSONResponse(content={
            "depth_data": downsampled_list,
            "original_dimensions": {
                "height": original_height,
                "width": original_width
            },
            "downsampled_dimensions": {
                "height": new_height,
                "width": new_width
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing depth data: {str(e)}")

async def get_rbc_session():
    return aiohttp.ClientSession(headers={"Authorization": f"Bearer {RBC_ACCESS_TOKEN}"})

async def create_transaction(session, member_id, points):
    transaction_data = {
        "partnerRefId": f"AWARD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "amount": points,
        "note": "Points awarded",
        "type": "PAYMENT"
    }
    try:
        async with session.post(f"{RBC_API_BASE_URL}/loyalty/{member_id}/transactions", json=transaction_data) as response:
            response_text = await response.text()
            logger.info(f"RBC API Response: Status {response.status}, Body: {response_text}")
            if response.status != 200:
                error_detail = f"Error creating transaction. Status: {response.status}, Response: {response_text}"
                logger.error(error_detail)
                raise HTTPException(status_code=response.status, detail=error_detail)
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Network error when calling RBC API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error when calling RBC API: {str(e)}")

@app.post("/award_rbc_points")
async def award_rbc_points(
    member_id: int = Query(..., description="The ID of the member to award points to"),
    points: int = Query(..., description="The number of points to award")
):
    logger.info(f"Awarding {points} points to member {member_id}")
    
    async with await get_rbc_session() as session:
        try:
            # Create a transaction record (which also awards the points)
            transaction = await create_transaction(session, member_id, points)
            
            return {
                "status": "success",
                "message": f"Awarded {points} points to member {member_id}",
                "member_id": member_id,
                "points_awarded": points,
                "transaction": transaction["transaction"]
            }
        except HTTPException as e:
            logger.error(f"Error awarding points: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_depth_provider": DEFAULT_DEPTH_PROVIDER,
        "default_depth_model": DEFAULT_DEPTH_MODEL,
        "output_dir": str(OUTPUT_DIR),
        "runtime": get_runtime_info(),
    }


@app.get("/models")
async def get_models():
    return {
        "default_provider": DEFAULT_DEPTH_PROVIDER,
        "providers": [
            {
                "id": "depth-anything-v2",
                "label": "Depth Anything V2",
                "model": DEFAULT_DEPTH_MODEL,
                "local": True,
                "gpu_supported": True,
                "models": DEPTH_MODELS,
            },
            {
                "id": "sapiens",
                "label": "Sapiens Depth",
                "model": "facebook/sapiens_depth",
                "local": False,
                "gpu_supported": False,
            },
        ],
        "completion_modes": [
            {
                "id": "none",
                "label": "No completion",
                "notes": "Estimate depth only from the input pixels.",
            },
            {
                "id": "mirror-auto",
                "label": "Auto mirror completion",
                "notes": "Mirror the visually richer half across the center before depth estimation.",
            },
            {
                "id": "mirror-left-to-right",
                "label": "Mirror left to right",
                "notes": "Use the left half to synthesize the right half.",
            },
            {
                "id": "mirror-right-to-left",
                "label": "Mirror right to left",
                "notes": "Use the right half to synthesize the left half.",
            },
        ],
        "completion_providers": completion_provider_rows(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
