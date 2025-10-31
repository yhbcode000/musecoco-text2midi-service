"""
MuseCoco Text-to-MIDI FastAPI Server

A RESTful API service for converting text descriptions into MIDI files using MuseCoco.
Provides async job processing with status tracking and MIDI file downloads.
"""

import shutil
import sys
import argparse
import threading
import uuid
import os
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from musecoco_text2midi_service.control import Text2Midi
from musecoco_text2midi_service.dao import load_config_from_file


# ============================================================================
# Utility Functions
# ============================================================================

def delete_folder_contents(folder_path: str) -> None:
    """
    Deletes all contents of the specified folder and recreates it.

    Args:
        folder_path: Path to the folder to clean
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


# ============================================================================
# Pydantic Models
# ============================================================================

class JobStatusEnum(str, Enum):
    """Enumeration of possible job statuses"""
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TextInput(BaseModel):
    """Input model for text-to-MIDI conversion"""
    text: str = Field(
        ...,
        description="Text description to convert into MIDI music",
        example="A cheerful piano melody in C major with a moderate tempo"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status queries"""
    job_id: str = Field(..., alias="jobId", description="Unique identifier for the job")
    status: JobStatusEnum = Field(..., description="Current status of the job")

    class Config:
        populate_by_name = True


class JobSubmitResponse(JobStatusResponse):
    """Response model for job submission"""
    message: str = Field(
        ...,
        description="Human-readable message about the job submission",
        example="Job submitted successfully. Use the job_id to check status."
    )


class MidiMetadata(BaseModel):
    """Metadata about the generated MIDI file"""
    # Add specific fields based on what your service returns
    # This is a flexible model that accepts any metadata
    class Config:
        extra = "allow"


class JobResultResponse(BaseModel):
    """Response model for completed job results"""
    job_id: str = Field(..., alias="jobId", description="Unique identifier for the job")
    status: JobStatusEnum = Field(..., description="Current status of the job")
    meta_data: Dict[str, Any] = Field(
        ...,
        alias="metaData",
        description="Metadata about the generated MIDI file"
    )

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    job_id: Optional[str] = Field(None, alias="jobId", description="Job ID if applicable")

    class Config:
        populate_by_name = True


# ============================================================================
# FastAPI Application Setup
# ============================================================================

# Initialize temporary storage
delete_folder_contents("storage/tmp")

# Create FastAPI app with metadata
app = FastAPI(
    title="MuseCoco Text-to-MIDI API",
    description="""
    🎵 **MuseCoco Text-to-MIDI Conversion Service**

    This API allows you to convert textual descriptions into MIDI music files using the MuseCoco model.

    ## Features

    * **Async Processing**: Submit jobs and check status asynchronously
    * **Job Tracking**: Track the progress of your MIDI generation jobs
    * **Metadata**: Receive detailed metadata about generated MIDI files
    * **File Download**: Download generated MIDI files directly

    ## Workflow

    1. **Submit Text** (`POST /submit-text`): Submit your text description
    2. **Check Status** (`GET /check-status/{job_id}`): Monitor job progress
    3. **Get Result** (`GET /get-result/{job_id}`): Retrieve metadata when complete
    4. **Download MIDI** (`GET /download-midi/{job_id}`): Download the MIDI file

    ## Example Usage

    ```python
    import requests

    # Submit job
    response = requests.post("http://localhost:8001/submit-text",
                            json={"text": "A peaceful piano melody"})
    job_id = response.json()["jobId"]

    # Check status
    status = requests.get(f"http://localhost:8001/check-status/{job_id}")

    # Download MIDI when ready
    midi_file = requests.get(f"http://localhost:8001/download-midi/{job_id}")
    ```
    """,
    version="2.0.0",
    contact={
        "name": "yhbcode000",
        "email": "hobart.yang@qq.com",
        "url": "https://github.com/yhbcode000/musecoco-text2midi-service"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Load configuration
config_path = "storage/config/main_config.yaml"
config = load_config_from_file(config_path)

# Initialize Text2Midi service
text2midi = Text2Midi(config)

# Job store (in-memory storage)
job_store: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Background Task Functions
# ============================================================================

def generate_midi(job_id: str, input_text: str) -> None:
    """
    Background task to generate MIDI from text.

    Args:
        job_id: Unique job identifier
        input_text: Text description to convert to MIDI
    """
    try:
        job_store[job_id]['status'] = JobStatusEnum.PROCESSING

        midi_data, meta_data = text2midi.text_to_midi(input_text, return_midi=True)
        midi_file_path = meta_data.get('file_path')
        save_root = meta_data.get('save_root')

        # Remove 'file_path' from meta_data before storing it (but keep save_root for continuation)
        if 'file_path' in meta_data:
            del meta_data['file_path']

        job_store[job_id]['status'] = JobStatusEnum.COMPLETED
        job_store[job_id]['result'] = {
            'metaData': meta_data,
            'midiFilePath': midi_file_path,
            'saveRoot': save_root
        }
    except Exception as e:
        job_store[job_id]['status'] = JobStatusEnum.FAILED
        job_store[job_id]['error'] = str(e)


def continue_generate_midi(new_job_id: str, original_save_root: str) -> None:
    """
    Background task to continue MIDI generation from a previous job.

    Args:
        new_job_id: Unique job identifier for the continuation
        original_save_root: Path to the save_root of the original generation
    """
    try:
        job_store[new_job_id]['status'] = JobStatusEnum.PROCESSING

        midi_data, meta_data = text2midi.continue_midi_generation(
            original_save_root=original_save_root,
            return_midi=True
        )
        midi_file_path = meta_data.get('file_path')
        save_root = meta_data.get('save_root')

        # Remove 'file_path' from meta_data before storing it (but keep save_root for continuation)
        if 'file_path' in meta_data:
            del meta_data['file_path']

        job_store[new_job_id]['status'] = JobStatusEnum.COMPLETED
        job_store[new_job_id]['result'] = {
            'metaData': meta_data,
            'midiFilePath': midi_file_path,
            'saveRoot': save_root
        }
    except Exception as e:
        job_store[new_job_id]['status'] = JobStatusEnum.FAILED
        job_store[new_job_id]['error'] = str(e)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/",
    summary="API Information",
    description="Returns basic information about the API and links to documentation",
    tags=["General"]
)
async def root():
    """Get API information and documentation links"""
    return {
        "name": "MuseCoco Text-to-MIDI API",
        "version": "2.0.0",
        "description": "Convert text descriptions into MIDI music files",
        "documentation": "/docs",
        "alternative_docs": "/redoc"
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API service is running and healthy",
    tags=["General"]
)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "musecoco-text2midi",
        "model_loaded": text2midi is not None
    }


@app.post(
    "/submit-text",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Text for MIDI Generation",
    description="""
    Submit a text description to be converted into a MIDI file.

    The job is processed asynchronously in the background. Use the returned `job_id`
    to check the status and retrieve results.

    **Note**: This endpoint clears the generation storage folder before processing.
    """,
    responses={
        202: {
            "description": "Job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "jobId": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "submitted",
                        "message": "Job submitted successfully. Use the job_id to check status."
                    }
                }
            }
        },
        400: {
            "description": "Invalid input",
            "model": ErrorResponse
        }
    },
    tags=["MIDI Generation"]
)
async def submit_text(text_input: TextInput):
    """
    Submit text for MIDI generation.

    Args:
        text_input: Text description to convert

    Returns:
        JobSubmitResponse with job_id and status
    """
    # Clean up generation folder
    delete_folder_contents("storage/generation")

    if not text_input.text or not text_input.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input is required and cannot be empty"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_store[job_id] = {'status': JobStatusEnum.SUBMITTED}

    # Start background thread to process the request without blocking the response
    threading.Thread(
        target=generate_midi,
        args=(job_id, text_input.text),
        daemon=True
    ).start()

    return JobSubmitResponse(
        jobId=job_id,
        status=JobStatusEnum.SUBMITTED,
        message="Job submitted successfully. Use the job_id to check status."
    )


@app.get(
    "/check-status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check Job Status",
    description="""
    Check the current status of a MIDI generation job.

    Possible statuses:
    - `submitted`: Job has been received and queued
    - `processing`: Job is currently being processed
    - `completed`: MIDI generation is complete
    - `failed`: Job failed (check error details with /get-result)
    """,
    responses={
        200: {
            "description": "Job status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "jobId": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "completed"
                    }
                }
            }
        },
        404: {
            "description": "Job ID not found",
            "model": ErrorResponse
        }
    },
    tags=["MIDI Generation"]
)
async def check_status(job_id: str):
    """
    Check the status of a job.

    Args:
        job_id: Unique job identifier

    Returns:
        JobStatusResponse with current status

    Raises:
        HTTPException: If job_id is not found
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found"
        )

    return JobStatusResponse(jobId=job_id, status=job['status'])


@app.get(
    "/get-result/{job_id}",
    response_model=JobResultResponse,
    summary="Get Job Result",
    description="""
    Retrieve the result and metadata of a completed MIDI generation job.

    This endpoint returns detailed metadata about the generated MIDI file,
    but not the file itself. Use `/download-midi/{job_id}` to download the actual file.
    """,
    responses={
        200: {
            "description": "Job result retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "jobId": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "completed",
                        "metaData": {
                            "duration": 120,
                            "tempo": 120,
                            "key": "C major"
                        }
                    }
                }
            }
        },
        404: {
            "description": "Job ID not found",
            "model": ErrorResponse
        },
        400: {
            "description": "Job not completed or failed",
            "model": ErrorResponse
        }
    },
    tags=["MIDI Generation"]
)
async def get_result(job_id: str):
    """
    Get the result of a completed job.

    Args:
        job_id: Unique job identifier

    Returns:
        JobResultResponse with metadata

    Raises:
        HTTPException: If job_id is not found or job is not completed
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found"
        )

    if job['status'] == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )

    if job['status'] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIDI generation is not complete yet. Current status: {job['status']}"
        )

    return JobResultResponse(
        jobId=job_id,
        status=job['status'],
        metaData=job['result']['metaData']
    )


@app.get(
    "/download-midi/{job_id}",
    response_class=FileResponse,
    summary="Download MIDI File",
    description="""
    Download the generated MIDI file for a completed job.

    The file is returned as an attachment with the appropriate MIDI content type.
    """,
    responses={
        200: {
            "description": "MIDI file download",
            "content": {"audio/midi": {}}
        },
        404: {
            "description": "Job ID or MIDI file not found",
            "model": ErrorResponse
        },
        400: {
            "description": "Job not completed yet",
            "model": ErrorResponse
        }
    },
    tags=["MIDI Generation"]
)
async def download_midi(job_id: str):
    """
    Download the generated MIDI file.

    Args:
        job_id: Unique job identifier

    Returns:
        FileResponse with the MIDI file

    Raises:
        HTTPException: If job_id is not found, job is not completed, or file doesn't exist
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found"
        )

    if job['status'] == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )

    if job['status'] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIDI generation is not complete yet. Current status: {job['status']}"
        )

    midi_file_path = job['result']['midiFilePath']
    if not os.path.exists(midi_file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MIDI file not found on server"
        )

    return FileResponse(
        path=midi_file_path,
        media_type="audio/midi",
        filename=f"generated_{job_id}.mid"
    )


@app.post(
    "/continue-generate/{job_id}",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Continue MIDI Generation",
    description="""
    Continue MIDI generation from a previously completed job.

    This endpoint takes a completed job and continues generation using the REMI tokens
    from the original job as a prefix. The generation length is doubled (2x the original
    REMI token count).

    The continued MIDI will be saved as a new file (continuation only, not merged with original).

    **Workflow:**
    1. Provide the job_id of a completed generation job
    2. System loads REMI tokens and attributes from the original job
    3. Calculates new max_len as 2x the original token count
    4. Generates continuation using MuseCoco
    5. Returns a new job_id to track the continuation

    Use the new job_id with `/check-status`, `/get-result`, and `/download-midi` endpoints.
    """,
    responses={
        202: {
            "description": "Continuation job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "jobId": "456e7890-e12b-34d5-b678-901234567890",
                        "status": "submitted",
                        "message": "Continuation job submitted successfully. Use the job_id to check status."
                    }
                }
            }
        },
        404: {
            "description": "Original job ID not found",
            "model": ErrorResponse
        },
        400: {
            "description": "Original job not completed or failed",
            "model": ErrorResponse
        }
    },
    tags=["MIDI Generation"]
)
async def continue_generate(job_id: str):
    """
    Continue MIDI generation from a previous job.

    Args:
        job_id: Job ID of the completed generation to continue from

    Returns:
        JobSubmitResponse with new job_id for the continuation

    Raises:
        HTTPException: If original job_id is not found or not completed
    """
    # Check if original job exists
    original_job = job_store.get(job_id)
    if not original_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found"
        )

    # Check if original job is completed
    if original_job['status'] == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot continue from failed job. Error: {original_job.get('error', 'Unknown error')}"
        )

    if original_job['status'] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Original job is not completed yet. Current status: {original_job['status']}"
        )

    # Get save_root from original job
    original_save_root = original_job['result'].get('saveRoot')
    if not original_save_root:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original job does not have save_root information. Cannot continue generation."
        )

    # Generate new job ID for continuation
    new_job_id = str(uuid.uuid4())
    job_store[new_job_id] = {
        'status': JobStatusEnum.SUBMITTED,
        'original_job_id': job_id
    }

    # Start background thread to process continuation
    threading.Thread(
        target=continue_generate_midi,
        args=(new_job_id, original_save_root),
        daemon=True
    ).start()

    return JobSubmitResponse(
        jobId=new_job_id,
        status=JobStatusEnum.SUBMITTED,
        message="Continuation job submitted successfully. Use the job_id to check status."
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for the FastAPI server.
    Supports command-line arguments for host and port configuration.
    """
    parser = argparse.ArgumentParser(
        description='MuseCoco Text-to-MIDI FastAPI Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind the server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port number to run the server'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        default=False,
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎵 MuseCoco Text-to-MIDI API Server")
    print("=" * 70)
    print(f"Server starting on: {args.host}:{args.port}")
    print(f"API Documentation: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"ReDoc Documentation: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc")
    print("=" * 70)

    import uvicorn
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
