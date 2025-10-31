#!/usr/bin/env python3
"""
Comprehensive API Test Script for MuseCoco Text-to-MIDI Service

Tests all endpoints:
1. Initial generation (submit-text)
2. Modification (generate-with-context)
3. Condition-Expansion (context-continue-generate)
4. Expansion (continue-generate)
"""

import os
import time
import requests
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8001"
OUTPUT_DIR = Path("./test_outputs")
POLL_INTERVAL = 5  # seconds
MAX_RETRIES = 120  # 10 minutes max wait

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

# ============================================================================
# Helper Functions
# ============================================================================

def print_header(text):
    """Print a colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_info(label, value):
    """Print labeled information"""
    print(f"{Colors.CYAN}{label}:{Colors.END} {value}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_progress(message):
    """Print progress message"""
    print(f"{Colors.YELLOW}⟳ {message}{Colors.END}")

def submit_text(text):
    """Submit text for MIDI generation"""
    print_progress(f"Submitting text: '{text}'")

    response = requests.post(
        f"{API_BASE_URL}/submit-text",
        json={"text": text}
    )

    if response.status_code == 202:
        data = response.json()
        job_id = data["jobId"]
        print_success(f"Job submitted successfully!")
        print_info("Job ID", job_id)
        return job_id
    else:
        print_error(f"Failed to submit: {response.status_code}")
        print(response.text)
        return None

def check_status(job_id):
    """Check job status"""
    response = requests.get(f"{API_BASE_URL}/check-status/{job_id}")

    if response.status_code == 200:
        data = response.json()
        return data["status"]
    else:
        print_error(f"Failed to check status: {response.status_code}")
        return None

def wait_for_completion(job_id, stage_name):
    """Wait for job to complete with progress updates"""
    print_progress(f"Waiting for {stage_name} to complete...")

    retries = 0
    while retries < MAX_RETRIES:
        status = check_status(job_id)

        if status == "completed":
            print_success(f"{stage_name} completed!")
            return True
        elif status == "failed":
            print_error(f"{stage_name} failed!")
            return False
        elif status in ["submitted", "processing"]:
            print(f"  Status: {status} (waiting {POLL_INTERVAL}s...)", end='\r')
            time.sleep(POLL_INTERVAL)
            retries += 1
        else:
            print_error(f"Unknown status: {status}")
            return False

    print_error(f"Timeout waiting for {stage_name}")
    return False

def get_result(job_id):
    """Get job result metadata"""
    response = requests.get(f"{API_BASE_URL}/get-result/{job_id}")

    if response.status_code == 200:
        data = response.json()
        return data["metaData"]
    else:
        print_error(f"Failed to get result: {response.status_code}")
        print(response.text)
        return None

def download_midi(job_id, filename):
    """Download MIDI file"""
    print_progress(f"Downloading MIDI: {filename}")

    response = requests.get(f"{API_BASE_URL}/download-midi/{job_id}")

    if response.status_code == 200:
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print_success(f"Downloaded: {filepath}")
        return str(filepath)
    else:
        print_error(f"Failed to download: {response.status_code}")
        return None

def report_metadata(metadata, title):
    """Print metadata report"""
    print(f"\n{Colors.BLUE}{'─'*80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'─'*80}{Colors.END}")

    for key, value in metadata.items():
        print_info(f"  {key}", value)

    print(f"{Colors.BLUE}{'─'*80}{Colors.END}\n")

def generate_with_context(original_job_id, new_text):
    """Test generate-with-context endpoint"""
    print_progress(f"Submitting context generation: '{new_text}'")

    response = requests.post(
        f"{API_BASE_URL}/generate-with-context/{original_job_id}",
        json={"text": new_text}
    )

    if response.status_code == 202:
        data = response.json()
        job_id = data["jobId"]
        print_success(f"Context generation submitted!")
        print_info("New Job ID", job_id)
        return job_id
    else:
        print_error(f"Failed to submit: {response.status_code}")
        print(response.text)
        return None

def context_continue_generate(original_job_id, new_text):
    """Test context-continue-generate endpoint"""
    print_progress(f"Submitting context-continue generation: '{new_text}'")

    response = requests.post(
        f"{API_BASE_URL}/context-continue-generate/{original_job_id}",
        json={"text": new_text}
    )

    if response.status_code == 202:
        data = response.json()
        job_id = data["jobId"]
        print_success(f"Context-continue generation submitted!")
        print_info("New Job ID", job_id)
        return job_id
    else:
        print_error(f"Failed to submit: {response.status_code}")
        print(response.text)
        return None

def continue_generate(original_job_id):
    """Test continue-generate endpoint"""
    print_progress(f"Submitting continuation generation")

    response = requests.post(f"{API_BASE_URL}/continue-generate/{original_job_id}")

    if response.status_code == 202:
        data = response.json()
        job_id = data["jobId"]
        print_success(f"Continuation submitted!")
        print_info("New Job ID", job_id)
        return job_id
    else:
        print_error(f"Failed to submit: {response.status_code}")
        print(response.text)
        return None

# ============================================================================
# Main Test Flow
# ============================================================================

def main():
    """Run comprehensive API tests"""

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print_header("MuseCoco Text-to-MIDI API Test Suite")
    print_info("API Base URL", API_BASE_URL)
    print_info("Output Directory", OUTPUT_DIR)
    print_info("Test Started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ========================================================================
    # STAGE 1: Original Generation
    # ========================================================================

    print_header("STAGE 1: Original Generation (submit-text)")

    original_text = "A peaceful piano melody"
    original_job_id = submit_text(original_text)

    if not original_job_id:
        print_error("Failed to submit original text. Aborting.")
        return

    if not wait_for_completion(original_job_id, "Original generation"):
        print_error("Original generation failed. Aborting.")
        return

    original_metadata = get_result(original_job_id)
    if original_metadata:
        report_metadata(original_metadata, "📄 Original Generation Metadata")

    download_midi(original_job_id, f"01_original_{original_job_id[:8]}.mid")

    # ========================================================================
    # STAGE 2: Modification (Generate-With-Context)
    # ========================================================================

    print_header("STAGE 2: Modification (generate-with-context)")

    modification_text = "with energetic jazz harmonies"
    modification_job_id = generate_with_context(original_job_id, modification_text)

    if not modification_job_id:
        print_error("Failed to submit modification. Continuing with other tests...")
    else:
        if wait_for_completion(modification_job_id, "Modification"):
            modification_metadata = get_result(modification_job_id)
            if modification_metadata:
                report_metadata(modification_metadata, "📄 Modification Metadata")
            download_midi(modification_job_id, f"02_modification_{modification_job_id[:8]}.mid")

    # ========================================================================
    # STAGE 3: Condition-Expansion (Context-Continue-Generate)
    # ========================================================================

    print_header("STAGE 3: Condition-Expansion (context-continue-generate)")

    condition_text = "with powerful orchestral crescendo"
    condition_job_id = context_continue_generate(original_job_id, condition_text)

    if not condition_job_id:
        print_error("Failed to submit condition-expansion. Continuing with other tests...")
    else:
        if wait_for_completion(condition_job_id, "Condition-Expansion"):
            condition_metadata = get_result(condition_job_id)
            if condition_metadata:
                report_metadata(condition_metadata, "📄 Condition-Expansion Metadata")
            download_midi(condition_job_id, f"03_condition_expansion_{condition_job_id[:8]}.mid")

    # ========================================================================
    # STAGE 4: Expansion (Continue-Generate)
    # ========================================================================

    print_header("STAGE 4: Expansion (continue-generate)")

    expansion_job_id = continue_generate(original_job_id)

    if not expansion_job_id:
        print_error("Failed to submit expansion. Continuing with other tests...")
    else:
        if wait_for_completion(expansion_job_id, "Expansion"):
            expansion_metadata = get_result(expansion_job_id)
            if expansion_metadata:
                report_metadata(expansion_metadata, "📄 Expansion Metadata")
            download_midi(expansion_job_id, f"04_expansion_{expansion_job_id[:8]}.mid")

    # ========================================================================
    # Summary
    # ========================================================================

    print_header("Test Summary")
    print_info("Original Job ID", original_job_id)
    print_info("Modification Job ID", modification_job_id if modification_job_id else "N/A")
    print_info("Condition-Expansion Job ID", condition_job_id if condition_job_id else "N/A")
    print_info("Expansion Job ID", expansion_job_id if expansion_job_id else "N/A")
    print_info("Output Directory", OUTPUT_DIR)
    print_info("Test Completed", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print_success("\n✓ All tests completed! Check the test_outputs/ directory for MIDI files.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nTest interrupted by user.")
    except Exception as e:
        print_error(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
