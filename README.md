# 🎵 MuseCoco Text-to-MIDI Service

The **MuseCoco Text-to-MIDI Service** is a refactored version of the [MuseCoco](https://github.com/microsoft/muzic) repository, designed as a deployable service module. This service adapts to new data, manages the history of its checkpoints, and abstracts away the underlying implementation details to provide a seamless interface for generating MIDI files from textual inputs. Detailed comments are included to facilitate easy navigation and understanding of the codebase.

## 📋 Table of Contents

- [🎵 MuseCoco Text-to-MIDI Service](#-musecoco-text-to-midi-service)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [💻 System Requirements](#-system-requirements)
  - [📂 Directory Structure](#-directory-structure)
  - [⚙️ Installation](#️-installation)
  - [🔧 Configuration](#-configuration)
  - [🚀 Usage](#-usage)
    - [Option 1: Command-Line Demo](#option-1-command-line-demo)
    - [Option 2: FastAPI REST API Server](#option-2-fastapi-rest-api-server)
      - [API Endpoints](#api-endpoints)
      - [Interactive API Documentation](#interactive-api-documentation)
      - [Example API Usage](#example-api-usage)
      - [Python Client Example](#python-client-example)
    - [Option 3: Python Package Import](#option-3-python-package-import)
  - [🧪 Running Tests](#-running-tests)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [Reference:](#reference)
  - [Recommendation](#recommendation)
  - [Notice](#notice)

## ✨ Features

- **Text-to-MIDI Conversion**: Converts textual descriptions into MIDI files using the MuseCoco model.
- **Adaptive Data Handling**: Capable of adapting to new data for more customized MIDI generation.
- **Checkpoint Management**: Manages the history of model checkpoints to ensure reproducibility and flexibility.
- **Abstracted Implementation**: Provides an abstract interface for easy integration while maintaining detailed internal documentation.

## 💻 System Requirements

This service requires a CUDA-compatible NVIDIA GPU. The development and testing was performed with:

- **GPU**: NVIDIA GeForce RTX 4090 (16GB VRAM)
- **NVIDIA Driver**: 570.195.03
- **CUDA Runtime**: 12.8
- **CUDA Toolkit**: 12.6.85

**Minimum Requirements**:
- CUDA-compatible NVIDIA GPU with at least 8GB VRAM
- NVIDIA Driver supporting CUDA 12.x
- Linux operating system (tested on Ubuntu)

## 📂 Directory Structure

The repository is organized as follows:

```plaintext
musecoco-text2midi-service/
├── src/
│   └── musecoco_text2midi_service/
│       ├── control/                   # Controllers for orchestrating service logic
│       │   ├── __init__.py
│       │   ├── _musecoco/             # MuseCoco model implementation
│       │   │   ├── attribute2music_dataprepare/
│       │   │   ├── attribute2music_model/
│       │   │   ├── evaluation/
│       │   │   ├── text2attribute_dataprepare/
│       │   │   ├── text2attribute_model/
│       │   │   ├── __init__.py
│       │   │   └── view.py
│       │   └── _text2midi.py
│       ├── dao/                       # Data Access Objects for configuration management
│       │   ├── __init__.py
│       │   └── _config_manager.py
│       ├── model/                     # Models representing the structure and workflow of MIDI generation
│       │   ├── __init__.py
│       │   └── _config_model.py
│       ├── utils/                     # Utility functions for common tasks
│       │   ├── __init__.py
│       │   └── _watch_dog.py
│       ├── view/                      # Views for API or CLI outputs
│       │   └── __init__.py
│       ├── __init__.py
│       └── main.py                    # CLI entry point for the service
├── storage/
│   ├── checkpoints/                   # Model checkpoints
│   │   └── linear_mask-1billion/
│   │       ├── checkpoint_2_280000.pt
│   │       └── README.md              # Instructions for managing checkpoints
│   ├── config/                        # Configuration files
│   │   ├── main_config.yaml           # Main configuration file
│   │   ├── att_key.json
│   │   └── num_labels.json
│   ├── data/                          # Training/evaluation data
│   ├── generation/                    # Generated output files
│   ├── input/                         # Input files for predictions
│   │   ├── predict_backup.json        # Example input format for predictions
│   │   └── predict.json
│   ├── log/                           # Log files
│   └── tmp/                           # Temporary files and outputs
├── tests/
│   └── test_text2midi.py              # Test modules for various components
├── docs/
│   └── openapi.yaml                   # OpenAPI specification for the REST API
├── .gitignore                         # Specifies files and directories to ignore in version control
├── .python-version                    # Python version specification
├── fastapi_server.py                  # FastAPI REST API server
├── inference.ipynb                    # Jupyter notebook for interactive inference
├── LICENSE                            # License file
├── pyproject.toml                     # Project metadata and dependencies (uv/pip)
├── README.md                          # Project description and instructions
└── uv.lock                            # Locked dependencies for reproducible builds
```

## ⚙️ Installation

To install the **MuseCoco Text-to-MIDI Service**, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yhbcode000/musecoco-text2midi-service.git
   cd musecoco-text2midi-service
   ```

2. **Install Dependencies with uv (GPU Required)**:

   This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management and **requires a CUDA-compatible GPU**. Install `uv` if you haven't already:

   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Important - Check Your CUDA Version First**:
   ```bash
   # Check your system CUDA version
   nvcc --version
   ```

   **Two-Step Installation** (required due to pytorch-fast-transformers build dependency):

   ```bash
   # Step 1: Install base dependencies (includes PyTorch)
   uv sync

   # Step 2: Install pytorch-fast-transformers (requires torch to be installed first)
   uv pip install pytorch-fast-transformers --no-build-isolation
   ```

   > **Note**: `pytorch-fast-transformers` must be installed separately because it requires PyTorch to be present during its build process.

## 🔧 Configuration

The service uses YAML configuration files located in `storage/config/`. The main configuration file is [`main_config.yaml`](storage/config/main_config.yaml), which is used by the `main.py` script. You can modify these files to configure parameters such as model checkpoints, logging settings, and API keys.

Checkpoints should follow the instructions provided in [`storage/checkpoints/linear_mask-1billion/README.md`](storage/checkpoints/linear_mask-1billion/README.md) and be saved in the same directory as the README.md file.

## 🚀 Usage

### Option 1: Command-Line Demo

To run the terminal-based demo application:

```bash
python src/musecoco_text2midi_service/main.py
```

The `src/musecoco_text2midi_service/main.py` file provides a terminal-based app demo.

> Refer to `storage/input/predict_backup.json` for examples of acceptable input formats for the service. This file contains sample data that illustrates how to structure text input for the MIDI generation process.

### Option 2: FastAPI REST API Server

To start the FastAPI REST API server on port 8001:

```bash
# Using uv to run the FastAPI server
uv run python fastapi_server.py --port 8001

# Or activate the environment first
source .venv/bin/activate  # On Linux/macOS
python fastapi_server.py --port 8001
```

The server accepts the following arguments:
- `--host` - Host address to bind (default: 0.0.0.0)
- `--port` - Port number (default: 8001)
- `--reload` - Enable auto-reload for development
- `--workers` - Number of worker processes (default: 1)

#### API Endpoints

The FastAPI server provides the following REST API endpoints:

- **GET** `/` - API information and documentation links
- **GET** `/health` - Health check endpoint
- **POST** `/submit-text` - Submit text for MIDI generation (returns a job ID)
- **GET** `/check-status/{job_id}` - Check the status of a MIDI generation job
- **GET** `/get-result/{job_id}` - Get the metadata of a completed MIDI generation
- **GET** `/download-midi/{job_id}` - Download the generated MIDI file

#### Interactive API Documentation

FastAPI provides **automatic interactive API documentation**:

- **Swagger UI**: `http://localhost:8001/docs` - Interactive API explorer with request/response examples
- **ReDoc**: `http://localhost:8001/redoc` - Clean, responsive API documentation

#### Example API Usage

```bash
# Submit a text for MIDI generation
curl -X POST http://localhost:8001/submit-text \
  -H "Content-Type: application/json" \
  -d '{"text": "This music uses a major key, with grand piano and cello, conveying edginess."}'

# Response:
# {
#   "jobId": "abc-123",
#   "status": "submitted",
#   "message": "Job submitted successfully. Use the job_id to check status."
# }

# Check job status
curl http://localhost:8001/check-status/abc-123
# Response: {"jobId": "abc-123", "status": "completed"}

# Get result metadata
curl http://localhost:8001/get-result/abc-123
# Response:
# {
#   "jobId": "abc-123",
#   "status": "completed",
#   "metaData": {...}
# }

# Download MIDI file
curl -O http://localhost:8001/download-midi/abc-123
```

#### Python Client Example

```python
import requests

# Submit job
response = requests.post(
    "http://localhost:8001/submit-text",
    json={"text": "A peaceful piano melody in C major"}
)
job_id = response.json()["jobId"]

# Poll for completion
import time
while True:
    status = requests.get(f"http://localhost:8001/check-status/{job_id}")
    if status.json()["status"] == "completed":
        break
    time.sleep(1)

# Download MIDI file
midi_file = requests.get(f"http://localhost:8001/download-midi/{job_id}")
with open("output.mid", "wb") as f:
    f.write(midi_file.content)
```

### Option 3: Python Package Import

You can also import the package into your own Python project:

```python
from musecoco_text2midi_service.control import Text2Midi
from musecoco_text2midi_service.dao import load_config_from_file

config = load_config_from_file("storage/config/main_config.yaml")
text2midi = Text2Midi(config)

input_text = "This music's use of major key creates a distinct atmosphere, with a playtime of 1 ~ 15 seconds. The rhythm in this song is very pronounced, and the music is enriched by grand piano, cello and drum. Overall, the song's length is around about 6 bars. The music conveys edginess."

midi_data, meta_data = text2midi.text_to_midi(input_text, return_midi=True)
``` 

## 🧪 Running Tests

To run the test suite, use:

```bash
pytest tests/
```

This command will execute all test cases in the `tests` directory and provide a report of the test results. Ensure that the project is built correctly before running the tests.

## 🤝 Contributing

We welcome contributions from the community. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with descriptive commit messages.
4. Push your changes to your forked repository.
5. Create a pull request with a detailed description of your changes.

## 📄 License

This project is licensed under Apache License 2.0 - see the LICENSE file for more details.

## Reference:

- https://askubuntu.com/questions/1288672/how-do-you-install-cuda-11-on-ubuntu-20-10-and-verify-the-installation
- https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/#conda-overview
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

## Recommendation 

- https://hydra.cc/docs/1.3/intro/

## Notice

- `pytorch-fast-transformers` requires a two-step installation process (see [Installation](#️-installation) section)
- This package has a build-time dependency on PyTorch, so it must be installed after PyTorch is available

---

Thank you for your interest in the MuseCoco Text-to-MIDI Service! If you have any questions or need further assistance, feel free to open an issue or contact us.
