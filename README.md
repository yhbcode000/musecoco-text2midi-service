# 🎵 MuseCoco Text-to-MIDI Service

The **MuseCoco Text-to-MIDI Service** is a refactored version of the [MuseCoco](https://github.com/microsoft/muzic) repository, designed as a deployable service module. This service adapts to new data, manages the history of its checkpoints, and abstracts away the underlying implementation details to provide a seamless interface for generating MIDI files from textual inputs. Detailed comments are included to facilitate easy navigation and understanding of the codebase.

## 📋 Table of Contents

- [🎵 MuseCoco Text-to-MIDI Service](#-musecoco-text-to-midi-service)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [📂 Directory Structure](#-directory-structure)
  - [⚙️ Installation](#️-installation)
  - [🔧 Configuration](#-configuration)
  - [🚀 Usage](#-usage)
  - [🧪 Running Tests](#-running-tests)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

## ✨ Features

- **Text-to-MIDI Conversion**: Converts textual descriptions into MIDI files using the MuseCoco model.
- **Adaptive Data Handling**: Capable of adapting to new data for more customized MIDI generation.
- **Checkpoint Management**: Manages the history of model checkpoints to ensure reproducibility and flexibility.
- **Abstracted Implementation**: Provides an abstract interface for easy integration while maintaining detailed internal documentation.

## 📂 Directory Structure

The repository is organized as follows:

```plaintext
musecoco-text2midi-service/
├── src/
│   └── musecoco_text2midi_service/
│       ├── control/                   # Controllers for orchestrating service logic
│       │   ├── __init__.py
│       │   ├── _musecoco/
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
│       └── view/                      # Views for API or CLI outputs
│           ├── __init__.py
│           └── _app_view.py
├── storage/
│   ├── checkpoints/                   # Model checkpoints
│   │   └── linear_mask-1billion/
│   │       ├── checkpoint_2_280000.pt
│   │       └── README.md              # Instructions for managing checkpoints
│   ├── config/                        # Configuration files
│   │   ├── main_config.yaml           # Main configuration file
│   │   ├── att_key.json
│   │   └── num_labels.json
│   ├── input/                         # Input files for predictions
│   │   ├── predict_backup.json        # Example input format for predictions
│   │   └── predict.json
│   ├── log/                           # Log files
│   └── tmp/                           # Temporary files and outputs
├── tests/
│   ├── __init__.py                    # Initializer for the tests package
│   └── ...                            # Test modules for various components
├── .gitignore                         # Specifies files and directories to ignore in version control
├── LICENSE                            # License file
├── main.py                            # Entry point for running the service
├── README.md                          # Project description and instructions
└── setup.py                           # Setup script for packaging and distribution
```

## ⚙️ Installation

To install the **MuseCoco Text-to-MIDI Service**, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/musecoco-text2midi-service.git
   cd musecoco-text2midi-service
   ```

2. **Install Dependencies**:

   Create a new environment and install the dependencies. Currently, PyTorch needs to be installed with `conda`:

   ```bash
   conda create -n MuseCoco python=3.8
   conda activate MuseCoco
   conda install pytorch=1.11.0 -c pytorch
   pip install -e .
   ```

   > **Note**: Other dependencies can be installed with `pip install -e .` or `pip install musecoco_text2midi_service`. PyTorch installation with `pip` will be supported later.

## 🔧 Configuration

The service uses YAML configuration files located in `storage/config/`. The main configuration file is [`main_config.yaml`](storage/config/main_config.yaml), which is used by the `main.py` script. You can modify these files to configure parameters such as model checkpoints, logging settings, and API keys.

Checkpoints should follow the instructions provided in [`storage/checkpoints/linear_mask-1billion/README.md`](storage/checkpoints/linear_mask-1billion/README.md) and be saved in the same directory as the README.md file.

## 🚀 Usage

To start the service, run:

```bash
python main.py
```

The `main.py` file provides a terminal-based app demo. 

> Refer to `storage/input/predict_backup.json` for examples of acceptable input formats for the service. This file contains sample data that illustrates how to structure text input for the MIDI generation process.

You can also import the package to your project.

```python
import musecoco_text2midi_service
```

> Later this will launch the MuseCoco Text-to-MIDI Service, allowing you to convert textual descriptions into MIDI files through the defined APIs or CLI. 

## 🧪 Running Tests

To run the test suite, use:

```bash
pytest /tests
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

---

Thank you for your interest in the MuseCoco Text-to-MIDI Service! If you have any questions or need further assistance, feel free to open an issue or contact us.
