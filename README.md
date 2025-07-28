# Spoton ML - MTMC Person Detection & Tracking Framework

This repository contains a Python framework for developing, training, evaluating, and comparing machine learning models for Multi-Target Multi-Camera (MTMC) person detection and tracking tasks, primarily focused on the MTMMC dataset format. It includes support for various detection models, tracking algorithms via BoxMOT, model explainability, and MLOps integration using MLflow and DagsHub.

## Key Features

*   **Configurable Workflows:** Manage experiments using YAML configuration files (`configs/`).
*   **Multiple Detection Models:** Supports strategies for:
    *   YOLO (via Ultralytics)
    *   RT-DETR (via Ultralytics)
    *   Faster R-CNN (ResNet50 backbone via TorchVision)
    *   RF-DETR (via `rfdetr` library with automatic COCO format conversion)
*   **Tracking & Re-ID:** Integrates with the BoxMOT library for various trackers (DeepSORT, StrongSORT, BoT-SORT, OCSORT, BoostTrack, ImprAssoc, etc.) combined with different Re-Identification models.
*   **MTMMC Dataset Support:** Includes data loaders specifically designed for the MTMMC dataset structure.
*   **Training Pipeline:** Dedicated pipelines for training object detection models (Faster R-CNN and RF-DETR).
*   **Evaluation:**
    *   Calculates standard detection metrics (mAP) using `torchmetrics`.
    *   Calculates tracking metrics (IDF1 focus) using `motmetrics`.
*   **Model Explainability:** Provides Grad-CAM explanations for Faster R-CNN detections using `captum`.
*   **MLOps Integration:** Logs experiments, parameters, metrics, and artifacts to MLflow. Supports remote tracking via DagsHub or other MLflow servers.
*   **Reproducibility:** Utilities for setting random seeds.
*   **Utilities:** Scripts for data download (S3/DagsHub), Re-ID weight conversion, and image sequence to video conversion.
*   **Testing:** Includes unit/integration tests using `pytest`.

## Project Structure

```
└── jubjones-spoton_ml/
    ├── README.md                 # This file
    ├── LICENSE                   # MIT License
    ├── pytest.ini                # Pytest configuration
    ├── requirements.txt          # Project dependencies
    ├── configs/                  # YAML configuration files for workflows
    │   ├── comparison_run_config.yaml # Detection model comparison
    │   ├── eda_config.yaml          # Exploratory Data Analysis
    │   ├── explainability_config.yaml # Model explainability (FasterRCNN)
    │   ├── fasterrcnn_training_config.yaml # Faster R-CNN training
    │   ├── rfdetr_training_config.yaml # RF-DETR training
    │   ├── single_run_config.yaml   # Single detection run
    │   └── tracking_reid_comparison_config.yaml # Tracker+ReID comparison
    ├── outputs/                  # Default output directory (e.g., explanations)
    ├── scripts/                  # Utility scripts (data download, weight export)
    ├── src/                      # Main source code
    │   ├── run_*.py              # Entry points for different workflows
    │   ├── core/                 # Core runner logic
    │   ├── data/                 # Data loading and dataset classes
    │   ├── detection/            # Detection model strategies
    │   ├── evaluation/           # Metrics calculation
    │   ├── explainability/       # Model explanation logic
    │   ├── inference/            # Inference helper functions
    │   ├── pipelines/            # High-level workflow pipelines
    │   ├── training/             # Model training logic
    │   └── utils/                # Helper utilities (config, device, logging, etc.)
    └── tests/                    # Unit and integration tests
```

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd jubjones-spoton_ml
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # OR
    .\.venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    *   **PyTorch:** Install PyTorch matching your system/CUDA setup first. Follow instructions on the [PyTorch website](https://pytorch.org/get-started/locally/). For example:
        ```bash
        # For CUDA 12.4 (adapt for your system)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        
        # For CPU only
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        
        # For macOS with MPS
        pip3 install torch torchvision torchaudio
        ```
    *   **Core Dependencies:** Install the remaining packages.
        ```bash
        pip install -r requirements.txt
        ```
    *   **RF-DETR Support (Optional):** For RF-DETR model training:
        ```bash
        pip install git+https://github.com/roboflow/rf-detr.git
        ```
        **Note:** RF-DETR requires specific dataset structure. The framework automatically handles the conversion from MTMMC format to RF-DETR's expected COCO format during training.

4.  **MLflow & DagsHub Setup (Optional but Recommended):**
    *   For remote tracking with DagsHub:
        *   Install DagsHub client (`pip install dagshub` - already in requirements).
        *   Login to DagsHub: `dagshub login`
        *   Create a `.env` file in the project root (`jubjones-spoton_ml/.env`) with your DagsHub repository details:
            ```dotenv
            DAGSHUB_REPO_OWNER=<Your_DagsHub_Username>
            DAGSHUB_REPO_NAME=<Your_DagsHub_Repository_Name>
            # MLFLOW_TRACKING_URI=https://dagshub.com/Your_DagsHub_Username/Your_DagsHub_Repository_Name.mlflow # Often set automatically by dagshub.init
            # AWS_ACCESS_KEY_ID=... # For S3 access if needed
            # AWS_SECRET_ACCESS_KEY=... # For S3 access if needed
            ```
    *   For other MLflow servers, set the `MLFLOW_TRACKING_URI` environment variable.
    *   If no remote URI is configured, MLflow will track locally in the `mlruns` directory.

5.  **Download Data:**
    *   The project expects the MTMMC dataset.
    *   Download the dataset manually or use the provided scripts (`scripts/download_from_dagshub.py` or `scripts/download_from_boto3.py`) if you have uploaded the data to DagsHub storage or an S3 bucket.
    *   **Crucially:** Update the `base_path` in all relevant `configs/*.yaml` files to point to the root directory of your downloaded MTMMC dataset.

6.  **Prepare Re-ID Weights (If using Tracking):**
    *   Some Re-ID models used by BoxMOT might require specific weight formats.
    *   Use the `scripts/export_reid_weights.py` script if you encounter issues loading Re-ID weights. See `scripts/README.md` for example usage.
    *   Place the prepared weights in the directory specified by `data.weights_base_dir` in `configs/tracking_reid_comparison_config.yaml` (default is `weights/reid`).

## Configuration

All workflows are driven by YAML configuration files located in the `configs/` directory.

*   **`base_path`:** **Must be updated** in relevant configs to point to your MTMMC dataset location.
*   **`environment`:** Set `device` (`auto`, `cuda:0`, `cpu`, `mps`) and random `seed`.
*   **`data`:** Configure dataset specifics, scenes, cameras, and subsetting options.
*   **`model`/`tracker`/`reid_model`:** Specify model types, paths, and parameters.
*   **`mlflow`:** Set experiment names.
*   **`training`:** Configure training hyperparameters (epochs, batch size, optimizer, etc.).
*   **`explainability`:** Configure Grad-CAM parameters (target layer, class index, etc.).

## Usage

Run different workflows using the `run_*.py` scripts in the `src/` directory. Ensure your configuration files are correctly set up before running.

*   **Exploratory Data Analysis (EDA):**
    ```bash
    python src/run_eda.py
    ```
    *(Config: `configs/eda_config.yaml`)*

*   **Train Faster R-CNN:**
    ```bash
    python src/run_training_fasterrcnn.py
    ```
    *(Config: `configs/fasterrcnn_training_config.yaml`)*

*   **Train RF-DETR:**
    ```bash
    python src/run_training_rfdetr.py
    ```
    *(Config: `configs/rfdetr_training_config.yaml`)*

*   **Run Single Detection Experiment:**
    ```bash
    python src/run_experiment.py
    ```
    *(Config: `configs/single_run_config.yaml`)*

*   **Run Detection Model Comparison:**
    ```bash
    python src/run_comparison.py
    ```
    *(Config: `configs/comparison_run_config.yaml`)*

*   **Run Tracking + Re-ID Comparison (using GT boxes):**
    ```bash
    python src/run_tracking_reid_comparison.py
    ```
    *(Config: `configs/tracking_reid_comparison_config.yaml`)*

*   **Run Model Explainability (Faster R-CNN):**
    ```bash
    python src/run_explainability.py
    ```
    *(Config: `configs/explainability_config.yaml`)*

## Explainability

The framework includes functionality to generate Grad-CAM explanations for trained Faster R-CNN models.

1.  **Configure:** Edit `configs/explainability_config.yaml`:
    *   Set `model.checkpoint_path` to your trained model checkpoint (local path or MLflow URI).
    *   Specify the `images_to_explain`.
    *   Define the `target_layer_name` (e.g., `backbone.body.layer4` for ResNet).
    *   Adjust `confidence_threshold_for_explanation`, `top_n_to_explain`, etc.
    *   Set `output_dir` for visualizations.
2.  **Run:**
    ```bash
    python src/run_explainability.py
    ```
3.  **Output:**
    *   Heatmap visualizations overlaid on images saved to the specified `output_dir`.
    *   A detailed reasoning log (`*_reasoning_log.txt`) summarizing the explanations.
    *   If MLflow logging is enabled (`mlflow.log_artifacts: true`), outputs are logged as artifacts.

## MLOps Integration (MLflow)

*   MLflow is used extensively for tracking experiments, parameters, metrics, and artifacts (configurations, logs, models, plots).
*   Runs are logged either locally (`mlruns/` directory) or to a remote server (DagsHub or specified by `MLFLOW_TRACKING_URI`).
*   Configure experiment names in the `mlflow:` section of the YAML configuration files.
*   Parent runs are created for comparison workflows, with individual model/tracker runs nested underneath.

## Dependencies

All required Python packages are listed in `requirements.txt`. Key libraries include:

*   `torch`, `torchvision`, `torchaudio`
*   `ultralytics`
*   `rfdetr`
*   `boxmot`
*   `captum`
*   `mlflow`, `dagshub`
*   `opencv-python`
*   `numpy`, `pandas`, `scipy`
*   `pyyaml`
*   `matplotlib`
*   `torchmetrics[detection]`
*   `motmetrics`
*   `pytest`

## Testing

Unit and integration tests are located in the `tests/` directory and can be run using `pytest`:

```bash
pytest
```
*(Requires `pytest` to be installed)*

## Troubleshooting

### RF-DETR Training Issues

If you encounter issues with RF-DETR training, common problems and solutions:

1. **Missing validation annotations file**: The framework automatically handles the conversion from MTMMC format to RF-DETR's expected COCO format. The validation data is created in a `valid/` directory structure that RF-DETR requires.

2. **Dataset path issues**: Ensure the `base_path` in `configs/rfdetr_training_config.yaml` points to your MTMMC dataset root directory.

3. **MLflow connection issues**: RF-DETR training uses MLflow for experiment tracking. If you encounter connection issues:
   - Check your `.env` file configuration
   - Verify DagsHub credentials with `dagshub login`
   - For local tracking, remove or comment out the `MLFLOW_TRACKING_URI` environment variable

4. **Memory issues**: RF-DETR models can be memory-intensive. Reduce the `batch_size` in the configuration if you encounter out-of-memory errors.

### General Issues

- **Device compatibility**: The framework automatically selects the best available device (CUDA > MPS > CPU). For model-specific requirements, check the device configuration in each config file.
- **Dataset loading**: Ensure your MTMMC dataset structure matches the expected format and all required scenes/cameras are present.
