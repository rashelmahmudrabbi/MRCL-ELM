# MRCL-ELM: Hybrid Deep Learning Model for Satellite Image Classification

## Overview

MRCL-ELM is a state-of-the-art land use and land cover classification system designed for satellite image analysis. This project implements a novel hybrid deep neural architecture that combines EfficientNet-B0, multi-stage residual blocks, feature fusion, LSTM layers, and Extreme Learning Machine (ELM) classifier to achieve high-accuracy satellite image classification.

### Key Features

- **Advanced Architecture**: Hybrid deep learning model integrating multiple cutting-edge techniques
- **High Performance**: Achieves >98% test accuracy on benchmark datasets
- **Real-time Inference**: Processing speed of approximately 2.1 ms per image
- **Interactive Interface**: Fully functional web application for interactive prediction
- **Multiple Dataset Support**: Validated on EuroSAT and UC Merced Land datasets

### Technical Specifications

- **Framework**: PyTorch 2.1
- **Language**: Python 3.10
- **License**: MIT
- **Primary Datasets**: EuroSAT, UC Merced Land Use

## Project Structure

```
MRCL-ELM/
│
├── Model/
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── MRCL-ELM.py
│   │   ├── preprocessing.py
│   │   ├── test.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   ├── .gitignore
│   │   ├── environment.yml
│   │   ├── LICENSE
│   │   ├── README.md
│   │   └── requirements.txt
│   │
│   ├── XAI/
│   │   ├── LIME/
│   │   │   ├── .gitignore
│   │   │   ├── README.md
│   │   │   ├── config.py
│   │   │   ├── data_utils.py
│   │   │   ├── model.py
│   │   │   ├── lime_explain.py
│   │   │   ├── visualize.py
│   │   │   ├── run_lime.py
│   │   │   └── requirements.txt
│   │   │
│   │   └── SHAP/
│   │       ├── .gitignore
│   │       ├── README.md
│   │       ├── config.py
│   │       ├── data_utils.py
│   │       ├── model.py
│   │       ├── shap_explain.py
│   │       ├── visualize.py
│   │       ├── run_shap.py
│   │       └── requirements.txt
│
├── web-App/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── requirements.txt
│
├── .gitignore
├── LICENSE
└── README.md

```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- pip or conda package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rashelmahmudrabbi/MRCL-ELM.git
   cd MRCL-ELM
   ```

2. **Create Virtual Environment**
   
   Using conda:
   ```bash
   conda env create -f environment.yml
   conda activate mrcl-elm
   ```
   
   Using pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Settings**
   
   Edit the configuration file at `Model/configs/config.yaml` to specify:
   - Dataset paths
   - Model hyperparameters
   - Training parameters
   - Output directories

## Usage

### Data Preprocessing

Prepare your satellite imagery dataset using the preprocessing module:

```bash
python src/preprocessing.py --data_dir /path/to/dataset --output_dir /path/to/processed
```

### Model Training

Train the MRCL-ELM model on your prepared dataset:

```bash
python src/train.py --config Model/configs/config.yaml
```

### Model Evaluation

Evaluate the trained model on test data:

```bash
python src/test.py --model_path /path/to/model.pth --data_dir /path/to/test_data
```

### Web Application

Launch the interactive web application for real-time predictions:

```bash
cd web-App
python app.py
```

Access the application at `http://localhost:5000`

### Python API Usage

```python
from src.MRCL_ELM import MRCLELMClassifier
from src.utils import load_config

# Load configuration
config = load_config('Model/configs/config.yaml')

# Initialize model
model = MRCLELMClassifier(config)

# Load trained weights
model.load_weights('path/to/model.pth')

# Predict on new image
prediction = model.predict('path/to/satellite_image.jpg')
print(f"Predicted class: {prediction}")
```

## Documentation

For detailed documentation, please refer to:

- **[INSTALLATION.md](docs/INSTALLATION.md)**: Comprehensive installation guide
- **[MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)**: Detailed model architecture description
- **[USAGE.md](docs/USAGE.md)**: Advanced usage examples and API reference

## Performance Metrics

- **Test Accuracy**: >98%
- **Inference Speed**: ~2.1 ms per image
- **Datasets**: EuroSAT, UC Merced Land Use
- **Model Size**: Optimized for deployment

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgments

This research utilizes the EuroSAT and UC Merced Land Use datasets for satellite image classification tasks.


**Version**: 1.0.0  
**Last Updated**: December 2025
