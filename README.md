# Deep learning to classify likely recurrent non-small cell lung cancer (NSCLC)

## Description
This repository contains code for training a deep learning model to predict recurrence of NSCLC from CT images. The DenseNet model is implemented using PyTorch and managed with Conda for environment management. The original model was trained on 110 images, and in March 2024, an additional 500 scans are expected to be added.

## Usage

### Installation
1. Ensure you have Conda installed on your system.
2. Clone this repository:
   ```
   git clone git@github.com:Pkaps25/nsclcrecurrence.git nsclc
   ```
3. Navigate to the project directory:
   ```
   cd nsclc
   ```
4. Create a Conda environment from the provided `environment.yml` file:
   ```
   conda env create -f environment.yml -n lung-cancer-classifier
   ```
5. Activate the Conda environment:
   ```
   conda activate lung-cancer-classifier
   ```

### Training
1. Set the DATA_DIR environment variable to point to the directory containing your data
   ```
   export DATA_DIR="path-to-your-data"
   ```
2. Use the provided Python script to train the classifier:
   ```
   python main.py
   ```

### Inference
1. After training, you can use the trained model for inference on new lung cancer images.
2. Example inference code:
   ```
   import torch
   from model import NoduleRecurrenceClassifier
   from datasets import CTImage

   # Load trained model
   model = NoduleRecurrenceClassifier()
   model.load_state_dict(torch.load('trained_model.pth'))
   model.eval()

   # Perform inference on new image
   image = CTImage(nodule_id)
   prediction = model(image)
   ```

## Requirements
- Python 3.7
- Conda

## File Structure
```
nsclcrecurrence/
│
├── ct_lung_class/                          # Directory for storing lung cancer image dataset
│   ├── train.py                            # Python script for training the model
│   ├── main.py                             # Wrapper for invoking model training
|   ├── datasets.py                         # Preprocessing and data loading
│             
├── environment.yml                # Conda environment file specifying dependencies
└── README.md                      # This README file
```

## Contributing
Contributions to this project are welcome! Feel free to open issues for bug fixes, feature requests, or general improvements. Pull requests are also appreciated.
