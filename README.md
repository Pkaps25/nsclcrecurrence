# Deep learning to classify likely recurrent non-small cell lung cancer (NSCLC)

## Description
This repository contains code for training a deep learning model to classify NSCLC by papillary subtype. The model is trained on a dataset of 305
NSCLC images.

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
4. Set the variables in `ct_lung_class/conf/config.yml` to match your system
5. Create a Conda environment from the provided `environment.yml` file:
   ```
   conda env create -f environment.yml -n nsclc
   ```
6. Activate the Conda environment:
   ```
   conda activate nsclc
   ```

### Training
1. The invocation to training is `ct_lung_class/run.py`. Invoke training with:
   ```
   python ct_lung_class/run.py --epochs 1000 --k-folds 5 --dilate 10 --resample 64 --tag --weight-decay 0.001 --learn-rate 0.001 label-run
   ```


## File Structure
```
ct_lung_class/
│
├── ct_lung_class/
│   ├── run.py                              # Entrypoint to training
│   ├── train.py                            # Driver for training
|   ├── datasets.py                         # Data preprocessing and augmentation
|   ├── image.py                            # Reading images and defining datasets
│
├── environment.yml                # Conda environment file specifying dependencies
└── README.md                      # This README file
```

## Contributing
Contributions to this project are welcome! Feel free to open issues for bug fixes, feature requests, or general improvements. Pull requests are also appreciated.
