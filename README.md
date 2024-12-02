# Deep learning to classify likely recurrent non-small cell lung cancer (NSCLC)

## Description
This repository contains code for training vision models for two tasks: Classifying papillary lung cancer vs NSCLC and classifying NSCLC vs SCLC. Data and annotations are proprietary.

## File Structure
```
ct_lung_class/
│
├── ct_lung_class/
│   ├── run.py                              # Entrypoint to training
│   ├── train.py                            # Main training logic
|   ├── datasets.py                         # Data preprocessing and augmentation
|   ├── image.py                            # Reading images and defining data objects/datasets
│
├── environment.yml                # Conda environment file specifying dependencies
└── README.md                      
```

## Contributing
Contributions to this project are welcome! Feel free to open issues for bug fixes, feature requests, or general improvements. Pull requests are also appreciated.
