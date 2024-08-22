# from collections import defaultdict
import datetime
# from functools import lru_cache
import glob
import logging
import os
from pathlib import Path
import tempfile
# from typing import List
import SimpleITK as sitk
# import multiprocessing as mp
# import xnat
import json
import csv
import zipfile


def setup_logger(name, log_file):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

# Set up the logger
logger = setup_logger(__name__, 'app.log')

PROJECT_ID = "17-353D_Prasad"

DATA_DIR = "/data/kaplinsp/dicom/"

      
import shutil
def process_files():
    zips = glob.glob("/data/kaplinsp/prasad_d/*.zip")
    for zip_file_path in zips:
        subject_id = Path(zip_file_path).stem
        dicom_dir = os.path.join("/data/kaplinsp/prasad_d", subject_id)
        if not os.path.exists(dicom_dir):
            os.mkdir(dicom_dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                dicom_files = glob.glob(os.path.join(temp_dir, '*', 'scans', '*', 'resources', 'DICOM', 'files/*dcm'))
                print(f"Moving files from {zip_file_path} to {dicom_dir}")
                for dcm in dicom_files:
                    dst = Path(dicom_dir) / Path(dcm).name
                    shutil.move(dcm, dst)
                
        
                

if __name__ == "__main__":
    process_files()
