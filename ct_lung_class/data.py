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


# def get_output_prefix(subject_str: str) -> str:
#     # proj_idx = subject_str.find(PROJECT_ID)
#     # str_idx = subject_str.rfind("`")
#     # acc = subject_str[proj_idx: str_idx]
#     # return acc[acc.rfind("_") + 1:]
#     return subject_str[subject_str.rfind("_")+1:]

# def get_xnat_session():
#     with open("xnat.cfg", 'r') as jsonfile:
#         config = json.load(jsonfile)
#     return xnat.connect(**config)

# def read_coord_csv():
#     with open("annots_michelle.csv") as csvfile:
#         reader = csv.DictReader(csvfile)
#         return list(reader)
    
# # EXCLUDED = {"32", "98", "226", "382"}    

# annot_csv = read_coord_csv()
# session = get_xnat_session()
# project = session.projects['17-353D_Prasad']    
# download = True
# downloads = 0
# for row in annot_csv:
#     try:
#         subject_id = row['XNAT Subject ID ']
#         subject = project.subjects[subject_id]
#         experiments = list(filter(lambda exp: exp.date == datetime.datetime.strptime(row["Pre CT"], '%m/%d/%y').date(), subject.experiments.values()))
#         if not experiments:
#             logger.warning(f"No experiemnts for {subject_id}")
#             continue
#         experiment = experiments[0]
#         scan = experiment.scans[row["Series"]]
#         file_path = f"/data/kaplinsp/prasad_d/{subject_id}.zip"
#         if os.path.exists(file_path):
#             logger.info(f"Skipping {subject_id} with existing file")
#             continue

#         logger.info(f"Downloading scan for {subject_id}")
#         if download:
#             downloads += 1
#             scan.download(file_path)
    
#     except Exception as e:
#         print(f"Encountering error for {subject_id}: {e}")

# logger.info(f"Total of {downloads} downloads")
      
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
