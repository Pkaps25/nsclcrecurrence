from functools import lru_cache
import logging
import os
from typing import List
from pyxnat import Interface
from tqdm import tqdm 
import SimpleITK as sitk
import multiprocessing as mp


logger = logging.getLogger(__name__)
PROJECT_ID = "17-353D_Prasad"

DATA_DIR = "/data/kaplinsp/dicom/"

def get_output_prefix(subject_str: str) -> str:
    proj_idx = subject_str.find(PROJECT_ID)
    str_idx = subject_str.rfind("`")
    acc = subject_str[proj_idx: str_idx]
    return acc[acc.rfind("_") + 1:]


# class XNATDataset:

#     def __init__(self) -> None:
#         self.central = Interface(config="xnat.cfg")
#         self.project = self.central.select.project(PROJECT_ID)

#     @lru_cache(maxsize=1)
#     def subjects(self) -> List[str]:
#         return self.project.subjects().get()

#     def collect_data(self, subject_id):
#         reader = sitk.ImageSeriesReader()

#         for subject_id in self.subjects()[:1]:
#             subject_str = repr(self.project.subject(subject_id))
#             output_prefix = get_output_prefix(subject_str)

#             experiments = self.project.subject(subject_id).experiments().get()[:1]
#             for experiment_id in experiments:
#                 output_path = os.path.join(DATA_DIR, output_prefix, experiment_id)
#                 if not os.path.exists(output_path):
#                     os.makedirs(output_path)
#                 resources = self.project.subject(subject_id).experiments(experiment_id).scans('4').resources().get()
#                 if len(resources) > 1:
#                     logger.warn(f"More than 1 resource detected for {subject_id}, {experiment_id}, skippping")
#                     continue
#                 elif len(resources) == 0: 
#                     logger.warn(f"No matching scans for {subject_id}, {experiment_id}")
#                     continue

#                 resource_id = resources.pop()
#                 files = (
#                     self.project.subject(subject_id).experiments(experiment_id).
#                     scans('4').resource(resource_id).files().get()
#                 )
                
#                 # check if files are already downloaded and skip if not
#                 if len(files) != len([name for name in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, name))]):
#                     for file_id in tqdm(files):
#                         tmp_file_path = os.path.join(output_path, file_id)
#                         (
#                             self.project.subject(subject_id).experiment(experiment_id).
#                             scan('4').resource(resource_id).file(file_id).get(tmp_file_path)
#                         )
#                 else:
#                     logger.warn(f"Files already downloaded for {subject_id}, {experiment_id}")

#             dicom_names = reader.GetGDCMSeriesFileNames(output_path)
#             reader.SetFileNames(dicom_names)
#             image = reader.Execute()
#             sitk.WriteImage(image, f"{DATA_DIR}/nod{output_prefix}_{experiment_id}.nrrd")


class XNATDataset:

    def __init__(self, project):
        self.project = project

    def collect_data(self, subject_id):
        reader = sitk.ImageSeriesReader()

        subject_str = repr(self.project.subject(subject_id))
        output_prefix = get_output_prefix(subject_str)

        experiments = self.project.subject(subject_id).experiments().get()[:1]
        for experiment_id in experiments:
            output_path = os.path.join(DATA_DIR, output_prefix, experiment_id)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            resources = self.project.subject(subject_id).experiments(experiment_id).scans('4').resources().get()
            if len(resources) > 1:
                logger.warning(f"More than 1 resource detected for {subject_id}, {experiment_id}, skippping")
                continue
            elif len(resources) == 0: 
                logger.warning(f"No matching scans for {subject_id}, {experiment_id}")
                continue

            resource_id = resources.pop()
            files = (
                self.project.subject(subject_id).experiments(experiment_id).
                scans('4').resource(resource_id).files().get()
            )
            
            # check if files are already downloaded and skip if not
            if len(files) != len([name for name in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, name))]):
                for file_id in tqdm(files):
                    tmp_file_path = os.path.join(output_path, file_id)
                    (
                        self.project.subject(subject_id).experiment(experiment_id).
                        scan('4').resource(resource_id).file(file_id).get(tmp_file_path)
                    )
            else:
                logger.warning(f"Files already downloaded for {subject_id}, {experiment_id}")

        dicom_names = reader.GetGDCMSeriesFileNames(output_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, f"{DATA_DIR}/nod{output_prefix}_{experiment_id}.nrrd")



if __name__ == "__main__":
    central = Interface(config="xnat.cfg")
    project = central.select.project(PROJECT_ID)
    subjects = project.subjects().get()
    with mp.Pool(processes=10) as pool:
        pool.map(
            XNATDataset(project).collect_data,
            subjects
        )
    pool.join()
    pool.close()