from ast import literal_eval
import csv
from dataclasses import dataclass
import functools
import logging
from typing import List, Optional, Tuple, Union
import SimpleITK as sitk
import numpy as np
from lungmask import LMInferer


logger = logging.getLogger(__name__)

def get_coord_csv(c1, c2, c3):
    if "R" in c1:
        x = -float(c1.strip("R"))
    else:
        x = float(c1.strip("L"))
    if "A" in c2:
        y = -float(c2.strip("A"))
    else:
        y = float(c2.strip("P"))
    if "I" in c3:
        z = -float(c3.strip("I"))
    else:
        z = float(c3.strip("S"))
    center = tuple([x, y, z])
    return center


# def create_image_cache(scope_str: str):
#     return FanoutCache(
#             "/data/kaplinsp/image-cache/" + scope_str,
#             disk=Disk,
#             shards=64,
#             timeout=1,
#             size_limit=3e11,
#         )
    
# raw_cache = create_image_cache("nod1")


Point = Union[int, float]
Coord3D = Union[Tuple[Point, Point, Point], np.array]

def ras_to_lps(tup: np.array) -> np.array:
    tup[0] *= -1
    tup[1] *= -1
    return tup


class NoduleImage:
    def __init__(self, image_file_path: str, center_lps: np.array):
        # self.nod_id = nod_id
        self.image_file_path = image_file_path
        self.center_lps = center_lps
        
    def _image(self):
        raise NotImplementedError("Subclasses must override")
    
    def _get_3d_slice(self, center: Coord3D, dims: Coord3D) -> Tuple[slice, slice, slice]:
        index = self.image.TransformPhysicalPointToIndex(center)
        size_x, size_y, size_z = dims
        
        start_x = int(max(0, index[0] - size_x // 2))
        end_x = int(min(self.image.GetWidth(), index[0] + size_x // 2))

        start_y = int(max(0, index[1] - size_y // 2))
        end_y = int(min(self.image.GetHeight(), index[1] + size_y // 2))

        start_z = int(max(0, abs(index[2]) - size_z // 2))
        end_z = int(min(self.image.GetDepth(), abs(index[2]) + size_z // 2))
        
        return slice(start_z, end_z), slice(start_y, end_y), slice(start_x, end_x)
        
    
    def lung_segmentation(self) -> np.array:
         inferrer = LMInferer(tqdm_disable=True)
         mask = inferrer.apply(self.image_array())
         mask[mask.nonzero()] = 1
         return mask
        
        
    def image_array(self) -> np.array:
        image_arr = sitk.GetArrayFromImage(self.image)
        return image_arr
        
    
    
    def nodule_slice(self, box_dim: Coord3D = (60,60,60)) -> Tuple[np.array, Tuple[slice,slice,slice]]:
        logger.info(f"Slicing nodule for {self.image_file_path}")
        slice_3d = self._get_3d_slice(self.center_lps, box_dim)

        image_array = self.image_array()
        
        sliced_arr = image_array[slice_3d] 
        # logger.info(f"Slice shape {sliced_arr.shape} for nodule {self.image_file_path}")
        pad_width = [(0, max(0, box_dim[2-i] - sliced_arr.shape[i])) for i in range(3)]
        padded_arr = np.pad(sliced_arr, pad_width=pad_width, mode='constant', constant_values=0)
        # logger.info(f"Padded shape {padded_arr.shape} for nodule {self.image_file_path}")
        return padded_arr, slice_3d
        # return sliced_arr

class NRRDNodule(NoduleImage):
    """Class for loading nodules from NRRD"""
    
    @property
    def image(self) -> sitk.Image:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(self.image_file_path)
        return reader.Execute()
    


class DICOMNodule(NoduleImage):
    """Class for loading an image of a nodule from DICOM"""
    
    @property
    def image(self) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(self.image_file_path)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        return image


@dataclass
class NoduleInfoTuple:
    is_nodule: int 
    nod_id: str
    center_lps: Coord3D
    file_path: str
    image_type: NoduleImage


class SampleGeneratorStrategy:
    def generate_nodule_info(self) -> List[NoduleInfoTuple]:
        raise NotImplementedError("You should implement this method!")


class R17SampleGeneratorStrategy(SampleGeneratorStrategy):
    def generate_nodule_info() -> List[NoduleInfoTuple]:
        coord_file = "/home/kaplinsp/ct_lung_class/ct_lung_class/annotations.csv"
        ids_to_exclude = set(["103", "128", "20", "26", "29", "69", "61"])

        nodule_infos = []

        with open(coord_file, newline="") as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
            next(reader)
            for row in reader:
                nod_name = row[0]
                if nod_name in ids_to_exclude:
                    logger.info(f"EXCLUDING: {nod_name}")
                    continue

                center = get_coord_csv(row[4], row[5], row[6])
                label = int(row[7])
                file_path = f"/data/etay/lung_hist_dat/original_dat_nrrds/nod{nod_name}.nrrd"
                nodule_infos.append(
                    NoduleInfoTuple(
                        label,
                        nod_name,
                        center,
                        file_path,
                        NRRDNodule
                    )
                )

        return nodule_infos

class PrasadSampleGeneratoryStrategy(SampleGeneratorStrategy):
    def generate_nodule_info() -> List[NoduleInfoTuple]:
        coord_file = "/home/kaplinsp/annots_michelle_label.csv"
        nodule_infos = []
        
        with open("/home/kaplinsp/ct_lung_class/ct_lung_class/exclude_michelle.csv", "r") as excludefile:
            exclude = [line.strip() for line in excludefile]
        
        with open(coord_file) as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            next(reader)
            for row in reader:
                
                subject_id = row[3]
                if subject_id in exclude:
                    logger.warning(f"Skipping {subject_id}")
                    continue 
                try:
                    coords = ras_to_lps(np.array(literal_eval(row[8])))
                except SyntaxError:
                    logger.warning(f"No coordinates for {subject_id}")
                    continue
                
                file_path = f"/data/kaplinsp/prasad_d/{subject_id}"
                nodule_infos.append(NoduleInfoTuple(int(row[-1]), subject_id, coords, file_path, DICOMNodule))
        
        return nodule_infos   
    
    
class NoduleInfoGenerator:
    def __init__(self):
        self.strategies = []

    def add_strategies(self, *strategy: SampleGeneratorStrategy):
        self.strategies.extend(strategy)

    def generate_all_samples(self) -> List[NoduleInfoTuple]:
        all_samples = []
        for strategy in self.strategies:
            all_samples.extend(strategy.generate_nodule_info())
        return all_samples
    
