from abc import ABC, abstractmethod
from ast import literal_eval
import csv
from dataclasses import dataclass
import logging
import os
import pickle
from typing import List, Tuple, Union
import SimpleITK as sitk
import numpy as np

from exclude_prasad import EXCLUDE_PRASAD_IDS

[
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod5.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod8.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod15.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod32.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod33.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod42.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod76.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod82.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod89.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod98.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod130.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod131.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod138.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod139.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod147.nrrd",
    "/data/etay/lung_hist_dat/original_dat_nrrds/nod185.nrrd",
]


# Manual segs fixed 8, 20, 32, 33, 130
# Sep 11: 5, 24, 42, 76, 128, 173
EXCLUDE_NODULE_FILES = [
    # (5, (68.5, 25.6, -52.8)),  # fixed NOT YET USED
    # (8, (104.4, 31.5, -105.0)), # fixed low confidence
    (15, (72.4, 4.2, -73.8)),  # TODO unable
    (20, (51.0, 32.6, -30.9)),  # TODO not being found
    # (24, (88.9, -153.8, -83.8)),  # fixed NOT YET USED
    # (26, (-76.1, 77.4, -126.0)), # fixed coordinate
    # (32, (-105.6, -87.2, -204.4)), # fixed seg
    (33, (-103.6, 41.3, -145.6)),  # fixed seg
    # (42, (-119.9, 36.6, -72.5)),  # fixed NOT YET USED
    (44, (-92.8, -34.3, -186.1)),  # TODO need help
    (61, (-82.3, -4.6, 167.5)),  # bad image
    (63, (-44.3, 44.4, -66.2)),  # TODO need help
    # (69, (89.3, 76.9, -128.7)), # fixed coordinate
    # (76, (-28.8, 38.9, -68.8)),  # fixed NOT YET USED
    (82, (-38.2, 17.1, -211.2)),  # fixed seg
    (84, (-64.6, 60.5, -114.5)),
    (86, (36.5, 54.6, -124.0)),  # TODO check if z coordinate is actually I240
    (89, (-86.5, -49.7, -114.0)),  # TODO No nodule
    (98, (35.8, 45.9, -102.5)),  # TODO need help
    (103, (112.6, 74.4, 2688.8)),  # bad image
    # (128, (113.4, 38.7, 72.5)),  # fixed NOT YET USED (bad coord)
    # (130, (-123.3, 55.5, -180.9)), # fixed seg
    (131, (-107.7, 73.5, -137.5)),  # TODO need help
    (138, (-96.8, 45.9, -145.0)),  # TODO need help
    (139, (-32.5, 65.7, -111.2)),  # TODO need help
    (144, (-79.9, 78.0, -142.5)),  # TODO need help
    (147, (-59.7, 31.8, -40.6)),  # TODO need help
    # (173, (-91.0, 92.9, -181.8)),  # fixed seg, merged, NOT YET USED
    (178, (-51.9, 12.3, -42.5)),  # TODO need help
    (185, (86.7, -153.3, 608.5)),  # TODO need help
]


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


Point = Union[int, float]
Coord3D = Union[Tuple[Point, Point, Point], np.array]
Slice3D = Tuple[slice, slice, slice]
Image = np.ndarray

CT_AIR, CT_BONE = -1000, 1000


def ras_to_lps(tup: np.array) -> np.array:
    tup[0] *= -1
    tup[1] *= -1
    return tup


class NoduleImage(ABC):
    def __init__(self, image_file_path: str, center_lps: np.array):
        self.image_file_path = image_file_path
        self.center_lps = center_lps

    @abstractmethod
    def image(self):
        pass

    @abstractmethod
    def nodule_segmentation_image(self) -> sitk.Image:
        pass

    def get_connected_component_id_for_nodule(self, labeled_segmentation_image: sitk.Image) -> int:
        index = labeled_segmentation_image.TransformPhysicalPointToIndex(self.center_lps)
        index_box = [slice(x - 10, x + 10) for x in index]
        center_box = sitk.GetArrayFromImage(labeled_segmentation_image[index_box])
        segs = np.unique(center_box)
        return int(np.max(segs))

    def extract_bounding_box_nodule(self, preprocess: bool, dilation_mm: int) -> sitk.Image:
        segmentation_image = self.nodule_segmentation_image()
        labeled_segmentation_image = sitk.ConnectedComponent(segmentation_image)
        segmentation_id = self.get_connected_component_id_for_nodule(labeled_segmentation_image)

        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        subset_segmentation = sitk.BinaryThreshold(
            labeled_segmentation_image,
            lowerThreshold=segmentation_id,
            upperThreshold=segmentation_id,
        )
        signed_distance_map = sitk.SignedMaurerDistanceMap(
            subset_segmentation, squaredDistance=False, useImageSpacing=True
        )
        dilated_segmentation = signed_distance_map < dilation_mm
        label_shape_filter.Execute(dilated_segmentation)
        bounding_box = label_shape_filter.GetBoundingBox(1)
        box_start = dilated_segmentation.TransformIndexToPhysicalPoint(
            bounding_box[0 : int(len(bounding_box) / 2)]
        )
        box_end = dilated_segmentation.TransformIndexToPhysicalPoint(
            [
                x + sz
                for x, sz in zip(
                    bounding_box[0 : int(len(bounding_box) / 2)],
                    bounding_box[int(len(bounding_box) / 2) :],
                )
            ]
        )

        # crop using the indexes computed for imgB
        ct: sitk.Image = self.image
        if preprocess:
            ct = sitk.Clamp(ct, ct.GetPixelIDValue(), CT_AIR, CT_BONE)
            ct = sitk.Normalize(ct)
            ct = sitk.RescaleIntensity(ct, 0, 1)

        imgB_start_index = ct.TransformPhysicalPointToIndex(box_start)
        imgB_end_index = ct.TransformPhysicalPointToIndex(box_end)
        return sitk.Slice(image1=ct, start=imgB_start_index, stop=imgB_end_index)


class NRRDNodule(NoduleImage):
    """Class for loading nodules from NRRD"""

    @property
    def image(self) -> sitk.Image:
        return sitk.ReadImage(self.image_file_path)

    def nodule_segmentation_image(self) -> sitk.Image:
        nod_id = os.path.basename(self.image_file_path).split("nod")[1].split(".")[0]
        seg_file = f"/data/kaplinsp/test_nnunet/lung_{nod_id}.nii.gz"
        return sitk.ReadImage(seg_file)


class DICOMNodule(NoduleImage):
    """Class for loading an image of a nodule from DICOM"""

    @property
    def image(self) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(self.image_file_path)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        return image

    def nodule_segmentation_image(self) -> sitk.Image:
        nod_id = os.path.basename(self.image_file_path).split("_")[-1]
        seg_file = f"/data/kaplinsp/test_nnunet/lung_pd{nod_id}.nii.gz"
        return sitk.ReadImage(seg_file)


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
        # ids_to_exclude = set(["103", "128", "20", "26", "29", "69", "61"])
        # exclude_ids = read_pickle("/data/kaplinsp/exclude_r17.p")

        nodule_infos = []

        with open(coord_file, newline="") as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
            next(reader)
            for row in reader:
                nod_name = row[0]
                file_path = f"/data/etay/lung_hist_dat/original_dat_nrrds/nod{nod_name}.nrrd"
                center = get_coord_csv(row[4], row[5], row[6])
                if (int(nod_name), center) in EXCLUDE_NODULE_FILES:
                    logger.info(f"EXCLUDING: {nod_name}")
                    continue

                label = int(row[7])
                nodule_infos.append(
                    NoduleInfoTuple(label, nod_name, center, file_path, NRRDNodule)
                )

        return nodule_infos


class PrasadSampleGeneratoryStrategy(SampleGeneratorStrategy):
    def generate_nodule_info() -> List[NoduleInfoTuple]:
        coord_file = "/home/kaplinsp/ct_lung_class/ct_lung_class/annots_michelle_label.csv"
        nodule_infos = []

        # with open(
        #     "/home/kaplinsp/ct_lung_class/ct_lung_class/exclude_michelle.csv", "r"
        # ) as excludefile:
        #     exclude = [line.strip() for line in excludefile]
        # exclude += read_pickle("/data/kaplinsp/exclude_prasad.p")

        with open(coord_file) as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            next(reader)
            for row in reader:

                subject_id = row[3]
                try:
                    coords = ras_to_lps(np.array(literal_eval(row[8])))
                except SyntaxError:
                    logger.warning(f"No coordinates for {subject_id}")
                    continue

                if (int(row[1]), tuple(coords)) in EXCLUDE_PRASAD_IDS:
                    print(f"Skipping {subject_id}")
                    continue
                file_path = f"/data/kaplinsp/prasad_d/{subject_id}"
                nodule_infos.append(
                    NoduleInfoTuple(int(row[-1]), subject_id, coords, file_path, DICOMNodule)
                )

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
