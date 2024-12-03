from abc import ABC, abstractmethod
from ast import literal_eval
import csv
from dataclasses import dataclass
import logging
import os
import pickle
import random
from typing import List, Tuple, Union

import pandas as pd
import SimpleITK as sitk
import numpy as np

from constants import EXCLUDE_NODULE_FILES, CT_AIR, CT_BONE, EXCLUDE_PRASAD_IDS


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

    import SimpleITK as sitk

    def extract_fixed_size_nodule(self, box_size: list, preprocess: bool = True):
        """
        Extracts a bounding box from the CT and segmentation images, ensuring spatial alignment.

        Parameters:
            box_size (list): The size of the box in physical units (e.g., mm) for
              each dimension [x, y, z].
            preprocess (bool): Whether to preprocess the CT image
              (clamping, normalizing, rescaling).

        Returns:
            tuple: Extracted CT bounding box and segmentation bounding box as sitk.Images.
        """
        # Work on the main CT image
        ct = self.image
        if preprocess:
            ct = sitk.Clamp(ct, ct.GetPixelIDValue(), CT_AIR, CT_BONE)
            ct = sitk.Normalize(ct)
            ct = sitk.RescaleIntensity(ct, 0, 1)

        def get_box_from_physical_point(image, point):

            spacing = image.GetSpacing()
            center = image.TransformPhysicalPointToIndex(point)

            # Calculate half the box size in pixels for each dimension
            half_box_size_pixels = [int((box_size[i] / spacing[i]) / 2) for i in range(3)]

            # Calculate the starting and ending indices for the box
            start = [int(center[i] - half_box_size_pixels[i]) for i in range(3)]
            end = [int(center[i] + half_box_size_pixels[i]) for i in range(3)]

            # Ensure the indices are within the image bounds
            start = [max(0, min(start[i], image.GetSize()[i] - 1)) for i in range(3)]
            end = [max(0, min(end[i], image.GetSize()[i] - 1)) for i in range(3)]

            # Extract the region
            box = image[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
            return box

        ct_box = get_box_from_physical_point(ct, self.center_lps)
        # seg_box = get_box_from_physical_point(self.nodule_segmentation_image(), self.center_lps)

        # return ct_box, seg_box
        return ct_box

    def extract_normal_lung_region(self, box_dim: int, preprocess: bool):
        segmentation_image = self.nodule_segmentation_image()
        normal_lung_mask = sitk.BinaryNot(segmentation_image)
        labeled_normal_lung = sitk.ConnectedComponent(normal_lung_mask)
        normal_lung_np = sitk.GetArrayFromImage(labeled_normal_lung)
        normal_indices = np.argwhere(normal_lung_np > 0)

        if len(normal_indices) == 0:
            raise ValueError(
                f"No normal lung regions found in the segmentation mask.: {self.file_path}"
            )

        random_index = random.choice(normal_indices)
        center_index = np.array(random_index[::-1])  # Convert from (z, y, x) to (x, y, z)

        # Compute the bounding box start and end indices
        half_box_size = np.array(box_dim) // 2
        start_index = center_index - half_box_size
        end_index = center_index + half_box_size

        # Ensure bounding box stays within the image bounds
        image_size = np.array(normal_lung_mask.GetSize())
        start_index = np.clip(start_index, 0, image_size - 1)
        end_index = np.clip(end_index, 0, image_size - 1)

        # Convert back to physical coordinates
        box_start = normal_lung_mask.TransformIndexToPhysicalPoint(start_index.tolist())
        box_end = normal_lung_mask.TransformIndexToPhysicalPoint(end_index.tolist())

        # Step 3: Extract and preprocess the image region
        ct: sitk.Image = self.image
        if preprocess:
            ct = sitk.Clamp(ct, ct.GetPixelIDValue(), CT_AIR, CT_BONE)
            ct = sitk.Normalize(ct)
            ct = sitk.RescaleIntensity(ct, 0, 1)

        imgB_start_index = ct.TransformPhysicalPointToIndex(box_start)
        imgB_end_index = ct.TransformPhysicalPointToIndex(box_end)
        return sitk.Slice(image1=ct, start=imgB_start_index, stop=imgB_end_index)

    def get_connected_component_id_for_nodule(self, labeled_segmentation_image: sitk.Image) -> int:
        index = labeled_segmentation_image.TransformPhysicalPointToIndex(self.center_lps)
        index_box = [slice(x - 5, x + 5) for x in index]
        center_box = sitk.GetArrayFromImage(labeled_segmentation_image[index_box])
        segs = np.unique(center_box)
        return int(np.max(segs))

    def extract_bounding_box_nodule(
        self, preprocess: bool, dilation_mm: int, box_size: int
    ) -> sitk.Image:
        segmentation_image = self.nodule_segmentation_image()
        labeled_segmentation_image = sitk.ConnectedComponent(segmentation_image)
        segmentation_id = self.get_connected_component_id_for_nodule(labeled_segmentation_image)
        if segmentation_id == 0:
            return self.extract_fixed_size_nodule([box_size] * 3, True)

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
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(self.image_file_path)
        image = reader.Execute()
        return image

    def nodule_segmentation_image(self) -> sitk.Image:
        if "etay" in self.image_file_path:
            nod_id = os.path.basename(self.image_file_path).split("nod")[1].split(".")[0]
            seg_file = f"/data/kaplinsp/test_nnunet/lung_{nod_id}.nii.gz"
        else:
            nod_id = os.path.basename(self.image_file_path)
            seg_file = f"/data/kaplinsp/transformation_seg/{nod_id.replace('nrrd', 'nii.gz')}"

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


class SampleGeneratorStrategy(ABC):

    @abstractmethod
    def generate_nodule_info(self) -> List[NoduleInfoTuple]:
        pass


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


class SCLCSampleGenerator(SampleGeneratorStrategy):

    def generate_nodule_info() -> List[NoduleInfoTuple]:
        coord_file = "/home/kaplinsp/ct_lung_class/ct_lung_class/annotations_transformed.csv"

        exclude_paths = ["/data/kaplinsp/transformation/A114.nrrd"]
        coord_df = pd.read_csv(coord_file, index_col=False)
        coord_df = coord_df[(coord_df["label"] != 2) & (~coord_df["path"].isin(exclude_paths))]
        return list(
            coord_df.apply(
                lambda row: NoduleInfoTuple(
                    row["label"],
                    row["path"],
                    (row["x"], row["y"], row["z"]),
                    row["path"],
                    NRRDNodule,
                ),
                axis=1,
            ).to_numpy()
        )


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
