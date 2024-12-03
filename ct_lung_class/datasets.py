import copy
from typing import List, Optional, Sequence, Tuple
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise
import numpy as np

from image import (
    Coord3D,
    NoduleInfoGenerator,
    PrasadSampleGeneratoryStrategy,
    R17SampleGeneratorStrategy,
    NoduleImage,
    NoduleInfoTuple,
    SCLCSampleGenerator,
    Slice3D,
    Image,
)
from util.disk import getCache
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int]


def getNoduleInfoList(
    dataset_names: Optional[List[str]] = ["r17", "prasad"]
) -> List[NoduleInfoTuple]:
    mapping = {
        "r17": R17SampleGeneratorStrategy,
        "prasad": PrasadSampleGeneratoryStrategy,
        "sclc": SCLCSampleGenerator,
    }
    generator = NoduleInfoGenerator()
    generator.add_strategies(*[mapping[name] for name in dataset_names])
    samples = generator.generate_all_samples()
    return samples


def resample_image(input_image: sitk.Image, output_size: Tuple[int, int, int]) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    # new_spacing = old_size * old_spacing / new_size
    output_spacing = [
        original_size[i] * original_spacing[i] / output_size[i] for i in range(len(original_size))
    ]
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(input_image)


@ (getCache("fixedsize")).memoize(typed=True)
def get_fixed_size_nodule(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    resample_size: int,
    box_size: List[int],
) -> torch.Tensor:
    log.info(f"Slicing nodule from image for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    raw_nodule = ct.extract_fixed_size_nodule(box_size, True)
    resampled_nodule = resample_image(raw_nodule, resample_size)
    nodule_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled_nodule)).unsqueeze(0)
    return nodule_tensor.to(torch.float32)


@ (getCache("box")).memoize(typed=True)
def getCtRawNodule(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    preprocess: bool,
    dilation: int,
    resample_size: int,
    box_size: List[int],
) -> torch.Tensor:
    log.info(f"Slicing nodule from image for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    raw_nodule = ct.extract_bounding_box_nodule(
        preprocess=preprocess, dilation_mm=dilation, box_size=box_size
    )
    resampled = resample_image(raw_nodule, resample_size)
    return torch.from_numpy(sitk.GetArrayFromImage(resampled)).to(torch.float32).unsqueeze(0)


def getCtAugmentedNodule(
    augmentation_dict: dict,
    noduleInfoTup: NoduleInfoTuple,
    preprocess: bool,
    dilation: int,
    resample_size: Tuple[int, int, int],
    box_size: List[int],
) -> Tuple[Image, Slice3D]:
    ct_chunk = getCtRawNodule(
        noduleInfoTup.file_path,
        noduleInfoTup.image_type,
        noduleInfoTup.center_lps,
        preprocess=preprocess,
        dilation=dilation,
        resample_size=resample_size,
        box_size=box_size,
    )
    rand_affine = RandAffine(
        mode=("bilinear"),
        prob=augmentation_dict["affine_prob"],
        translate_range=[augmentation_dict["translate"]] * 3,
        rotate_range=(np.pi / 2, np.pi / 2),
        scale_range=[augmentation_dict["scale"] * 3],
        padding_mode=augmentation_dict["padding"],
    )
    transform = Compose([rand_affine, RandFlip(0.5), RandGaussianNoise(0.2)])
    ct_t = transform(ct_chunk).to(torch.float32)
    return ct_t


class NoduleDataset(Dataset):
    def __init__(
        self,
        nodule_info_list,
        dilate,
        resample,
        box_size,
        isValSet_bool=None,
        augmentation_dict=None,
    ):
        self.augmentation_dict = augmentation_dict
        self.noduleInfo_list = copy.copy(nodule_info_list)
        self.dilate = dilate
        self.resample = resample
        self.box_size = box_size

        self.negative_list = [nt for nt in self.noduleInfo_list if not nt.is_nodule]

        self.pos_list = [nt for nt in self.noduleInfo_list if nt.is_nodule]

        log.info(
            "{!r}: {} {} samples".format(
                self,
                len(self.noduleInfo_list),
                "validation" if isValSet_bool else "training",
            )
        )

    def shuffleSamples(self):
        pass

    def __len__(self):
        return len(self.noduleInfo_list)

    def __getitem__(self, ndx) -> DatasetItem:
        noduleInfo_tup = self.noduleInfo_list[ndx]

        if self.augmentation_dict:
            nodule_t = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup,
                preprocess=True,
                dilation=self.dilate,
                resample_size=self.resample,
                box_size=self.box_size,
            )
        else:
            nodule_t = getCtRawNodule(
                noduleInfo_tup.file_path,
                noduleInfo_tup.image_type,
                noduleInfo_tup.center_lps,
                preprocess=True,
                dilation=self.dilate,
                resample_size=self.resample,
                box_size=self.box_size,
            )

        assert not torch.any(torch.isnan(nodule_t)) and torch.all(
            torch.isfinite(nodule_t)
        ), noduleInfo_tup.file_path
        return nodule_t, noduleInfo_tup.is_nodule


normal_cache = getCache("normallung")


@normal_cache.memoize(typed=True)
def random_space_in_image(nodule_info_tuple, resample_size):
    logging.info(f"Slicing normal lung for {nodule_info_tuple.file_path}")
    nodule = nodule_info_tuple.image_type(
        nodule_info_tuple.file_path, nodule_info_tuple.center_lps
    )
    region = nodule.extract_normal_lung_region(box_dim=30, preprocess=True)
    return sitk.GetArrayFromImage(resample_image(region, resample_size))
