import copy
import math
import random
from typing import List, Optional, Tuple
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise
import numpy as np

from image import (
    Coord3D,
    NoduleInfoGenerator,
    R17SampleGeneratorStrategy,
    NoduleImage,
    NoduleInfoTuple,
    Slice3D,
    Image,
)
from util.disk import getCache
from util.logconf import logging
from util.util import IrcTuple

log = logging.getLogger(__name__)
log.setLevel(logging.CRITICAL)


image_cache = getCache("image_slices")
seg_cache = getCache("segmentations")
nodule_seg_cache = getCache("nodules")
bb_cache = getCache("box")

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int]


def getNoduleInfoList() -> List[NoduleInfoTuple]:
    generator = NoduleInfoGenerator()
    # generator.add_strategies(PrasadSampleGeneratoryStrategy)
    generator.add_strategies(R17SampleGeneratorStrategy)
    return generator.generate_all_samples()


def resample_image(input_image: sitk.Image, output_size: int) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    # new_spacing = old_size * old_spacing / new_size
    output_spacing = [
        original_size[i] * original_spacing[i] / output_size for i in range(len(original_size))
    ]
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize([output_size] * 3)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(input_image)


# @image_cache.memoize(typed=True)
@bb_cache.memoize(typed=True)
def getCtRawNodule(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    preprocess: bool,
    dilation: int,
    resample_size,
) -> Tuple[Image, Slice3D]:
    log.info(f"Slicing nodule from image for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    # return ct.nodule_slice(box_dim=width_irc, preprocess=preprocess)
    raw_nodule = ct.extract_bounding_box_nodule(preprocess=preprocess, dilation_mm=dilation)
    resampled = resample_image(raw_nodule, resample_size)
    return sitk.GetArrayFromImage(resampled)


@seg_cache.memoize(typed=True)
def get_segmentation(nodule_file_path: str, image_type: NoduleImage, center_lps: Coord3D):
    log.info(f"Segmenting lung for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    return ct.lung_segmentation()


@nodule_seg_cache.memoize(typed=True)
def get_nodule_segmentation(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    dilation: Optional[int] = None,
) -> Image:
    log.info(f"Segmenting nodule for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    segmentation = ct.nodule_segmentation_image()
    if dilation is not None:
        signed_distance_map = sitk.SignedMaurerDistanceMap(
            segmentation, squaredDistance=False, useImageSpacing=True
        )
        segmentation = signed_distance_map < dilation

    return sitk.GetArrayFromImage(segmentation)


def slice_and_pad_segmentation(
    seg_type: str,
    nodule_info_tup: NoduleInfoTuple,
    box_dim: Coord3D,
    slice_3d: Slice3D,
    dilation: Optional[int] = None,
):
    if seg_type == "lung":
        segmentation = get_segmentation(
            nodule_info_tup.file_path, nodule_info_tup.image_type, nodule_info_tup.center_lps
        )
    elif seg_type == "nodule":
        segmentation = get_nodule_segmentation(
            nodule_info_tup.file_path,
            nodule_info_tup.image_type,
            nodule_info_tup.center_lps,
            dilation,
        )
    sliced_seg = segmentation[slice_3d]
    pad_width = [(0, max(0, box_dim[2 - i] - sliced_seg.shape[i])) for i in range(3)]
    padded_arr = np.pad(sliced_seg, pad_width=pad_width, mode="constant", constant_values=0)
    return padded_arr


def getCtAugmentedNodule(
    augmentation_dict: dict,
    noduleInfoTup: NoduleInfoTuple,
    width_irc: IrcTuple,
    preprocess: bool,
    dilation: int,
    resample_size: int,
) -> Tuple[Image, Slice3D]:
    ct_chunk = getCtRawNodule(
        noduleInfoTup.file_path,
        noduleInfoTup.image_type,
        noduleInfoTup.center_lps,
        preprocess=preprocess,
        dilation=dilation,
        resample_size=resample_size,
    )
    rand_affine = RandAffine(
        mode=("bilinear"),
        prob=augmentation_dict["affine_prob"],
        translate_range=[augmentation_dict["translate"]] * 3,
        rotate_range=(np.pi / 6, np.pi / 6, np.pi / 4),
        scale_range=[augmentation_dict["scale"] * 3],
        padding_mode=augmentation_dict["padding"],
    )
    transform = Compose([rand_affine, RandFlip(), RandGaussianNoise()])
    ct_t = transform(ct_chunk).unsqueeze(0).to(torch.float32)
    return ct_t
    # return torch.tensor(ct_chunk).unsqueeze(0).to(torch.float32)
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if "offset" in augmentation_dict:
            offset_float = augmentation_dict["offset"]
            random_float = random.random() * 2 - 1
            transform_t[i, 3] = offset_float * random_float

        if "scale" in augmentation_dict:
            scale_float = augmentation_dict["scale"]
            random_float = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if "rotate" in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode="border",
        align_corners=False,
    ).to("cpu")

    if "noise" in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict["noise"]

        augmented_chunk += noise_t

    return augmented_chunk[0]


class NoduleDataset(Dataset):
    def __init__(
        self,
        nodule_info_list,
        dilate,
        resample,
        isValSet_bool=None,
        sortby_str="random",
        augmentation_dict=None,
    ):
        self.augmentation_dict = augmentation_dict
        self.noduleInfo_list = copy.copy(nodule_info_list)
        self.dilate = dilate
        self.resample = resample

        if sortby_str == "random":
            random.shuffle(self.noduleInfo_list)
        elif sortby_str == "nod_id":
            self.noduleInfo_list.sort(key=lambda x: x.file_path)
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

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

        width_irc = (50, 50, 50)

        if self.augmentation_dict:
            nodule_t = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup,
                width_irc,
                preprocess=True,
                dilation=self.dilate,
                resample_size=self.resample,
            )
        else:
            nodule_a = getCtRawNodule(
                noduleInfo_tup.file_path,
                noduleInfo_tup.image_type,
                noduleInfo_tup.center_lps,
                preprocess=True,
                dilation=self.dilate,
                resample_size=self.resample,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not noduleInfo_tup.is_nodule, noduleInfo_tup.is_nodule],
            dtype=torch.long,
        )
        assert not torch.any(torch.isnan(nodule_t)), noduleInfo_tup.file_path
        return nodule_t, pos_t, noduleInfo_tup.nod_id
