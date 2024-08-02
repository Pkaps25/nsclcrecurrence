import copy
import functools
import math
import random
from typing import List, Tuple
import sys

import numpy as np


# import torchio as tio
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import numpy.typing

# from transforms import RandomCrop
from datatsets_peter import (
    Coord3D, NoduleInfoGenerator, PrasadSampleGeneratoryStrategy, 
    R17SampleGeneratorStrategy, NoduleImage, NoduleInfoTuple, Slice3D, Image
)
from util.disk import getCache
from util.logconf import logging
from util.util import IrcTuple 

# DATA_DIR = os.getenv("DATA_DIR")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


image_cache = getCache("image_slices")
seg_cache = getCache("segmentations")
nodule_seg_cache = getCache("nodules")

# NoduleInfoTuple = namedtuple(
#     "NoduleInfoTuple",
#     ["isNodule_bool", "nod_id", "center_xyz", "file_path"]
# )

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int]


def getNoduleInfoList() -> List[NoduleInfoTuple]:
    generator = NoduleInfoGenerator()
    # generator.add_strategies(PrasadSampleGeneratoryStrategy, R17SampleGeneratorStrategy)
    generator.add_strategies(R17SampleGeneratorStrategy)
    return generator.generate_all_samples()

    
@image_cache.memoize(typed=True)
def getCtRawNodule(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    width_irc: Coord3D,
    preprocess: bool = False
) -> Tuple[Image, Slice3D]:
    ct = image_type(nodule_file_path, center_lps)
    return ct.nodule_slice(box_dim=width_irc, preprocess=preprocess)

@seg_cache.memoize(typed=True)
def get_segmentation(nodule_file_path: str, image_type: NoduleImage, center_lps: Coord3D):
    log.info(f"Segmenting lung for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    return ct.lung_segmentation()

@nodule_seg_cache.memoize(typed=True)
def get_nodule_segmentation(nodule_file_path: str, image_type: NoduleImage, center_lps: Coord3D):
    log.info(f"Segmenting nodule for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    return ct.nodule_segmentation()

def slice_and_pad_segmentation(seg_type: str, nodule_info_tup: NoduleInfoTuple, box_dim: Coord3D, slice_3d: Slice3D):
    if seg_type == "lung":
        segmentation = get_segmentation(nodule_info_tup.file_path, nodule_info_tup.image_type, nodule_info_tup.center_lps)
    elif seg_type == "nodule":
        segmentation = get_nodule_segmentation(nodule_info_tup.file_path, nodule_info_tup.image_type, nodule_info_tup.center_lps)
    sliced_seg = segmentation[slice_3d]
    pad_width = [(0, max(0, box_dim[2-i] - sliced_seg.shape[i])) for i in range(3)]
    padded_arr = np.pad(sliced_seg, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_arr

def getCtAugmentedNodule(
    augmentation_dict: dict,
    noduleInfoTup: NoduleInfoTuple,
    width_irc: IrcTuple,
    use_cache: bool = True,
    preprocess: bool = True,
) -> Tuple[Image, Slice3D]:
    if use_cache:
        ct_chunk, slice_3d = getCtRawNodule(noduleInfoTup.file_path, noduleInfoTup.image_type, noduleInfoTup.center_lps, width_irc, preprocess=preprocess)
    else:
        ct: NoduleImage = noduleInfoTup.image_type(noduleInfoTup.file_path, noduleInfoTup.center_lps)
        ct_chunk, slice_3d = ct.nodule_slice(box_dim=width_irc, preprocess=preprocess)

    
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

    return augmented_chunk[0], slice_3d


class NoduleDataset(Dataset):
    def __init__(
        self,
        nodule_info_list,
        isValSet_bool=None,
        sortby_str="random",
        augmentation_dict=None,
        use_cache=True,
        segmented=True,
    ):
        self.augmentation_dict = augmentation_dict
        self.use_cache = use_cache
        self.noduleInfo_list = copy.copy(nodule_info_list)
        self.segmented = segmented

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
        # TODO
        pass
        # if False:
        #     random.shuffle(self.negative_list)
        #     random.shuffle(self.pos_list)

    def __len__(self):
        return len(self.noduleInfo_list)
    

    def __getitem__(self, ndx) -> DatasetItem:
        noduleInfo_tup = self.noduleInfo_list[ndx]

        width_irc = (40, 40, 30)

        if self.augmentation_dict:
            nodule_t, slice_3d = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup,
                width_irc,
                self.use_cache,
                preprocess=False
            )
        elif self.use_cache:
            nodule_a, slice_3d = getCtRawNodule(
                noduleInfo_tup.file_path,
                noduleInfo_tup.image_type,
                noduleInfo_tup.center_lps,
                width_irc,
                preprocess=False
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
        else:
            ct: NoduleImage = noduleInfo_tup.image_type(noduleInfo_tup.file_path, noduleInfo_tup.center_lps)
            nodule_a, slice_3d = ct.nodule_slice(box_dim=width_irc, preprocess=False)

            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not noduleInfo_tup.is_nodule, noduleInfo_tup.is_nodule],
            dtype=torch.long,
        )
        # lung_segmentation = slice_and_pad_segmentation("lung", noduleInfo_tup, width_irc, slice_3d)
        nod_segmentation = slice_and_pad_segmentation("nodule", noduleInfo_tup, width_irc, slice_3d)
        # lung_segmentation_t = torch.from_numpy(lung_segmentation).unsqueeze(0)
        nod_segmentation_t = torch.from_numpy(nod_segmentation).unsqueeze(0)
        # nodule_t_segmented = torch.cat([nodule_t, lung_segmentation_t, nod_segmentation_t], dim=0)
        nodule_t_segmented = nodule_t * nod_segmentation_t

        return nodule_t_segmented, pos_t, noduleInfo_tup.nod_id 
