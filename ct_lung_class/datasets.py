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

# from transforms import RandomCrop
from datatsets_peter import (
    Coord3D, NoduleInfoGenerator, PrasadSampleGeneratoryStrategy, 
    R17SampleGeneratorStrategy, NoduleImage, NoduleInfoTuple
)
from util.disk import getCache
from util.logconf import logging
from util.util import IrcTuple 

# DATA_DIR = os.getenv("DATA_DIR")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


image_cache = getCache("image_slices")
seg_cache = getCache("segmentations")

# NoduleInfoTuple = namedtuple(
#     "NoduleInfoTuple",
#     ["isNodule_bool", "nod_id", "center_xyz", "file_path"]
# )

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int]


def getNoduleInfoList() -> List[NoduleInfoTuple]:
    generator = NoduleInfoGenerator()
    generator.add_strategies(PrasadSampleGeneratoryStrategy, R17SampleGeneratorStrategy)
    # generator.add_strategies(R17SampleGeneratorStrategy)
    return generator.generate_all_samples()



@image_cache.memoize(typed=True)
def getCtRawNodule(
    nodule_file_path: str,
    image_type: NoduleImage,
    center_lps: Coord3D,
    width_irc: Coord3D,
) -> Tuple[np.array, Tuple[slice,slice,slice]]:
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    return ct.nodule_slice(box_dim=width_irc)


@seg_cache.memoize(typed=True)
def get_segmentation(nodule_file_path: str, image_type: NoduleImage, center_lps: Coord3D):
    log.info(f"Segmenting nodule for {nodule_file_path}")
    ct: NoduleImage = image_type(nodule_file_path, center_lps)
    return ct.lung_segmentation()

def slice_and_pad_segmentation(nodule_info_tup: NoduleInfoTuple, box_dim: Coord3D, slice_3d: Tuple[slice,slice,slice]):
    segmentation = get_segmentation(nodule_info_tup.file_path, nodule_info_tup.image_type, nodule_info_tup.center_lps)
    sliced_seg = segmentation[slice_3d]
    pad_width = [(0, max(0, box_dim[2-i] - sliced_seg.shape[i])) for i in range(3)]
    padded_arr = np.pad(sliced_seg, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_arr
    
def preprocess(image: np.array) -> np.array:
    pass

def getCtAugmentedNodule(
    augmentation_dict: dict,
    noduleInfoTup: NoduleInfoTuple,
    width_irc: IrcTuple,
    use_cache: bool = True,
) -> np.array:
    if use_cache:
        ct_chunk, slice_3d = getCtRawNodule(noduleInfoTup.file_path, noduleInfoTup.image_type, noduleInfoTup.center_lps, width_irc)
    else:
        ct: NoduleImage = noduleInfoTup.image_type(noduleInfoTup.file_path, noduleInfoTup.center_lps)
        ct_chunk, slice_3d = ct.nodule_slice(box_dim=width_irc)

    
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
        val_stride=0,
        isValSet_bool=None,
        nod_id=None,
        ratio_int=0,
        sortby_str="random",
        augmentation_dict=None,
        use_cache=True,
        segmented=True,
    ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.use_cache = use_cache
        self.noduleInfo_list = copy.copy(getNoduleInfoList())
        self.segmented = segmented

        if nod_id:
            self.noduleInfo_list = [x for x in self.noduleInfo_list if x.nod_id == nod_id]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.noduleInfo_list = self.noduleInfo_list[::val_stride]
            assert self.noduleInfo_list
        elif val_stride > 0:
            del self.noduleInfo_list[::val_stride]
            assert self.noduleInfo_list

        if sortby_str == "random":
            random.shuffle(self.noduleInfo_list)
        elif sortby_str == "nod_id":
            # self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
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
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        return len(self.noduleInfo_list)
    

    def __getitem__(self, ndx) -> DatasetItem:
        noduleInfo_tup: NoduleInfoTuple = None
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)
            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                noduleInfo_tup = self.noduleInfo_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                noduleInfo_tup = self.pos_list[pos_ndx]
        else:
            noduleInfo_tup = self.noduleInfo_list[ndx]

        width_irc = (40, 40, 30)

        if self.augmentation_dict:
            nodule_t, slice_3d = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            nodule_a, slice_3d = getCtRawNodule(
                noduleInfo_tup.file_path,
                noduleInfo_tup.image_type,
                noduleInfo_tup.center_lps,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
        else:
            ct: NoduleImage = noduleInfo_tup.image_type(noduleInfo_tup.file_path, noduleInfo_tup.center_lps)
            nodule_a, slice_3d = ct.nodule_slice(box_dim=width_irc)

            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not noduleInfo_tup.is_nodule, noduleInfo_tup.is_nodule],
            dtype=torch.long,
        )
        segmentation = slice_and_pad_segmentation(noduleInfo_tup, width_irc, slice_3d)
        segmentation_t = torch.from_numpy(segmentation).unsqueeze(0)
        nodule_t_segmented = torch.cat([nodule_t, segmentation_t], dim=0)

        return nodule_t_segmented, pos_t, noduleInfo_tup.nod_id 
