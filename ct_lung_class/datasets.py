import copy
import functools
import math
import random
from typing import List, Tuple

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


raw_cache = getCache("part2ch11_raw")


# NoduleInfoTuple = namedtuple(
#     "NoduleInfoTuple",
#     ["isNodule_bool", "nod_id", "center_xyz", "file_path"]
# )

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int]


def getNoduleInfoList() -> List[NoduleInfoTuple]:
    generator = NoduleInfoGenerator()
    generator.add_strategies(PrasadSampleGeneratoryStrategy, R17SampleGeneratorStrategy)
    return generator.generate_all_samples()


# class CTImage:
#     def __init__(self, nod_id, nrrd_path):
#         # nrrd_path = os.path.join(DATA_DIR, f"nod{nod_id}.nrrd")
#         # nrrd_path = os.path.join("/data/kaplinsp/prasad_d/", f"{nod_id}.nrrd")

#         reader = sitk.ImageFileReader()
#         reader.SetImageIO("NrrdImageIO")
#         reader.SetFileName(nrrd_path)
#         ct_nrrd = reader.Execute()
#         ct_a = np.array(sitk.GetArrayFromImage(ct_nrrd), dtype=np.float32)

#         # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
#         # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
#         # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
#         # The upper bound nukes any weird hotspots and clamps bone down

#         ct_a.clip(-1350, 150, ct_a)

#         self.nod_id = nod_id
#         self.hu_a = ct_a

#         self.origin_xyz = XyzTuple(*ct_nrrd.GetOrigin())
#         self.vxSize_xyz = XyzTuple(*ct_nrrd.GetSpacing())
#         self.direction_a = np.array(ct_nrrd.GetDirection()).reshape(3, 3)

#     def getRawNodule(self, center_xyz: XyzTuple, width_irc: IrcTuple) -> Tuple[np.array, IrcTuple]:
#         center_irc = xyz2irc(
#             center_xyz,
#             self.origin_xyz,
#             self.vxSize_xyz,
#             self.direction_a,
#         )

#         slice_list = []
#         for axis, center_val in enumerate(center_irc):
#             start_ndx = int(round(center_val - width_irc[axis] / 2))
#             end_ndx = int(start_ndx + width_irc[axis])

#             # print(self.nod_id)
#             assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
#                 [self.nod_id, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis]
#             )
#             if start_ndx < 0:
#                 start_ndx = 0
#                 end_ndx = int(width_irc[axis])

#             if end_ndx > self.hu_a.shape[axis]:
#                 end_ndx = self.hu_a.shape[axis]
#                 start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

#             slice_list.append(slice(start_ndx, end_ndx))

#         return self.hu_a[tuple(slice_list)], center_irc


# @functools.lru_cache(1, typed=True)
# def getCt(nod_id: int, nod_path) -> CTImage:
#     return CTImage(nod_id, nod_path)


@raw_cache.memoize(typed=True)
def getCtRawNodule(
    noduleInfoTup: NoduleInfoTuple, width_irc: Coord3D
) -> np.array:
    ct: NoduleImage = noduleInfoTup.image_type(noduleInfoTup.file_path, noduleInfoTup.center_lps)
    return ct.nodule_slice(box_dim=width_irc, segmented=False)


def getCtAugmentedNodule(
    augmentation_dict: dict,
    noduleInfoTup: NoduleInfoTuple,
    width_irc: IrcTuple,
    use_cache: bool = True,
) -> np.array:
    if use_cache:
        ct_chunk = getCtRawNodule(noduleInfoTup, width_irc)
    else:
        ct: NoduleImage = noduleInfoTup.image_type(noduleInfoTup.file_path, noduleInfoTup.center_lps)
        ct_chunk = ct.nodule_slice(box_dim=width_irc, segmented=False)

    
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
        val_stride=0,
        isValSet_bool=None,
        nod_id=None,
        ratio_int=0,
        sortby_str="random",
        augmentation_dict=None,
        use_cache=False,
    ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.use_cache = use_cache
        self.noduleInfo_list = getNoduleInfoList()

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

        width_irc = (64, 64, 64)

        if self.augmentation_dict:
            nodule_t = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            nodule_a = getCtRawNodule(
                noduleInfo_tup,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
        else:
            ct: NoduleImage = noduleInfo_tup.image_type(noduleInfo_tup.file_path, noduleInfo_tup.center_lps)
            nodule_a = ct.nodule_slice(box_dim=width_irc, segmented=False)

            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not noduleInfo_tup.is_nodule, noduleInfo_tup.is_nodule],
            dtype=torch.long,
        )

        return nodule_t, pos_t, noduleInfo_tup.nod_id 
