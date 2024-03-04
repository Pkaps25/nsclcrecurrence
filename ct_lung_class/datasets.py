import copy
import csv
import functools
import math
import os
import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

# import torchio as tio
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

# from transforms import RandomCrop
from util.disk import getCache
from util.logconf import logging
from util.util import IrcTuple, XyzTuple, xyz2irc

DATA_DIR = os.getenv("DATA_DIR")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


raw_cache = getCache("part2ch11_raw")


NoduleInfoTuple = namedtuple(
    "NoduleInfoTuple",
    "isNodule_bool, nod_id,center_xyz",
)

DatasetItem = Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]


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


@functools.lru_cache(1)
def getNoduleInfoList(requireOnDisk_bool=True) -> List[NoduleInfoTuple]:
    coord_file = "/home/kaplinsp/ct_lung_class/ct_lung_class/annotations.csv"
    # take out 103, 128, 20(2), 26, bc errors in CT slice skipping
    ids_to_exclude = set(["103", "128", "20", "26", "29", "69", "61"])

    nodule_infos = []

    with open(coord_file, newline="") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            nod_name = row[0]
            if nod_name in ids_to_exclude:
                log.info(f"EXCLUDING: {nod_name}")
                continue

            center = get_coord_csv(row[4], row[5], row[6])
            label = bool(int(row[7]))
            nodule_infos.append(
                NoduleInfoTuple(
                    label,
                    nod_name,
                    center,
                )
            )

    nodule_infos.sort(reverse=True)
    return nodule_infos


class CTImage:
    def __init__(self, nod_id: int):
        nrrd_path = os.path.join(DATA_DIR, f"nod{nod_id}.nrrd")

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(nrrd_path)
        ct_nrrd = reader.Execute()
        ct_a = np.array(sitk.GetArrayFromImage(ct_nrrd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down

        ct_a.clip(-1350, 150, ct_a)

        self.nod_id = nod_id
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_nrrd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_nrrd.GetSpacing())
        self.direction_a = np.array(ct_nrrd.GetDirection()).reshape(3, 3)

    def getRawNodule(self, center_xyz: XyzTuple, width_irc: IrcTuple) -> Tuple[np.array, IrcTuple]:
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            # print(self.nod_id)
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.nod_id, center_xyz, self.origin_xyz,
                    self.vxSize_xyz, center_irc, axis]
            )
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        return self.hu_a[tuple(slice_list)], center_irc


@functools.lru_cache(1, typed=True)
def getCt(nod_id: int) -> CTImage:
    return CTImage(nod_id)


@raw_cache.memoize(typed=True)
def getCtRawNodule(
    nod_id: int, center_xyz: XyzTuple, width_irc: IrcTuple
) -> Tuple[np.array, IrcTuple]:
    ct = getCt(nod_id)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)
    return ct_chunk, center_irc


def getCtAugmentedNodule(
    augmentation_dict: dict,
    nod_id: int,
    center_xyz: XyzTuple,
    width_irc: IrcTuple,
    use_cache: bool = True,
) -> Tuple[np.array, IrcTuple]:
    if use_cache:
        ct_chunk, center_irc = getCtRawNodule(nod_id, center_xyz, width_irc)
    else:
        ct = getCt(nod_id)
        ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)

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

    return augmented_chunk[0], center_irc


class NoduleDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        isValSet_bool=None,
        nod_id=None,
        ratio_int=0,
        sortby_str="random",
        augmentation_dict=None,
        noduleInfo_list=None,
    ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if noduleInfo_list:
            self.noduleInfo_list = copy.copy(noduleInfo_list)
            self.use_cache = False
        else:
            self.noduleInfo_list = copy.copy(getNoduleInfoList())
            self.use_cache = True

        if nod_id:
            self.noduleInfo_list = [
                x for x in self.noduleInfo_list if x.nod_id == nod_id]

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
            self.noduleInfo_list.sort(key=lambda x: (x.nod_id, x.center_xyz))
        elif sortby_str == "label_and_size":
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [
            nt for nt in self.noduleInfo_list if not nt.isNodule_bool]

        self.pos_list = [nt for nt in self.noduleInfo_list if nt.isNodule_bool]

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
            nodule_t, center_irc = getCtAugmentedNodule(
                self.augmentation_dict,
                noduleInfo_tup.nod_id,
                noduleInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            nodule_a, center_irc = getCtRawNodule(
                noduleInfo_tup.nod_id,
                noduleInfo_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
            # nodule_t = preprocess(nodule_t)
        else:
            ct = getCt(noduleInfo_tup.nod_id)
            nodule_a, center_irc = ct.getRawNodule(
                noduleInfo_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
            # nodule_t = preprocess(nodule_t)

        pos_t = torch.tensor(
            [not noduleInfo_tup.isNodule_bool, noduleInfo_tup.isNodule_bool],
            dtype=torch.long,
        )

        # nodule_t = tio.RescaleIntensity(percentiles=(0.5,99.5))(nodule_t)
        # nodule_t = tio.ZNormalization()(nodule_t)
        return nodule_t, pos_t, noduleInfo_tup.nod_id, torch.tensor(center_irc)
