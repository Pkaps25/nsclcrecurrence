import copy
import csv
import functools
import math
import random
from collections import namedtuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from util.disk import getCache
from util.logconf import logging
from util.util import XyzTuple, xyz2irc

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache("part2ch11_raw")

# CandidateInfoTuple = namedtuple(
#   'CandidateInfoTuple',
#  'isNodule_bool, diameter_mm, series_uid, center_xyz',
# )
NoduleInfoTuple = namedtuple(
    "NoduleInfoTuple",
    "isNodule_bool, nod_id,center_xyz",
)


def get_coord(filename):
    with open(filename, newline="") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
        for row in reader:
            if "#" in row[0]:
                continue
            else:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
        center = tuple([x, y, z])
    return center


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
def getNoduleInfoList(requireOnDisk_bool=True):
    # nod_list=glob.glob('/home/zive/lung_hist_dat/*')
    coord_file = "/home/kaplinsp/ct_lung_class/ct_lung_class/annots_forpy.csv"  # take out 103, 128, 20(2), 26, bc errors in CT slice skipping
    # take out 103, 128, 20(2), 26, bc errors in CT slice skipping
    ids_to_exclude = set(["103", "128", "20", "26", "29", "69", "61"])

    noduleInfo_list = []

    with open(coord_file, newline="") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
        fields = next(reader)
        for row in reader:
            nod_name = row[0]
            if nod_name in ids_to_exclude:
                print("EXCLUDING: ", nod_name)
                continue
            center = get_coord_csv(row[4], row[5], row[6])
            label = bool(int(row[7]))
            noduleInfo_list.append(
                NoduleInfoTuple(
                    label,
                    nod_name,
                    center,
                )
            )

    noduleInfo_list.sort(reverse=True)
    return noduleInfo_list


class Ct:
    def __init__(self, nod_id):
        nrrd_path = f"/data/etay/lung_hist_dat/original_dat_nrrds/nod{nod_id}.nrrd"

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(nrrd_path)
        ct_nrrd = reader.Execute()
        ct_a = np.array(sitk.GetArrayFromImage(ct_nrrd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down

        # ct_a.clip(-1000, 1000, ct_a)
        ct_a.clip(-1350, 150, ct_a)

        # self.series_uid = series_uid
        self.nod_id = nod_id
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_nrrd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_nrrd.GetSpacing())
        self.direction_a = np.array(ct_nrrd.GetDirection()).reshape(3, 3)

    def getRawNodule(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        # print(self.nod_id)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            # print(self.nod_id)
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.nod_id, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis]
            )
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(nod_id):
    return Ct(nod_id)


@raw_cache.memoize(typed=True)
def getCtRawNodule(nod_id, center_xyz, width_irc):
    ct = getCt(nod_id)
    # print(nod_id)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)
    return ct_chunk, center_irc


def getCtAugmentedNodule(
    augmentation_dict, nod_id, center_xyz, width_irc, use_cache=True
):
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
        # self.noduleInfo_list = copy.copy(getNoduleInfoList())
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
                x for x in self.noduleInfo_list if x.nod_id == nod_id
            ]

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

        self.negative_list = [nt for nt in self.noduleInfo_list if not nt.isNodule_bool]

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

    def __getitem__(self, ndx):
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

        # width_irc = (32, 48, 48)
        # width_irc=(64,64,64)
        # width_irc=(60,60,60)
        width_irc = (64, 64, 64)
        # width_irc=(30,30,30)
        # width_irc=(50,50,50)

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

        else:
            ct = getCt(noduleInfo_tup.nod_id)
            nodule_a, center_irc = ct.getRawNodule(
                noduleInfo_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not noduleInfo_tup.isNodule_bool, noduleInfo_tup.isNodule_bool],
            dtype=torch.long,
        )
        return nodule_t, pos_t, noduleInfo_tup.nod_id, torch.tensor(center_irc)
