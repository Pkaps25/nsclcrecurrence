from ast import literal_eval
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from image import ras_to_lps


def exclude_non_segmented_nodule(nod_id, center):
    seg_file = f"/data/kaplinsp/test_nnunet/lung_pd{nod_id}.nii.gz"
    if not os.path.exists(seg_file):
        return True
    seg = sitk.ReadImage(seg_file)
    labeled_image = sitk.ConnectedComponent(seg)
    index = seg.TransformPhysicalPointToIndex(center)
    index_box = [slice(x - 10, x + 10) for x in index]
    center_box = sitk.GetArrayFromImage(labeled_image[index_box])
    segs = np.unique(center_box)
    # extract the segmentation for the nodule given center
    return len(segs) == 0 or len(segs) > 2 or (len(segs) == 1 and segs[0] == 0)


annots = pd.read_csv("annots_michelle.csv")
exclude = []
for nod_id, center in annots[["PT #", "Coordinates "]].values:
    try:
        center_parsed = ras_to_lps(np.array(literal_eval(center)))
    except:  # noqa E722
        exclude.append((nod_id, ""))
        continue
    result = exclude_non_segmented_nodule(nod_id, center_parsed)
    if result:
        exclude.append((nod_id, tuple(center_parsed)))
print(exclude)
