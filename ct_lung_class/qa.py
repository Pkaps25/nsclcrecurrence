import SimpleITK as sitk
import numpy as np


def exclude_non_segmented_nodule(nod_id, center):
    seg_file = f"/data/kaplinsp/test_nnunet/lung_{nod_id}.nii.gz"
    seg = sitk.ReadImage(seg_file)
    labeled_image = sitk.ConnectedComponent(seg)
    index = seg.TransformPhysicalPointToIndex(center)
    index_box = [slice(x - 10, x + 10) for x in index]
    center_box = sitk.GetArrayFromImage(labeled_image[index_box])
    segs = np.unique(center_box)
    # extract the segmentation for the nodule given center
    return len(segs) == 0 or len(segs) > 2 or (len(segs) == 1 and segs[0] == 0)
