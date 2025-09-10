from functools import cached_property
import itertools
from typing import Generator, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import getNoduleInfoList
from image import NoduleInfoTuple


from functools import cached_property
from typing import Generator, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import getNoduleInfoList
from image import NoduleInfoTuple


class DataManager:
    def __init__(self, k_folds: int, val_ratio, test_ratio, dataset_names: List[str]) -> None:
        self.k_folds = k_folds
        self.dataset_names = dataset_names

    @cached_property
    def nodule_info_list(self) -> List[NoduleInfoTuple]:
        return getNoduleInfoList(self.dataset_names)

    def split(self) -> Generator[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple], List[NoduleInfoTuple]], None, None]:
        data = self.nodule_info_list
        labels = [nod.is_nodule for nod in data]

        # Step 1: Always split out test set (15%)
        trainval_data, test_data, trainval_labels, _ = train_test_split(
            data, labels, test_size=0.15, stratify=labels, random_state=42
        )

        if self.k_folds == 1:
            # Step 2a: Simple val split (15% of trainval → ~12.75% of original)
            train_data, val_data, _, _ = train_test_split(
                trainval_data, trainval_labels, test_size=0.1765, stratify=trainval_labels, random_state=42
            )
            # 0.1765 ≈ 15% of the 85% trainval set → results in ~70/15/15 split
            yield train_data, val_data, test_data

        else:
            # Step 2b: Cross-validation on trainval
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(trainval_data, trainval_labels):
                train_set = [trainval_data[i] for i in train_idx]
                val_set = [trainval_data[i] for i in val_idx]
                yield train_set, val_set, test_data




# class DataManager:

#     def __init__(self, k_folds: int, val_ratio: float, dataset_names: List[str]) -> None:
#         self.k_folds = k_folds
#         self.val_ratio = val_ratio
#         self.dataset_names = list(dataset_names)

#     @cached_property
#     def nodule_info_list(self) -> List[NoduleInfoTuple]:
#         return getNoduleInfoList(self.dataset_names)

#     def split(self) -> Generator[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]], None, None]:
#         nods = [[nod] for nod in self.nodule_info_list]
#         labels = [nod.is_nodule for nod in self.nodule_info_list]
#         if self.k_folds == 1:
#             x_train, x_test = train_test_split(nods, test_size=self.val_ratio, stratify=labels)
#             yield list(itertools.chain(*x_train)), list(itertools.chain(*x_test))
#         else:
#             kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
#             splits = kfold.split(nods, labels)
#             for train_index, test_index in splits:
#                 yield [nods[i][0] for i in train_index], [nods[i][0] for i in test_index]
