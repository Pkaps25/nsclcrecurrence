from functools import cached_property
import itertools
from typing import Generator, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import getNoduleInfoList
from image import NoduleInfoTuple


class DataManager:

    def __init__(self, k_folds: int, val_ratio: float, dataset_names: List[str]) -> None:
        self.k_folds = k_folds
        self.val_ratio = val_ratio
        self.dataset_names = list(dataset_names)

    @cached_property
    def nodule_info_list(self) -> List[NoduleInfoTuple]:
        return getNoduleInfoList(self.dataset_names)

    def split(self) -> Generator[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]], None, None]:
        nods = [[nod] for nod in self.nodule_info_list]
        labels = [nod.is_nodule for nod in self.nodule_info_list]
        if self.k_folds == 1:
            x_train, x_test = train_test_split(nods, test_size=self.val_ratio, stratify=labels)
            yield list(itertools.chain(*x_train)), list(itertools.chain(*x_test))
        else:
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
            splits = kfold.split(nods, labels)
            for train_index, test_index in splits:
                yield [nods[i][0] for i in train_index], [nods[i][0] for i in test_index]
