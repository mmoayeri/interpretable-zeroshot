from abc import ABC, abstractmethod

class Attributer(ABC):
    def __init__(self, dset):
        self.dset = dset

    def infer_attrs(self, cls_ind: int) -> List[str]:
        # for class in dset given by cls_ind, return list of attributes

    def infer_all_attrs(self):
        pass


# Decision: bake into dataset