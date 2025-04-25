import numpy as np

import torch
from torch.utils.data import Dataset
import pickle
import torch.distributed as dist

# from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

try:
    import pyddstore2 as dds
except ImportError:
    pass


class DistDataset(Dataset):
    """Distributed dataset class"""

    def __init__(
        self, data, label, ddstore_width=None, use_mq=False, role=1, mode=0
    ):
        super().__init__()

        self.dataset = list()
        self.label = label
        self.rank = dist.get_rank()
        self.comm_size = dist.get_world_size()
        print("init: rank,size,label =", self.rank, self.comm_size, label)
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        self.ddstore = dds.PyDDStore(self.ddstore_comm, use_mq=use_mq, role=role, mode=mode)

        ## register local data
        for x in data:
            self.dataset.append(x)
        total_ns = len(self.dataset)
        dist.all_reduce(total_ns, op=torch.distributed.ReduceOp.SUM)
        self.total_ns = total_ns

        print("[%d] DDStore: %d %d" %(self.rank, len(self.dataset), self.total_ns))

        self.ddstore.add(self.label, self.dataset)
        print("Init done.")

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    def get(self, idx):
        # print ("[%d] idx:"%self.rank, idx)
        data_object = self.ddstore.get(
            self.label, idx, decoder=lambda x: pickle.loads(x)
        )
        return data_object

    def __getitem__(self, idx):
        return self.get(idx)

