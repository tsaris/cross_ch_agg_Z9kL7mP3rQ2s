# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

## equally split
def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        multi_dataset_training=False,
        data_par_size: int = 1,
        ddp_group = None,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.multi_dataset_training = multi_dataset_training
        self.data_par_size = data_par_size
        self.ddp_group = ddp_group

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        ## (10/24/23) jyc: worker info is None when num_workers = 0
        if worker_info is None:
            assert torch.distributed.is_initialized()
            ## (10/24/23) jyc: dummy worker_info 
            class dummy:
                num_workers = 1
                id = 0
            worker_info = dummy()
            iter_start = 0
            iter_end = len(self.file_list)

        else:
            if not torch.distributed.is_initialized():
                ddp_rank = 0
                self.data_par_size = 1
            else:

                ddp_rank = torch.distributed.get_rank(group = self.ddp_group)

            num_workers_per_ddp = worker_info.num_workers
            ## (10/23/23) jyc: we assume num_workers_per_ddp == 1 on Frontier
            assert num_workers_per_ddp == 1
            if self.multi_dataset_training:
                # num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", None))
                # ## (10/23/23) jyc: we assume more than a single node can process a dataset
                # num_nodes_per_dataset = int(os.environ.get("NUM_NODES_PER_DATASET", None))
                # num_gpus_per_node = int(world_size / num_nodes)
                # num_shards = num_workers_per_ddp * num_gpus_per_node * num_nodes_per_dataset
                ## (01/03/2024) jyc: For non-uniform multisets
                # num_shards = int(os.environ.get("NUM_RANKS_PER_DATASET", None))
                # rank = global_rank % num_shards
                gx = os.environ.get("DATASET_GROUP_LIST", None)
                group_list = list(map(lambda x: int(x), gx.split(":")))
                group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
                group_size = group_list[group_id]
                group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]
                num_shards = group_size
                rank = group_rank
            else:
                num_shards = num_workers_per_ddp * self.data_par_size
                rank = ddp_rank
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            # print ("%d: per_worker: %d"%(global_rank, per_worker))
            if per_worker == 0:
                self.file_list = (self.file_list * math.ceil(num_shards/len(self.file_list)))[:num_shards]
                per_worker = 1
            assert per_worker > 0
            ## (10/24/23) jyc: FIXME
            ## Add assert to ensure that all has the same number of files
            ## All has to have the same number of batches
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        # print ("num_shards:", num_shards)
        # print ("file_list:", self.file_list)
        use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE", 0))
        if use_ddstore:
            rx = list(nsplit(range(len(self.file_list)), num_shards))[rank]
            iter_start = rx[0]
            iter_end = rx[-1] + 1

        print ("global rank %d: ddp rank %d iter_start,iter_end = %d %d"%(torch.distributed.get_rank(), ddp_rank,iter_start, iter_end))
        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            print ("global rank %d: ddp rank %d NpyReader: %s"%(torch.distributed.get_rank(), ddp_rank, path))
            data = np.load(path)

            if use_ddstore:
                ## Add zero data if not exists
                k = data.files[0]
                shape = data[k].shape
                dtype = data[k].dtype
                zeros = np.zeros(shape, dtype=dtype)

                rtn = dict()
                for k in self.variables:
                    if k in data:
                        rtn[k] = data[k]
                    else:
                        rtn[k] = zeros
                yield rtn, self.variables, self.out_variables
            else:
                for k in self.variables:
                    if k not in data.files:
                        print ("No variable in data:", k)
                yield {k: data[k] for k in self.variables}, self.variables, self.out_variables



class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, max_predict_range: int = 6, random_lead_time: bool = False, hrs_each_step: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.random_lead_time = random_lead_time
        self.hrs_each_step = hrs_each_step

    def __iter__(self):
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x[: -self.max_predict_range]  # N, C, H, W

            if self.random_lead_time:
                predict_ranges = torch.randint(low=1, high=self.max_predict_range, size=(inputs.shape[0],))
            else:
                predict_ranges = torch.ones(inputs.shape[0]).to(torch.long) * self.max_predict_range
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(inputs.dtype)
            output_ids = torch.arange(inputs.shape[0]) + predict_ranges
            outputs = y[output_ids]

            yield inputs, outputs, lead_times, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info

    def __iter__(self):
        for (inp, out, lead_times, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables, self.region_info
                else:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []

        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

