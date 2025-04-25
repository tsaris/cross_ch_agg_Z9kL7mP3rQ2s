# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

from climax.utils.data_utils import DEFAULT_VARIABLE_LIST
from climax.pretrain.distdataset import DistDataset

def collate_fn(batch):
    use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE",0))
    if use_ddstore:
        torch.distributed.Barrier()
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
    )





class NativePytorchDataModule(torch.nn.Module):
    """Native pytorch daata module for multi-source data.

    Args:
        dict_root_dirs (Dict): Dictionary of root directories for each source.
        dict_start_idx (Dict): Dictionary of start indices ratio (between 0.0 and 1.0) for each source.
        dict_end_idx (Dict): Dictionary of end indices ratio (between 0.0 and 1.0) for each source.
        dict_buffer_sizes (Dict): Dictionary of buffer sizes for each source.
        dict_in_variables (Dict): Dictionary of input variables for each source.
        dict_out_variables (Dict): Dictionary of output variables for each source.
        dict_max_predict_ranges (Dict, optional): Dictionary of maximum predict ranges for each source.
        dict_random_lead_time (Dict, optional): Dictionary of whether to use random lead time for each source.
        dict_hrs_each_step (Dict, optional): Dictionary of hours each step for each source.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
        data_par_size (int, optional): the size of a data parallel group
        ddp_group : the data parallel group
    """

    def __init__(
        self,
        dict_root_dirs: Dict,
        dict_start_idx: Dict,
        dict_end_idx: Dict,
        dict_buffer_sizes: Dict,
        dict_in_variables: Dict,
        dict_out_variables: Dict,
        dict_max_predict_ranges: Dict = {"mpi-esm": 28},
        dict_random_lead_time: Dict = {"mpi-esm": True},
        dict_hrs_each_step: Dict = {"mpi-esm": 6},
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_par_size: int = 1,
        ddp_group = None,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )



        ## (1/4/2024) jyc: subselect
        valid = dict_root_dirs.keys()
        dict_start_idx = dict((k, dict_start_idx[k]) for k in valid)
        dict_end_idx = dict((k, dict_end_idx[k]) for k in valid)
        dict_buffer_sizes = dict((k, dict_buffer_sizes[k]) for k in valid)
        dict_in_variables = dict((k, dict_in_variables[k]) for k in valid)
        dict_out_variables = dict((k, dict_out_variables[k]) for k in valid)
        dict_max_predict_ranges = dict((k, dict_max_predict_ranges[k]) for k in valid)
        dict_random_lead_time = dict((k, dict_random_lead_time[k]) for k in valid)
        dict_hrs_each_step = dict((k, dict_hrs_each_step[k]) for k in valid)
         
        def get_climatology(root_dir, partition="test", variables=None):
            path = os.path.join(root_dir, partition, "climatology.npz")
            clim_dict = np.load(path)
            clim = np.concatenate([clim_dict[var] for var in variables])
            clim = torch.from_numpy(clim)
            return clim


        def get_normalize(root_dir, variables=None):
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if var != "total_precipitation":
                    mean.append(normalize_mean[var])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            normalize_std = np.concatenate([normalize_std[var] for var in variables])
            tmp = transforms.Normalize(normalize_mean, normalize_std)
            return tmp.mean, tmp.std
        

        self.test_clim = get_climatology(dict_root_dirs['mpi-esm'], "test", dict_out_variables['mpi-esm'])
        self.output_transforms = get_normalize(dict_root_dirs['mpi-esm'], dict_out_variables['mpi-esm'])
            

        assert len(dict_root_dirs) <= data_par_size, "the number of data parallel GPUs (data_par_size) needs to be at least equal to the number of datasets. Try to increase data_par_size"

        print("len(dict_root_dirs)",len(dict_root_dirs))


        gx = os.environ.get("DATASET_GROUP_LIST", None)
        if gx is None:
            gx = ":".join(["%d"%(data_par_size//len(dict_root_dirs)),]*len(dict_root_dirs))
            os.environ["DATASET_GROUP_LIST"] = gx


        self.dict_root_dirs = dict_root_dirs
        self.dict_start_idx = dict_start_idx
        self.dict_end_idx = dict_end_idx
        self.dict_buffer_sizes = dict_buffer_sizes 
        self.dict_in_variables = dict_in_variables
        self.dict_out_variables = dict_out_variables
        self.dict_max_predict_ranges = dict_max_predict_ranges
        self.dict_random_lead_time = dict_random_lead_time
        self.dict_hrs_each_step = dict_hrs_each_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ddp_group = ddp_group
        self.data_par_size = data_par_size

        # (11/03/23) jyc: using default variable list
        use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE", 0))
        in_variables = {}
        for k, list_out in dict_in_variables.items():
            if list_out is not None:
                in_variables[k] = list_out
            else:
                in_variables[k] = DEFAULT_VARIABLE_LIST
            ## (12/27/23) jyc: filter out
            in_variables[k] = [ x for x in in_variables[k] if x in DEFAULT_VARIABLE_LIST ]
            if use_ddstore:
                in_variables[k] = DEFAULT_VARIABLE_LIST
        self.dict_in_variables = in_variables

        out_variables = {}
        for k, list_out in dict_out_variables.items():
            if list_out is not None:
                out_variables[k] = list_out
            else:
                out_variables[k] = dict_in_variables[k]
            ## (12/27/23) jyc: filter out
            out_variables[k] = [ x for x in out_variables[k] if x in DEFAULT_VARIABLE_LIST ]
            if use_ddstore:
                out_variables[k] = DEFAULT_VARIABLE_LIST
        self.dict_out_variables = out_variables

        self.dict_lister_trains = {
            k: list(dp.iter.FileLister(os.path.join(root_dir, "train"))) for k, root_dir in dict_root_dirs.items()
        }
        self.dict_lister_tests = {
            k: list(dp.iter.FileLister(os.path.join(root_dir, "test"))) for k, root_dir in dict_root_dirs.items()
        }
        
        self.train_dataset_args = {
            k: {
                "max_predict_range": dict_max_predict_ranges[k],
                "random_lead_time": dict_random_lead_time[k],
                "hrs_each_step": dict_hrs_each_step[k],
            }
            for k in dict_root_dirs.keys()
        }

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(self.dict_out_variables)

        self.dict_data_train: Optional[Dict] = None

    def get_normalize(self, dict_variables: Optional[Dict] = None):
        if dict_variables is None:
            dict_variables = self.dict_in_variables
        dict_transforms = {}
        for k in dict_variables.keys():
            root_dir = self.dict_root_dirs[k]
            variables = dict_variables[k]
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if (var != "total_precipitation") and (var in normalize_mean):
                    mean.append(normalize_mean[var])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)

            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            std = []
            for var in variables:
                if (var != "total_precipitation") and (var in normalize_std):
                    std.append(normalize_std[var])
                else:
                    std.append(np.array([1.0], dtype=np.float32))
            normalize_std = np.concatenate(std)

            # normalize_std = np.concatenate([normalize_std[var] for var in variables])
            dict_transforms[k] = transforms.Normalize(normalize_mean, normalize_std)
        return dict_transforms



    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(list(self.dict_root_dirs.values())[0], "lat.npy"))
        lon = np.load(os.path.join(list(self.dict_root_dirs.values())[0], "lon.npy"))
        return lat, lon



    def setup(self, stage: Optional[str] = None):
        use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE", 0))
        # load datasets only if they're not loaded already
        if not self.dict_data_train:
            ## (1/3/24) jyc: This is a temporary fix
            ## Make num of files per worker be same to ensure to have the same num of batches
            gx = os.environ.get("DATASET_GROUP_LIST", None)
            group_list = list(map(lambda x: int(x), gx.split(":")))

            #files per gpu list
            num_files = list()

            for i, k in enumerate(self.dict_lister_trains.keys()):
                lister_train = self.dict_lister_trains[k]
                files_per_gpu = (len(lister_train)-1)//group_list[i] + 1
                num_files.append(files_per_gpu)


            mx_files_per_gpu = max(num_files)


            dict_data_train = {}
            for i, k in enumerate(self.dict_lister_trains.keys()):
                lister_train = self.dict_lister_trains[k]
                m = mx_files_per_gpu * group_list[i]
                _lister_train = np.random.choice(lister_train, size=m, replace=True).tolist()
                if torch.distributed.get_rank()==0 or torch.distributed.get_rank()==(torch.distributed.get_world_size()-1):
                    print("rank ",torch.distributed.get_rank(),"lister_train reset:",k,len(lister_train), len(_lister_train),mx_files_per_gpu)

                lister_train = lister_train #_lister_train
                start_idx = self.dict_start_idx[k]
                end_idx = self.dict_end_idx[k]
                variables = self.dict_in_variables[k]
                out_variables = self.dict_out_variables[k]
                max_predict_range = self.dict_max_predict_ranges[k]
                random_lead_time = self.dict_random_lead_time[k]
                hrs_each_step = self.dict_hrs_each_step[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.dict_buffer_sizes[k]
                
                dict_data_train[k] = IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            lister_train,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            variables=variables,
                            out_variables=out_variables,
                            multi_dataset_training=True,
                            data_par_size = self.data_par_size,
                            ddp_group = self.ddp_group,
                        ),
                        max_predict_range=max_predict_range,
                        random_lead_time=random_lead_time,
                        hrs_each_step=hrs_each_step,
                    ),
                    transforms,
                    output_transforms,
                )                
                """
                dict_data_train[k] = ShuffleIterableDataset(
                    IndividualForecastDataIter(
                        Forecast(
                            NpyReader(
                                lister_train,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                out_variables=out_variables,
                                multi_dataset_training=True,
                                data_par_size = self.data_par_size,
                                ddp_group = self.ddp_group,
                            ),
                            max_predict_range=max_predict_range,
                            random_lead_time=random_lead_time,
                            hrs_each_step=hrs_each_step,
                        ),
                        transforms,
                        output_transforms,
                    ),
                    buffer_size,
                )
                """
            self.dict_data_train = dict_data_train


            num_files = list()
            for i, k in enumerate(self.dict_lister_tests.keys()):
                lister_test = self.dict_lister_tests[k]
                files_per_gpu = (len(lister_test)-1)//group_list[i] + 1
                num_files.append(files_per_gpu)


            mx_files_per_gpu = max(num_files)


            dict_data_test = {}
            for i, k in enumerate(self.dict_lister_tests.keys()):
                lister_test = self.dict_lister_tests[k]
                m = mx_files_per_gpu * group_list[i]
                _lister_test = np.random.choice(lister_test, size=m, replace=True).tolist()
                if torch.distributed.get_rank()==0 or torch.distributed.get_rank()==(torch.distributed.get_world_size()-1):
                    print("rank ",torch.distributed.get_rank(),"lister_test reset:",k,len(lister_test), len(_lister_test),mx_files_per_gpu)

                lister_test = lister_test #_lister_test
                start_idx = self.dict_start_idx[k]
                end_idx = self.dict_end_idx[k]
                variables = self.dict_in_variables[k]
                out_variables = self.dict_out_variables[k]
                max_predict_range = self.dict_max_predict_ranges[k]
                random_lead_time = self.dict_random_lead_time[k]
                hrs_each_step = self.dict_hrs_each_step[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.dict_buffer_sizes[k]
                dict_data_test[k] = IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            lister_test,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            variables=variables,
                            out_variables=out_variables,
                            multi_dataset_training=False,
                            data_par_size = self.data_par_size,
                            ddp_group = self.ddp_group,
                        ),
                        max_predict_range=max_predict_range,
                        random_lead_time=False,
                        hrs_each_step=hrs_each_step,
                    ),
                    transforms,
                    output_transforms,
                )                
                
                
            self.dict_data_test = dict_data_test





    def train_dataloader(self):
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
            
        assert torch.distributed.is_initialized()

        ddp_rank = torch.distributed.get_rank(group=self.ddp_group)
        gx = os.environ.get("DATASET_GROUP_LIST", None)
        group_list = list(map(lambda x: int(x), gx.split(":")))

        assert self.data_par_size == sum(group_list), "data_par_size, group_list: %d %d"%(self.data_par_size, sum(group_list))
        group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
        group_size = group_list[group_id]
        group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]

        for idx, k in enumerate(self.dict_data_train.keys()):
            if idx == group_id:
                data_train = self.dict_data_train[k]
                break

        use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE", 0))


        print("use_ddstore is :",use_ddstore,flush=True)

        if use_ddstore:
            opt = {
                "use_mq": 0,
                "role": 0,
                "mode": 0,
            }
            trainset = DistDataset(data_train, "trainset", **opt)
            sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            train_loader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=True,
                sampler=sampler,
                collate_fn=collate_fn,
            )
    
            return train_loader
        else:
            # This assumes that the number of datapoints are going to be the same for all datasets
            return DataLoader(
                data_train,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )

    def test_dataloader(self):
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
            
        assert torch.distributed.is_initialized()

        ddp_rank = torch.distributed.get_rank(group=self.ddp_group)
        gx = os.environ.get("DATASET_GROUP_LIST", None)
        group_list = list(map(lambda x: int(x), gx.split(":")))

        assert self.data_par_size == sum(group_list), "data_par_size, group_list: %d %d"%(self.data_par_size, sum(group_list))
        group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
        group_size = group_list[group_id]
        group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]

        for idx, k in enumerate(self.dict_data_test.keys()):
            if idx == group_id:
                data_test = self.dict_data_test[k]
                break

        use_ddstore = int(os.environ.get("CLIMAX_USE_DDSTORE", 0))


        print("use_ddstore is :",use_ddstore,flush=True)

        if use_ddstore:
            opt = {
                "use_mq": 0,
                "role": 0,
                "mode": 0,
            }
            testset = DistDataset(data_test, "testset", **opt)
            sampler = torch.utils.data.distributed.DistributedSampler(testset)
            test_loader = torch.utils.data.DataLoader(
                testset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=True,
                sampler=sampler,
                collate_fn=collate_fn,
            )
    
            return test_loader
        else:
            # This assumes that the number of datapoints are going to be the same for all datasets
            return DataLoader(
                data_test,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
