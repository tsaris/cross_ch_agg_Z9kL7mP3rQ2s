# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from datetime import timedelta
#from pytorch_lightning.cli import LightningCLI
import torch
import torch.optim as optim
import torch.distributed as dist
import os
import socket
import psutil
import re
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

#from climax.pretrain.datamodule import NativePytorchDataModule
from climax.pretrain.geoai_datasets import MillionAIDDataset_full

import random
import yaml
import sys
import numpy as np
import functools
import datetime

from climax.utils.metrics import lat_weighted_mse, mse
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from climax.vision_transformer import Block, VariableMapping_Attention
#from climax.arch import ClimaX

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
   size_based_auto_wrap_policy, wrap, transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import CPUOffload
from torch.cuda.amp.grad_scaler import GradScaler

import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

import time

from torch.distributed.fsdp import MixedPrecision
from climax.utils.metrics import lat_weighted_mse, lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc
from torchvision.transforms import transforms

# verify we have FSDP activation support ready by importing:
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)

from datasets.sentinel2 import Sentinel2
from datasets.helpers import none_skipper_collate, load_files, denormalize, vis_helper, load_files_appl
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
class ToyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tensor_files = [f for f in os.listdir(root_dir) if f.endswith(".pt")]
        self.transform = transforms.CenterCrop(size=(192, 384))

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        hs_image_path = os.path.join(self.root_dir, self.tensor_files[idx])
        hs_image = torch.load(hs_image_path)

        #hs_image = np.load(hs_image_path)['a']
        #hs_image = torch.from_numpy(hs_image)
        #hs_image = self.transform(hs_image)
        
        return hs_image[0], []

def hs_to_rgb(hs_image,):
    hs_bands = np.load('/proj/tsarisa/cp_from_home/data/appl_tmp/vnir_bands.npy')
    rgb_bands = [650, 532, 473]
    idx = [np.argmin(np.abs(hs_bands - band)) for band in rgb_bands]
    rgb = np.stack([hs_image[..., i] for i in idx], axis=-1)
    return rgb

def np_to_pil(np_image):
    pil_image = Image.fromarray((255*np_image).astype(np.uint8))
    pil_image = ImageEnhance.Brightness(pil_image).enhance(3)
    return pil_image

def plot(batch, pred, mask, epoch):

    transform = transforms.CenterCrop(size=(192, 384))
    
    original_image = batch.squeeze().cpu().detach().numpy()
    original_image = hs_to_rgb(original_image)
    original_image = np.moveaxis(original_image, -1, 0)
    original_image = torch.from_numpy(original_image)
    original_image = transform(original_image)
    original_image = original_image.squeeze().cpu().detach().numpy()
    original_image = np.moveaxis(original_image, 0, -1)
    original_image = np_to_pil(original_image)

    reco = pred.to(torch.float).squeeze().cpu().detach().numpy()
    reco = hs_to_rgb(reco)
    reco = np.moveaxis(reco, -1, 0)
    reco = torch.from_numpy(reco)
    reco = transform(reco)
    reco = reco.squeeze().cpu().detach().numpy()
    reco = np.moveaxis(reco, 0, -1)
    reco = np_to_pil(reco)
    
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0,].imshow(original_image)
    axs[0,].set_title('Original Image')
    #axs[1,].imshow(im_masked)
    axs[1,].set_title('Masked Image')
    axs[2,].imshow(reco)
    axs[2,].set_title('Reconstructed Image')
    plt.savefig("tmp_%s.png"%(str(epoch)))

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
    print(output,flush=True)
    p.export_chrome_trace("/lustre/orion/lrn036/scratch/xf9/measurements/_" + str(p.step_num) + ".json")


def reduce_tensor(tensor: torch.Tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM,)
    rt /= world_size
    return rt


"""
Setup sequence, data, tensor model, and sequence_plus_data parallel groups
"""

def init_par_groups(data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size):

    tensor_par_group = None

    for i in range(data_par_size *seq_par_size):
        ranks = [j for j in range(i*tensor_par_size,(i+1)*tensor_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," tensor_par_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:
            tensor_par_group = group




    seq_par_group = None

    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [t*tensor_par_size*seq_par_size+i+j*tensor_par_size for j in range(seq_par_size)]

            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size, " TENSOR_PAR_SIZE ",tensor_par_size," seq_par_group ranks ",ranks,flush=True)

            group = dist.new_group(ranks)

            if world_rank in ranks:

                seq_par_group = group




    ddp_group = None

    fsdp_group = None

    simple_ddp_group = None

    for i in range(tensor_par_size *seq_par_size):
        ranks = [i+j*tensor_par_size *seq_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]

 
            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," fsdp_ranks",fsdp_ranks)


            group = dist.new_group(fsdp_ranks)

            if world_rank in fsdp_ranks:

                fsdp_group = group


        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]

 
            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," simple_ddp_ranks",simple_ddp_ranks)


            group = dist.new_group(simple_ddp_ranks)

            if world_rank in simple_ddp_ranks:

                simple_ddp_group = group


 
        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," ddp_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            ddp_group = group






    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size * seq_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_seq_ort_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            data_seq_ort_group = group


    return seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group









"""
Setup communication routines: Credit to https://github.com/ORNL/HydraGNN
"""

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0
    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NTASKS") and os.getenv("SLURM_PROCID"):
        ## CADES, Frontier, Perlmutter
        world_size = int(os.environ["SLURM_NTASKS"])
        world_rank = int(os.environ["SLURM_PROCID"])
    ## Fall back to default
    if world_size is None:
        world_size = 1
    return int(world_size), int(world_rank)


def find_ifname(myaddr):
    """
    Find socket ifname for a given ip adress. This is for "GLOO" ddp setup.
    Usage example:
        find_ifname("127.0.0.1") will return a network interface name, such as "lo". "lo0", etc.
    """
    ipaddr = socket.gethostbyname(myaddr)
    ifname = None
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address == ipaddr:
                ifname = nic
                break
        if ifname is not None:
            break
    return ifname


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)
    return nlist


def initialize_process():
    """ "Initialize process"""
    if os.getenv("DDSTORE_BACKEND") is not None:
        backend = os.environ["DDSTORE_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")
    world_size, world_rank = init_comm_size_and_rank()
    ## Default setting
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8889")
    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES/Frontier/Perlmutter specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]
    try:
        if backend in ["nccl", "gloo"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)
        if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
            ifname = find_ifname(master_addr)
            if ifname is not None:
                os.environ["GLOO_SOCKET_IFNAME"] = ifname
        print("Distributed data parallel: %s master at %s:%s" % (backend, master_addr, master_port))
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    except KeyError:
        print("process has to be initialized within a job - Running in sequential mode")
    return world_size, world_rank




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def configure_optimizer(model,lr,beta_1,beta_2,weight_decay):
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
        {
            "params": decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": 0,
        },
        ]
    )

    return optimizer



def configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min):
    
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr,
        eta_min,
    )

    return lr_scheduler


def training_step(batch, dict_in_variables, device, batch_idx: int, net,):
    
    x, _ = batch
    x = x.to(device)
    variables = dict_in_variables['mpi-esm']

    # SHOULD BE LOADED BY THE DATA
    geo_location = (torch.ones(x.shape[0]).to(torch.long) * 1).to(x.dtype)
    geo_location = geo_location.to(device)

    loss_dict, pred, mask = net.forward(x, geo_location, variables, [mse],)
    loss_dict = loss_dict[0]

    loss = loss_dict["loss"]

    return loss, pred, mask



def main(args, device, world_size, world_rank, local_rank):


# Load config file for experiment
#    try:
    config_path = sys.argv[1]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if world_rank==0: 
        print(conf,flush=True)


    seed = conf['seed_everything']
    #seed_everything(seed)

    max_epochs=conf['trainer']['max_epochs']

    checkpoint_path =conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']  
 
    fsdp_size = args.fsdp_size
 
    simple_ddp_size = args.simple_ddp_size
    
    tensor_par_size = args.tensor_par_size

    seq_par_size = args.seq_par_size

    cpu_offload_flag = conf['parallelism']['cpu_offloading']
  
    lr = float(conf['model']['lr'])

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps =  conf['model']['warmup_steps']

    max_steps =  conf['model']['max_steps']

    warmup_start_lr =  float(conf['model']['warmup_start_lr'])

    eta_min =  float(conf['model']['eta_min'])

    class_path = conf['model']['net']['class_path']

    default_vars =  conf['model']['net']['init_args']['default_vars']

    img_size =  conf['model']['net']['init_args']['img_size']

    patch_size =  conf['model']['net']['init_args']['patch_size']
 
    emb_dim =  conf['model']['net']['init_args']['embed_dim']

    depth =  conf['model']['net']['init_args']['depth']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    num_heads = conf['model']['net']['init_args']['num_heads']

    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    drop_path = conf['model']['net']['init_args']['drop_path']

    drop_rate = conf['model']['net']['init_args']['drop_rate']
    
    mask_ratio = conf['model']['net']['init_args']['mask_ratio']

    aggregated_variables = conf['model']['net']['init_args']['aggregated_variables']

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_in_variables = conf['data']['dict_in_variables']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    img_size_x = img_size[0]
    img_size_y = img_size[1]

    data_par_size = fsdp_size * simple_ddp_size

    if (args.arch == "orbit"):
        from climax.arch import ClimaX
    elif (args.arch == "orbit_linear"):
        from climax.arch_linear import ClimaX
    else:
        print("Not supported")
        exit(-1)
   
    
    if world_rank==0:
        print("max_epochs",max_epochs,"data_par_size",data_par_size,"fsdp_size",fsdp_size,"simple_ddp_size",simple_ddp_size,"tensor_par_size",tensor_par_size,"seq_par_size",seq_par_size,"cpu_offloading",cpu_offload_flag,flush=True)
        print("lr ",lr,"beta_1 ",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"class_path",class_path,"default_vars",default_vars,flush=True)
        print("img_size",img_size,"img_size_x",img_size_x,"img_size_y",img_size_y,"patch_size",patch_size,"emb_dim",emb_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"aggregated_variables",aggregated_variables,flush=True)
        print("warmup_steps",warmup_steps,"max_steps",max_steps,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,flush=True)
        print("checkpoint_path",checkpoint_path,"checkpoint_filename",checkpoint_filename,"resume_from_checkpoint",resume_from_checkpoint, flush=True)

    assert (img_size_x%patch_size)==0, "image_size_x % patch_size must be 0"
    assert (img_size_y%patch_size)==0, "image_size_y % patch_size must be 0"

    assert seq_par_size ==1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) ==0, "model heads % tensor parallel size must be 0"


    print("rank",world_rank,"Before initialize parallelism groups",flush=True)



    #initialize parallelism groups
    seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size)



    print("rank",world_rank,"After initialize parallelism groups",flush=True)
    

    ###########################
    #initialize ClimaX model
    model = ClimaX(default_vars=default_vars,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=emb_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        drop_rate=drop_rate,
        parallel_patch_embed=False,
        aggregated_variables=aggregated_variables,
        seq_par_size = seq_par_size,
        tensor_par_size = tensor_par_size,
        data_par_size = data_par_size,
        _ddp_params_and_buffers_to_ignore = ['pos_embed'],
        tensor_par_group = tensor_par_group,
        mask_ratio = mask_ratio
    ).to(device)


    print("rank",world_rank,"After initialize model",flush=True)



 
    #print("rank",world_rank,"after model torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


    if not resume_from_checkpoint:
        epoch_start = 0
        #if train for scratch
        if world_rank==0:       
            print("resume from checkpoint was set to False. Pretrain from scratch.",flush=True)

        # save the model weights of the first rank in tensor par group. Synchronize parameters outside the training block among tensor parallel GPUs.

        if world_rank==0:
 
            # Check whether the specified checkpointing path exists or not

            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")        



            #save initialial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
            init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

            print("rank",dist.get_rank(),"init_model_dict.keys()",init_model_dict.keys(),flush=True)

            torch.save(init_model_dict,
                    checkpoint_path+'/initial_'+str(dist.get_rank())+'.pth')

            print("rank", dist.get_rank(),"after torch.save for initial",flush=True)
            
            del init_model_dict
 
        print("rank", dist.get_rank(),"before dist.barrier(group)",flush=True)
                      
        dist.barrier()

        print("rank", dist.get_rank(),"after dist.barrier(group)",flush=True)


        if world_rank!=0 and world_rank <tensor_par_size:


           #load initial model weights and synchronize model weights that are not in the trianing block among sequence parallel GPUs
           src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

           print("rank",dist.get_rank(),"src_rank",src_rank,flush=True)

           map_location = 'cpu'
           #map_location = 'cuda:'+str(device)
           model.load_state_dict(torch.load(checkpoint_path+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)
           


    else:  
        if world_rank< tensor_par_size:
            if os.path.exists(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

                #map_location = 'cuda:'+str(device)
                map_location = 'cpu'

                checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch_start = torch.tensor(checkpoint['epoch']).to(device)
                del checkpoint
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            else:
                print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)

                sys.exit("checkpoint path does not exist")

    #wait for the completion of model weight loading before moving on



    print("rank", dist.get_rank(),"Before the second dist.barrier()",flush=True)

    if world_rank >= tensor_par_size and resume_from_checkpoint:
        epoch_start = torch.tensor(0).to(device)
    elif not resume_from_checkpoint:
        epoch_start = torch.tensor(0).to(device)
        
    dist.broadcast(epoch_start, src=0)
    epoch_start = epoch_start.item()
    
    dist.barrier()

    print("rank", dist.get_rank(),"After the second dist.barrier()",flush=True)
 



    # check the number of parameters before FSDP wrapping

    if world_rank==0:

        total_params = torch.tensor(0,dtype=torch.long)
        params_per_gpu = torch.tensor(0,dtype=torch.long)

        for name, param in model.named_parameters():
            print("parameter name ",name," requires_gradient ",param.requires_grad, "size",param.shape, flush=True)

            params_per_gpu = params_per_gpu + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)

            if 'attn' not in name and 'mlp' not in name and 'var_agg' not in name: 
                total_params = total_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)
            else:
                total_params = total_params + tensor_par_size * torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)

        print("total_params before FSDP",total_params,"params_per_gpu",params_per_gpu,flush=True)



 
    #print("rank",world_rank,"after model parameter counting torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)




    #set up DDP and synchronize model weights among GPUs in the same data_seq_ort_group

    #model = DDP(model, device_ids=[local_rank], output_device=[local_rank],find_unused_parameters=True,process_group= data_seq_ort_group )


    #my_auto_wrap_policy = functools.partial(
    #    size_based_auto_wrap_policy, min_num_params=1e9
    #)


    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block, VariableMapping_Attention   # < ---- Your Transformer layer class
        },
    )

    bfloatPolicy = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    #add hybrid sharded FSDP
    if fsdp_size >1 and simple_ddp_size >1:
        model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add fully sharded FSDP
    elif fsdp_size >1 and simple_ddp_size ==1:
        model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add unsharded DDP
    else:
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False, use_orig_params=True )
    
    check_fn = lambda submodule: isinstance(submodule, Block) or isinstance(submodule, VariableMapping_Attention)


    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )



    if world_rank==0:

        print("total_params after FSDP",model,flush=True)






    ##############################
    #initialize optimizer
    optimizer = configure_optimizer(model,lr,beta_1,beta_2,weight_decay)


    if resume_from_checkpoint:

        print("optimizer resume from checkpoint was set to True. Checkpoint path found.",flush=True)

        src_rank = world_rank - tensor_par_size * dist.get_rank(group=data_seq_ort_group)
        print("rank",dist.get_rank(),"src_rank",src_rank,flush=True)

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

        checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint



    ###############################
    #initialize scheduler
    scheduler = configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min)

    #if (img_size_x!=img_size_y):
    #    print("Needs to be the same size x,y because of transformations")
    #    exit(-1)
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size_x, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'million-aid':
        dataset_train = MillionAIDDataset_full(
            dict_root_dirs['mpi-esm'],
            train=True,
            transform=transform_train,
            tag=None)
        
    elif args.dataset == 'sentinel2':
        filepaths, filescount = load_files(dict_root_dirs['mpi-esm'])
        #dataset = Sentinel2(filepaths, metadata='datasets/seusa_fmpoints_lookup_latlon.csv', transform=None)
        dataset = Sentinel2(filepaths,locdata='datasets/seusa_fmpoints_lookup_latlon.csv',labeldata='datasets/pointsIndices_labels.csv')
        #dataset_train = Subset(dataset, range(200000)) #200K subset
        dataset_train = Subset(dataset, range(34)) #200K subset         
        print(f'dataset: {len(dataset)}')
        print(f'dataset_train: {len(dataset_train)}')

        for i in dataset_train:
            x, _ = i
            print(x)
            print(x.shape, type(x))
            break

        exit(-1)
        
        
    elif args.dataset == 'appl':
        dataset_train = ToyDataset(dict_root_dirs['mpi-esm'])               
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=data_par_size,
        rank=dist.get_rank(group=ddp_group),
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=pin_memory,
        drop_last=True,
        #collate_fn=none_skipper_collate
    )

    if world_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    #
    scaler = GradScaler(init_scale=8192, growth_interval=100)
    min_scale= 128

    step = 0

    for epoch in range(epoch_start,epoch_start+max_epochs):
    
        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        # visualization 4x per epoch
        vis_freq = len(train_dataloader) // 4
        for batch_idx, batch in enumerate(train_dataloader):
            #Skip empty batch
            #if not batch:
            #    continue

            loss, pred, mask = training_step(batch, dict_in_variables, device, batch_idx,model,)

            epoch_loss += loss.detach()
            
            if world_rank==0:
                # Log batch loss to TensorBoard
                if log_writer:
                    log_writer.add_scalar("Train/Batch Loss", loss.item(), step)

                print("epoch-train: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss.item(),"get_lr ",scheduler.get_lr(),flush=True)

            #
            optimizer.zero_grad()
            scaler.scale(loss).backward() 
            scaler.step(optimizer)    
            scaler.update()
            
            if scaler._scale <min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler._scale)
            
            step+=1

            scheduler.step()            

            #if (step==20): break

        # Data loop ends here
        if world_rank==0:
            if log_writer:
                avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
                # Log average epoch loss to TensorBoard
                log_writer.add_scalar("Train/Epoch Loss", avg_epoch_loss, epoch)
                # Log the learning rate to TensorBoard
                log_writer.add_scalar("Train/Learning Rate", scheduler.get_last_lr()[0], epoch)

                #if (epoch==1):                
                #denormalize_fn = denormalize                
                #sample_view = vis_helper(pred, mask, batch[0], denormalize_fn=denormalize_fn,bands=[1,2,3])
                #sample_view = vis_helper(pred, mask, original_image, denormalize_fn=denormalize_fn,bands=[1,2,3])                
                #log_writer.add_image('visual', sample_view, global_step=epoch)

                #if (epoch==0):
                #    plot(batch[0][0], pred[0], mask[0], epoch)                
                
            print("epoch: ",epoch," epoch_loss ",epoch_loss.item()/(len(train_dataloader)),flush=True)

        #
        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()

        if epoch %100 ==0 and world_rank < tensor_par_size:
            torch.save({
                'epoch': epoch_start+max_epochs,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                }, checkpoint_path+"/"+checkpoint_filename+"_epoch_"+str(epoch)+"_rank_"+str(world_rank)+".ckpt")

        dist.barrier()
        del model_states
        del optimizer_states
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
    parser.add_argument("yaml_config", default='./config/ViT.yaml', type=str, help='path')

    parser.add_argument('--arch',)
    
    parser.add_argument('--fsdp_size', type=int, default=1)
    parser.add_argument('--simple_ddp_size', type=int, default=1)
    parser.add_argument('--seq_par_size', type=int, default=1)
    parser.add_argument('--tensor_par_size', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='sentinel2')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    world_size = int(os.environ['WORLD_SIZE'])
    world_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    

    #os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    #os.environ['MASTER_PORT'] = "29500"
    #os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    #os.environ['RANK'] = os.environ['SLURM_PROCID']

    #world_size = int(os.environ['SLURM_NTASKS'])
    #world_rank = int(os.environ['SLURM_PROCID'])
    #local_rank = int(os.environ['SLURM_LOCALID'])


    NOW = str(datetime.datetime.now()).replace(" ","--")
    #log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = './logs/{}_dataset[{}]_batch_size[{}]/'. \
        format(NOW,args.dataset, args.batch_size)

    args.log_dir = log_dir

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    
    main(args, device, world_size, world_rank, local_rank)

    dist.destroy_process_group()
