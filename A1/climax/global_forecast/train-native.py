# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from datetime import timedelta
#from pytorch_lightning.cli import LightningCLI
import torch
import torch.optim as optim
import torch.distributed as dist
import os
import socket
import psutil
import re
from climax.global_forecast.datamodule import NativePytorchGlobalForecastDataModule
from climax.arch import ClimaX
import random
import yaml
import sys
import numpy as np
import functools

from climax.utils.metrics import lat_weighted_mse
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

from climax.vision_transformer import Block, VariableMapping_Attention
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import (
   size_based_auto_wrap_policy, wrap, transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import CPUOffload
from torch.cuda.amp.grad_scaler import GradScaler

import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import time
from torch.distributed.fsdp import MixedPrecision
from climax.utils.pos_embed import interpolate_pos_embed

# verify we have FSDP activation support ready by importing:
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)
from torch.utils.tensorboard import SummaryWriter
import subprocess
import datetime

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
    print(output,flush=True)
    p.export_chrome_trace("/lustre/orion/lrn036/scratch/xf9/measurements/_" + str(p.step_num) + ".json")

def load_pretrained_weights(model, pretrained_path, device, src_rank):
    if "{src_rank}" in pretrained_path:
        pretrained_path = pretrained_path.replace("{src_rank}", f"{src_rank}")
    if not os.path.exists(pretrained_path):
        print("Loading pre-trained checkpoint was set to True. But the path does not exist.", pretrained_path, flush=True)
        sys.exit("pre-trained checkpoint path does not exist")
    # map_location = 'cuda:'+str(device)
    map_location = 'cpu'
    checkpoint = torch.load(pretrained_path, map_location=map_location)

    print("Loading pre-trained checkpoint from: %s" % pretrained_path)
    checkpoint_model = checkpoint["model_state_dict"]

    del checkpoint
    # interpolate positional embedding
    interpolate_pos_embed(model, checkpoint_model, new_size=model.img_size)


    state_dict = model.state_dict()
    if model.parallel_patch_embed:
        if "token_embeds.proj_weights" not in checkpoint_model.keys():
            raise ValueError(
                "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
            )

    # checkpoint_keys = list(checkpoint_model.keys())
    for k in list(checkpoint_model.keys()):
        if "channel" in k:
            print("k:", k)
            checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
            del checkpoint_model[k]
    for k in list(checkpoint_model.keys()):
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del checkpoint_model[k]
        elif checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint: no matching", checkpoint_model[k].shape, state_dict[k].shape)
            del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

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
            if world_rank==0:
                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size, " TENSOR_PAR_SIZE ",tensor_par_size," seq_par_group ranks ",ranks,flush=True)
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

#            if world_rank==0:
#                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," fsdp_ranks",fsdp_ranks)

            group = dist.new_group(fsdp_ranks)
            if world_rank in fsdp_ranks:
                fsdp_group = group

        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]

#            if world_rank==0:
#                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," simple_ddp_ranks",simple_ddp_ranks)

            group = dist.new_group(simple_ddp_ranks)
            if world_rank in simple_ddp_ranks:
                simple_ddp_group = group

#        if world_rank==0:
#            print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," ddp_group ranks ",ranks)
        group = dist.new_group(ranks)
        if world_rank in ranks:
            ddp_group = group

    data_seq_ort_group = None
    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size * seq_par_size)]
#        if world_rank==0:
#            print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_seq_ort_group ranks ",ranks)
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




global_step = 0
def training_step(batch, device, batch_idx: int, net: ClimaX, lat, logger=None):
    global global_step

    x, y, lead_times, variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    lead_times = lead_times.to(device)

    loss_dict, _ = net.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=lat)
    loss_dict = loss_dict[0]
    if logger is not None:
        for var in loss_dict.keys():
            logger.add_scalar(
                "train/" + var,
                loss_dict[var].to(torch.float),
                global_step=global_step,
            )

    loss = loss_dict["loss"]
    global_step += 1

    return loss

## For early stop
def timedelta_parse(text):
    """
    Convert input string to timedelta.
    format: [[[d-]h:]m:]s
    """
    tokens = text.replace("-", ":").split(":")
    return datetime.timedelta(**{ key: float(val) for val, key in zip(tokens[::-1], ("seconds", "minutes", "hours", "days")) })

def check_earlystop(world_rank, t0):
    ## Early stop
    jobid = os.getenv("SLURM_JOB_ID", None)
    should_stop = False
    if jobid is not None:
        if world_rank == 0:
            cmd = f"squeue -h -j {jobid} -o %L"
            proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
            timestr = proc.stdout.decode('utf-8').strip()
            left = timedelta_parse(timestr).total_seconds()
            esitmated = time.time() - t0
            should_stop = torch.tensor(left < esitmated, dtype=torch.bool).to(device)
            print ("should_stop:", left, esitmated, should_stop.item())
        else:
            should_stop = torch.tensor(False, dtype=torch.bool).to(device)

        dist.broadcast(should_stop, src=0)
        should_stop = should_stop.item()
    return should_stop

def main(device):
    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()
# Load config file for experiment
#    try:
    config_path = sys.argv[1]
    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
    if world_rank==0: 
        print(conf,flush=True)

    max_epochs=conf['trainer']['max_epochs']
    checkpoint_path =conf['trainer']['checkpoint_path']
    pretrained_path =conf['trainer']['pretrained_path']
    checkpoint_filename = conf['trainer']['checkpoint_filename']
    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']
    fsdp_size = conf['parallelism']['fsdp_size']
    simple_ddp_size = conf['parallelism']['simple_ddp_size']
    tensor_par_size = conf['parallelism']['tensor_par_size']
    seq_par_size = conf['parallelism']['seq_par_size']

    ## jyc: Adjust decomposition
    b = world_size//(fsdp_size * simple_ddp_size * seq_par_size * tensor_par_size)
    simple_ddp_size = b * simple_ddp_size
    print ("fsdp_size,simple_ddp_size,seq_par_size,tensor_par_size",fsdp_size,simple_ddp_size,seq_par_size,tensor_par_size)
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
    aggregated_variables = conf['model']['net']['init_args']['aggregated_variables']
    root_dir = conf['data']['root_dir']
    variables = conf['data']['variables']
    out_variables = conf['data']['out_variables']
    predict_range = conf['data']['predict_range']
    hrs_each_step = conf['data']['hrs_each_step']
    buffer_size = conf['data']['buffer_size']
    batch_size = conf['data']['batch_size']
    num_workers = conf['data']['num_workers']
    pin_memory = conf['data']['pin_memory']
    img_size_x = img_size[0]
    img_size_y = img_size[1]
    data_par_size = fsdp_size * simple_ddp_size
   
    if world_rank==0:
        print("max_epochs",max_epochs,"data_par_size",data_par_size,"fsdp_size",fsdp_size,"simple_ddp_size",simple_ddp_size,"tensor_par_size",tensor_par_size,"seq_par_size",seq_par_size,"cpu_offloading",cpu_offload_flag,flush=True)
        print("lr ",lr,"beta_1 ",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"class_path",class_path,"default_vars",default_vars,flush=True)
        print("img_size",img_size,"img_size_x",img_size_x,"img_size_y",img_size_y,"patch_size",patch_size,"emb_dim",emb_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"aggregated_variables",aggregated_variables,flush=True)
        print("root_dir",root_dir,"batch_size",batch_size,"num_workers",num_workers,flush=True)
        print("warmup_steps",warmup_steps,"max_steps",max_steps,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,flush=True)
        print("checkpoint_path",checkpoint_path,"checkpoint_filename",checkpoint_filename,"resume_from_checkpoint",resume_from_checkpoint, flush=True)

        conf['trainer']['checkpoint_path'] = checkpoint_path
        conf['parallelism']['fsdp_size']= fsdp_size
        conf['parallelism']['simple_ddp_size'] = simple_ddp_size
        conf['parallelism']['tensor_par_size'] = tensor_par_size
        conf['parallelism']['seq_par_size'] = seq_par_size
        with open(os.path.join(checkpoint_path, "conf.yaml"), 'w') as f:
            yaml.dump(conf, f)

    assert (img_size_x%patch_size)==0, "image_size_x % patch_size must be 0"
    assert (img_size_y%patch_size)==0, "image_size_y % patch_size must be 0"

    assert seq_par_size ==1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) ==0, "model heads % tensor parallel size must be 0"

    assert len(pretrained_path) > 0, "pretrained_path cannot be empty"

    #initialize parallelism groups
    seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size)

    #print("rank",world_rank,"before model torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

    ## Tensorboard logger
    logger = None
    if world_rank==0:
        logger_path = os.path.join(checkpoint_path, "logs")
        print("logger_path", logger_path)
        os.makedirs(logger_path, exist_ok=True)
        logger = SummaryWriter(logger_path)

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
        tensor_par_group = tensor_par_group,
    ).to(device)


    #print("rank",world_rank,"after model torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

    if not resume_from_checkpoint:
        epoch_start = 0
        #if train from pretrained model
        if world_rank < tensor_par_size:
            if os.path.exists(pretrained_path+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to Flase. Resume from pretrained model path:",pretrained_path,flush=True)
                print("Load model states. rank",world_rank,flush=True)
                #map_location = 'cuda:'+str(device)
                map_location = 'cpu'
                checkpoint = torch.load(pretrained_path+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint
            else:
                print("resume from checkpoint was set to False. But the pretrained path does not exist.",flush=True)
                sys.exit("pretrained path does not exist")
        #wait for the completion of model weight loading before moving on
        dist.barrier()

    #if resume from checkpoint, load model checkpoint among the the first tensor parallel group. Then broadcast to FSDP and simple DDP groups
    else:
        if world_rank < tensor_par_size:
            if os.path.exists(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found,",checkpoint_path,flush=True)
                print("Load model states. rank",world_rank,flush=True)
                #map_location = 'cuda:'+str(device)
                map_location = 'cpu'
                checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch_start = checkpoint['epoch']
                del checkpoint
            else:
                print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)
                sys.exit("checkpoint path does not exist")
        #wait for the completion of model weight loading before moving on
        dist.barrier()

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

    local_rank = int(os.environ['SLURM_LOCALID'])
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
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )




    #without simple DDP. Just FSDP and tensor parallel
#    model = FSDP(model, device_id=local_rank, process_group= data_seq_ort_group, sync_module_states=True, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
#    model = FSDP(model, device_id=local_rank, process_group= data_seq_ort_group, auto_wrap_policy = my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers = False, mixed_precision=dist.fsdp.MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True))
#    model = FSDP(model, device_id=local_rank, process_group= data_seq_ort_group, auto_wrap_policy = my_auto_wrap_policy)

    #print("rank",world_rank,"after FSDP torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

    #non_reentrant_wrapper = functools.partial(
    #    checkpoint_wrapper,
    #    offload_to_cpu=False,
    #    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #)

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

        map_location = 'cpu'

        checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint
    #    print("Load model optimizer. rank",world_rank,"src_rank",src_rank,flush=True)
    #    optim_state_dict = checkpoint['optimizer_state_dict']

    #    optim_state_dict = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=optim_state_dict)
    #    optimizer.load_state_dict(optim_state_dict)




    ###############################
    #initialize scheduler
    scheduler = configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min)



    epoch_start = 0

    #print("rank",world_rank,"after read and load model state torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

#    dist.all_reduce(model.module.pos_embed.data, op=dist.ReduceOp.SUM, group=ddp_group , async_op=False)

#    model.module.pos_embed.data = model.module.pos_embed.data / data_par_size

    #print("rank",dist.get_rank(),"module.token_embeds.0.proj.weight",model.module.token_embeds[0].proj.weight[0,0,0,0].data,"module.blocks.0.norm1.weight",model.module.blocks[0].norm1.weight[0].data,"module.blocks.5.mlp.fc1.weight",model.module.blocks[5].mlp.fc1.weight[0,0].data,flush=True)

    data_module = NativePytorchGlobalForecastDataModule(
        root_dir=root_dir,
        variables=variables,
        buffer_size=buffer_size,
        out_variables=out_variables,
        predict_range=predict_range,
        hrs_each_step=hrs_each_step,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        data_par_size = data_par_size,
        ddp_group = ddp_group,
    ).to(device)

    data_module.setup()
    lat, lon = data_module.get_lat_lon()
    train_dataloader = data_module.train_dataloader()
    scaler = GradScaler(init_scale=8192, growth_interval=100)
    min_scale= 128

    print("rank",world_rank,"Before the start of epoch, torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

    dist.barrier()

    for epoch in range(epoch_start,epoch_start+max_epochs):
        ## timer per epoch
        t0 = time.time()
        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)
    
        for batch_idx, batch in enumerate(train_dataloader):

            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic1 = time.perf_counter() 

            loss = training_step(batch, device, batch_idx,model, lat, logger=logger)

            #torch.cuda.synchronize(device=device)
            #tic2 = time.perf_counter()

            epoch_loss += loss.detach()

            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss,"time.time()-t0",time.time()-t0,flush=True)
    
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            #torch.cuda.synchronize(device=device)
            #tic3 = time.perf_counter() 

            scaler.step(optimizer)
    
            scaler.update()
   
            if scaler._scale <min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler._scale)
    
            if world_rank==0:
                print("rank",world_rank,"batch_idx",batch_idx,"get_lr",scheduler.get_lr(),"after optimizer step torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter() 
                print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",flush=True)


            scheduler.step()

    
        if world_rank==0:
            print("epoch: ",epoch," rank",world_rank," epoch_loss ",epoch_loss,flush=True)

        if world_rank ==0:    
            # Check whether the specified checkpointing path exists or not
            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")
            logger.close()


        print("rank",world_rank,"Before torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()

        if world_rank < tensor_par_size:
            torch.save({
                'epoch': epoch_start+max_epochs,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                }, checkpoint_path+"/"+checkpoint_filename + f"_rank_{world_rank}.ckpt")


        ## Early stop
        should_stop = check_earlystop(world_rank, t0)
        if should_stop:
            print("No time left. Early stop.", )
            break

        dist.barrier()
        del model_states
        del optimizer_states

        print("rank",world_rank,"After torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    #torch.backends.cudnn.benchmark = True
    # dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

    initialize_process()
    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    main(device)
    dist.destroy_process_group()
