import glob
import random
import numpy as np
import torch
import webdataset as wds
import time
import math
import json
import tqdm
import sys
import functools
import os
from pathlib import Path
from importlib import import_module
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from absl import logging
import wandb
import ml_collections
from src.loss import mae_loss
from src import utilities
from src import misc
from src import lr_sched
from src.misc import NativeScalerWithGradNormCount as NativeScaler
import argparse
import timm.optim.optim_factory as optim_factory
from src.data import dataset_helper, parsing_utilities
from src.data.features import LogMelSpec
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributed.elastic.multiprocessing.errors import record
from torchdata.datapipes.iter import FileLister, FileOpener, Shuffler
from torchdata.dataloader2 import DataLoader2, DistributedReadingService, MultiProcessingReadingService, SequentialReadingService
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=True)
    parser.add_argument("--config", default="", type=str, help="path to config file")
    parser.add_argument('--workdir', default='', type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--fake_data', action='store_true')
    parser.add_argument('--bias_decay', action='store_true')
    parser.add_argument('--use_rs', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--precision", default='float32', type=str)
    parser.add_argument("--min_lr", type=float, default=0.)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def train(train_iter, model,
          criterion, optimizer,
          epoch, steps_per_epoch,
          device,
          loss_scaler,
          args,
          wandb_logger=None,
          total_steps_counter=0,
          ):
    model.train()
    prefetcher = utilities.Prefetcher(train_iter, device)
    inp, _ = prefetcher.next()
    accum_iter = args.accum_iter
    autocast_dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    autocast_enabled = True if "16" in args.precision else False
    step_times = []

    loss_values = []

    data_iter_step = 0
    #with model.join():
        # while inp is not None:
    for step_index in range(steps_per_epoch):
            t0 = time.time()
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / args.steps_per_epoch + epoch, args)
            # print("now forwarding..")
            if step_index == 0:
               print("inp shape:", inp.shape)
            with torch.autocast(device_type="cuda", 
                                dtype=autocast_dtype,
                                enabled=autocast_enabled):
                pred, target, mask = model(inp)
                loss = criterion(pred, target, mask)
            loss = loss.mean()
            loss_value = loss.item()
            # print("got loss")

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter
            cond = (data_iter_step + 1) % accum_iter == 0
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad_value, 
                                    parameters=model.parameters(),
                                    update_grad=cond)
            if cond:
                optimizer.zero_grad()
        
            # torch.cuda.synchronize()
            # print("grad_norm: {} | loss: {}".format(grad_norm, loss))
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            grad_norm_reduce = misc.all_reduce_mean(grad_norm)
            # print("loss value reduced:", loss_value_reduce)
            lr = optimizer.param_groups[0]["lr"]
            step_times.append(time.time()-t0)
            steps_per_sec = 1 / (sum(step_times[-5:]) / len(step_times[-5:]))
            samples_per_second = args.global_bs * steps_per_sec
            if data_iter_step % args.print_freq == 0:
                print("Epoch: {:03d} [{:04d}] | loss: {:.04f} | grad_norm: {:.04f} | steps_per_second: {:.02f} | samples_per_second: {:.02f} | lr: {:.08f}".format(
                    epoch, data_iter_step, loss_value_reduce, grad_norm_reduce, steps_per_sec, samples_per_second, lr)
                )
            if wandb_logger is not None and misc.is_main_process():
                wandb_logger.log({
                    "train_loss": loss_value_reduce,
                    "step": total_steps_counter,
                    "lr": lr,
                    "steps_per_second": steps_per_sec,
                    "samples_per_second": samples_per_second,
                    "grad_norm": grad_norm_reduce
                })
            
            loss_values.append(loss_value_reduce)
            data_iter_step += 1
            total_steps_counter += 1
            inp, _ = prefetcher.next()
    
    mean_loss = sum(loss_values)/len(loss_values)
    print("Epoch: {} | Mean tr loss: {}".format(epoch, mean_loss))
    return {"loss": mean_loss}, total_steps_counter


def setup_data(config, train=False, num_workers=4, use_reading_service=False):
    _, samples = dataset_helper.get_data_dirs(config.data.train_dirs, config.data.train_samples)
    frequency_first = config.model.model_args.get("frequency_first", False)
    parser_fn = functools.partial(
            parsing_utilities.np_spec_parser, 
            req_num_frames=config.data.num_frames,
            crop_type="random" if train else "center",
            flip_ft=frequency_first
        )
    record_parser_fn = functools.partial(
            parsing_utilities.numpy_record_parser,
            numpy_spec_parser_fn=parser_fn,
        )
    shuffle_buffer = config.get("shuffle_buffer_multiplier", 1000)
    eff_shuffle_buffer = ((shuffle_buffer*config.batch_size*10)//(num_workers*torch.distributed.get_world_size()))
    print("Effective shuffle buffer:", eff_shuffle_buffer)
    dp = FileLister(config.data.train_dirs, "*.tar")
    dp = Shuffler(dp, buffer_size=1000)
    # dp = FileOpener(dp, mode='b')
    # dp = Shuffler(dp, buffer_size=1000)
    dp = dp.sharding_filter()
    dp = FileOpener(dp, mode='b')
    dp = dp.load_from_tar(length=samples).map(dataset_helper.decode_np_tar).webdataset()
    dp = Shuffler(dp, buffer_size=eff_shuffle_buffer)
    # dp = dp.prefetch(2)
    dp = dp.map(dataset_helper.fix_keys_for_tp).map(record_parser_fn).map(dataset_helper.return_data)
    # dp = dp.prefetch(config.batch_size*20)
    dp = dp.cycle()

    if use_reading_service:
       print("!!!!!!!!!! using reading service !!!!!!!!!!!")
       dp = dp.batch(config.batch_size, drop_last=False).prefetch(config.batch_size*20)
       dp = dp.collate()
       mp_rs = MultiProcessingReadingService(num_workers=num_workers)
       dist_rs = DistributedReadingService()
       rs = SequentialReadingService(dist_rs, mp_rs)
       loader = DataLoader2(dp, reading_service=rs)
    else:
       loader = torch.utils.data.DataLoader(
           dp, batch_size=config.batch_size, shuffle=True,
           num_workers=num_workers, drop_last=False, pin_memory=False
       )

    return loader, samples//(config.batch_size * torch.distributed.get_world_size()), samples


@record
def main(args):
    config = args.config
    misc.init_distributed_mode(args)
    print("WORLD_SIZE:", args.world_size)
    print("LOCAL_RANK:", args.gpu)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("num_tasks:", num_tasks)
    print("global_rank:", global_rank)

    workdir = args.workdir

    args.output_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)
    
    wandb_logger = None
    if misc.is_main_process() and not args.no_wandb:
        wandb_logger = wandb.init(project='{}'.format(config.wandb.get("project", "audax-cola")),
                                  group="{}".format(config.data.dataset_name),
                                  config=config.to_dict(), name=workdir.split("/")[-1])
    else:
        wandb_logger = None

    device = torch.device("cuda:{}".format(args.gpu))

    train_iter, steps_per_epoch, samples = setup_data(config, train=True, num_workers=args.num_workers, use_reading_service=args.use_rs)
    if args.fake_data:
        print("USING FAKE DATA")
        tr_set = datasets.FakeData(samples, (1, 200, 80), 1000, transforms.ToTensor())
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(tr_set)
        train_iter = torch.utils.data.DataLoader(
            tr_set, batch_size=config.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    num_tr_steps = int(steps_per_epoch * config.num_epochs)

    args.steps_per_epoch = steps_per_epoch
    args.warmup_epochs = config.opt.warmup_epochs
    args.epochs = config.num_epochs

    print("Total steps: {} | Steps per epoch: {}".format(num_tr_steps, args.steps_per_epoch))

    # create model here

    model = utilities.get_model(config)
    model.to(device)
    model_without_ddp = model
    print(model)

    if args.distributed:
        print("DISTRIBUTED IS TRUEEEEEEEE...")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # create criterion here
    use_norm_pix = config.opt.get("norm_pix_loss", False)
    criterion = functools.partial(mae_loss, norm_pix_loss=use_norm_pix)

    # create optimizer here
    args.accum_iter = config.opt.get("grad_accum_steps", 1)
    args.clip_grad_value = config.opt.get("clip_grad_value", None)
    base_learning_rate = args.accum_iter * config.opt.learning_rate * config.batch_size * args.world_size / 256.
    wd = config.opt.weight_decay
    args.lr = base_learning_rate
    args.global_bs = config.batch_size * args.world_size    

    if args.bias_decay:
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=wd)
    else:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, wd)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    final_epochs = list(range(config.num_epochs))[-5:]
    if args.resume:
       misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    elif os.path.exists(args.output_dir):
       func = lambda x:int(x.split("/")[-1].split("-")[-1].replace(".pth",""))
       existing_ckpts = sorted(glob.glob(os.path.join(args.output_dir, "*.pth")), key=func)
       if len(existing_ckpts) != 0:
           latest_ckpt = existing_ckpts[-1]
           args.resume = latest_ckpt
           misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    total_steps_counter = 0
    for epoch in range(args.start_epoch, config.num_epochs):
        train_stats, total_steps_counter = train(
            train_iter, model,
            criterion, optimizer, epoch, steps_per_epoch, device,
            loss_scaler, args, wandb_logger,
            total_steps_counter
        )

        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs or epoch in final_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.workdir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # print(log_stats)
        # if not args.use_rs:
        #    train_iter, steps_per_epoch, samples = setup_data(config, train=True, num_workers=args.num_workers, use_reading_service=args.use_rs)

    print("outside the train loop")
    misc.barrier()
    print("barrier done")
    if wandb_logger is not None:
        wandb_logger.finish()
    misc.cleanup()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # print(args)
    config = import_module(args.config).get_config()
    # print(config)
    args.config = config
    if args.workdir:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
    main(args)