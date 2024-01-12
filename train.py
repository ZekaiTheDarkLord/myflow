import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

from data_utils.datasets import build_train_dataset
from myflow import GMFlow
from loss import flow_loss_func
from data_utils.evaluate import validate_things, validate_sintel
from load_model import my_load_weights, my_freeze_model

import re
import datetime
import time


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    print('Use %d GPUs' % torch.cuda.device_count())
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = GMFlow().to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:

        state_dict = my_load_weights(args.resume)

        model_without_ddp.load_state_dict(state_dict, strict=args.strict_resume)

        my_freeze_model(model_without_ddp)

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        # if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
        #         args.no_resume_optimizer:
        #     print('Load optimizer')
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     start_epoch = checkpoint['epoch']
        #     start_step = checkpoint['step']
        # print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # training datset
    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    train_sampler = None

    # shuffle = False if args.distributed else True
    shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, args.lr,
    #     args.num_steps + 10,
    #     pct_start=0.05,
    #     cycle_momentum=False,
    #     anneal_strategy='cos',
    #     last_epoch=last_epoch,
    # )

    total_steps = start_step
    epoch = start_epoch
    print('Start training')
    print('len:', len(train_loader))

    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, img2, flow_gt_list, valid_list = sample
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow_gt_list = [flow_gt.to(device) for flow_gt in flow_gt_list]
            valid_list = [valid.to(device) for valid in valid_list]

            model.init_hw(img1.shape[-2]//args.upsample_factor, img1.shape[-1]//args.upsample_factor)

            flow_preds = model(img1, img2)
            flow_preds_bw = model(img2, img1)

            print(flow_preds[2].shape)
            print(flow_preds_bw[2].shape)

            loss, metrics = flow_loss_func(flow_preds, flow_gt_list, valid_list,
                                           gamma=args.gamma,
                                           max_flow=args.max_flow,
                                           )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # lr_scheduler.step()

            print(total_steps, metrics['epe'], metrics['mag'], optimizer.param_groups[0]['lr'], datetime.datetime.now().strftime("%H:%M:%S"))

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.val_freq == 0:
                print('Start validation')

                val_results = {}
                # support validation on multiple datasets

                if 'things' in args.val_dataset:
                    start_time = time.time()
                    test_results_dict = validate_things(args, model_without_ddp, test_set=True)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)
                    end_time = time.time()
                    print('validate_things:', end_time-start_time)
                    train_results_dict = validate_things(args, model_without_ddp, test_set=False)
                    if args.local_rank == 0:
                        val_results.update(train_results_dict)

                if 'sintel' in args.val_dataset:
                    results_dict = validate_sintel(model_without_ddp,
                                                   count_time=args.count_time,
                                                   padding_factor=args.padding_factor,
                                                   with_speed_metric=args.with_speed_metric,
                                                   evaluate_matched_unmatched=args.evaluate_matched_unmatched,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    val_results.update(results_dict)

                if args.local_rank == 0:
                    print(val_results)

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)

                        for k, v in val_results.items():
                            f.write("| %s: %.3f " % (k, v))

                        f.write('\n\n')

                model.train()

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
