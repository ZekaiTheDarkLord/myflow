from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from data_utils import frame_utils

from data_utils.utils import InputPadder

from data_utils import datasets

@torch.no_grad()
def validate_things(args, model,
                    val_things_clean_only=True,
                    test_set=True
                    ):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        if val_things_clean_only:
            if dstype == 'frames_finalpass':
                continue

        val_dataset = datasets.FlyingThings3D(dstype=dstype, test_set=test_set, validate=True,
                                          )
        print('Number of validation image pairs: %d' % len(val_dataset))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True)

        epe_1_list = []
        epe_2_list = []

        for sample in val_loader:
            # img1, img2, flow_gt_list, valid_list, _ = [x.to('cuda') for x in sample]
            img1, img2, flow_gt_list, valid_list = sample
            img1 = img1.to('cuda')
            img2 = img2.to('cuda')
            flow_gt_list = [flow_gt.to('cuda') for flow_gt in flow_gt_list]
            valid_list = [valid.to('cuda') for valid in valid_list]

            model.init_hw(img1.shape[-2]//args.upsample_factor, img1.shape[-1]//args.upsample_factor)

            flow_preds = model(img1, img2)

            # mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
            # valid = valid > 0.5

            epe_1 = torch.sum((flow_preds[-1] - flow_gt_list[-1]) ** 2, dim=1).sqrt()
            epe_1 = epe_1.view(-1)[valid_list[-1].view(-1)]
            epe_1_list.append(epe_1.mean().item())

            epe_2 = torch.sum((flow_preds[-2] - flow_gt_list[-2]) ** 2, dim=1).sqrt()
            epe_2 = epe_2.view(-1)[valid_list[-2].view(-1)]
            epe_2_list.append(epe_2.mean().item())

        epe_1_mean = np.mean(epe_1_list)
        epe_2_mean = np.mean(epe_2_list)

        test_set_str = 'test_' if test_set else 'train_'

        results[test_set_str+'things_1'] = epe_1_mean
        results[test_set_str+'things_0.5'] = epe_2_mean

    return results


@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(aug_params = {'crop_size': args.image_size, 'min_scale': -0.2}, split='training', dstype=dstype)

        print('Number of validation image pairs: %d' % len(val_dataset))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True)

        epe_list = []

        for sample in val_loader:
            img1, img2, flow_gt, valid = [x.to('cuda') for x in sample]

            flow_preds = model(img1, img2)

            mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
            valid = valid > 0.5

            epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)]

            epe_list.append(epe.mean().item())

        epe = np.mean(epe_list)

        print("Validation Sintel test set (%s) EPE: %.3f" % (dstype, epe))
        results['sintel_' + dstype + '_epe'] = epe

    return results
