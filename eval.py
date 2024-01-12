import torch

import argparse
import numpy as np
import os

from myflow import GMFlow
from data_utils.evaluate import validate_things

def get_args_parser():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default='/data/gmflow/pretrained/myflow_occ_3.pth', type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')

    # GMFlow model
    parser.add_argument('--upsample_factor', default=8, type=int)

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)


    return parser


def main(args):
    print('Use %d GPUs' % torch.cuda.device_count())
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda')

    # model
    model = GMFlow().to(device)

    checkpoint = torch.load(args.resume, map_location='cuda')

    model.load_state_dict(checkpoint['model'], strict=True)

    results = validate_things(args, model)
    print(results)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
