import torch
import re
import config

# prev_feature_dim = 64


def my_load_weights(weight_path):

    print('Load checkpoint: %s' % weight_path)

    checkpoint = torch.load(weight_path, map_location='cuda')

    state_dict = {}

    for k, v in checkpoint['model'].items():

        # if k.startswith('up_sampler.'):
        #     continue
        # if k == 'merge_conv.0.weight':
        #     continue
        # elif k.startswith('merge_conv.'):
        #     continue

        state_dict[k] = v

    return state_dict


def my_freeze_model(model):
    for name, param in model.named_parameters():
        pass
        # if not name.startswith('flow_conv.'):
        #     param.requires_grad = False
        # if args.fix and (not 'local_head' in name) and (not 'Reranker' in name):
        #     param.requires_grad = False
        # if param.requires_grad: