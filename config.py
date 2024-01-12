import torch

# width = 80
# height = 64
feature_dim = 64
# attn_num_splits = 2
# batch_size = 1

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')