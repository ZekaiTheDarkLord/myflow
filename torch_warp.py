import numpy as np
import torch
import tensorflow as tf
from tf_warp import tf_get_pixel_value


def torch_get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a 4D tensor image using PyTorch.

    Input
    -----
    - img: tensor of shape (B, C, H, W)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    B, C, H, W = img.shape

    batch_indices = torch.arange(B, dtype=torch.long).view(B, 1, 1)
    batch_indices = batch_indices.expand(B, H, W)

    indices = torch.stack([batch_indices, y, x], dim=-1)

    pixels = img[indices[:, :, :, 0], :, indices[:, :, :, 1], indices[:, :, :, 2]]

    return pixels


# return (B, H, W, C)
def torch_warp(img, flow, H, W):
    # img = img.permute(0, 2, 3, 1)
    flow = flow.permute(0, 2, 3, 1)

    # Create a mesh grid
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    grid = torch.stack((x, y), dim=-1).unsqueeze(0)

    # Add flow to the grid
    flows = grid + flow

    # Clip the flow values to avoid going out of image boundaries
    max_y = H - 1
    max_x = W - 1
    x = flows[..., 0]
    y = flows[..., 1]
    x0 = x.floor().clamp(0, max_x).to(torch.int32)
    x1 = (x0 + 1).clamp(0, max_x)
    y0 = y.floor().clamp(0, max_y).to(torch.int32)
    y1 = (y0 + 1).clamp(0, max_y)

    # Get pixel values at corner coordinates
    Ia = torch_get_pixel_value(img, x0, y0)
    Ib = torch_get_pixel_value(img, x0, y1)
    Ic = torch_get_pixel_value(img, x1, y0)
    Id = torch_get_pixel_value(img, x1, y1)

    # Calculate the weights
    wa = (x1.type(torch.float32) - x) * (y1.type(torch.float32) - y)
    wb = (x1.type(torch.float32) - x) * (y - y0.type(torch.float32))
    wc = (x - x0.type(torch.float32)) * (y1.type(torch.float32) - y)
    wd = (x - x0.type(torch.float32)) * (y - y0.type(torch.float32))

    # Add dimensions for broadcasting
    wa = wa.unsqueeze(-1)
    wb = wb.unsqueeze(-1)
    wc = wc.unsqueeze(-1)
    wd = wd.unsqueeze(-1)

    # Compute the output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def get_pixel_value_test():
    B, C, H, W = 1, 3, 4, 4  # Batch size, Channels, Height, Width
    torch_img = torch.rand(B, C, H, W)
    tf_img = tf.convert_to_tensor(torch_img.permute(0, 2, 3, 1).numpy())

    # Flattening the x and y coordinate tensors
    x_coords = torch.tensor([0, 1, 2, 3]).repeat_interleave(4).view(B, H, W)
    y_coords = torch.tensor([0, 1, 2, 3]).repeat(4).view(B, H, W)

    # Calling both functions
    torch_output = torch_get_pixel_value(torch_img, x_coords, y_coords)
    tf_output = tf_get_pixel_value(tf_img, x_coords.numpy(), y_coords.numpy())

    # Comparing outputs
    print(np.allclose(torch_output.numpy(), tf_output.numpy()))
