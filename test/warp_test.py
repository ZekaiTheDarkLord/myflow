import tensorflow as tf
import torch
import numpy as np

from tf_warp import tensorflow_warp, tf_get_pixel_value
from torch_warp import torch_warp, torch_get_pixel_value
from utils import flow_warp


def create_test_data(H, W, C):
    # Create a test image and coordinate vectors
    img = np.random.rand(1, H, W, C).astype(np.float32)
    x = np.random.randint(0, W, size=(1, H, W))
    y = np.random.randint(0, H, size=(1, H, W))
    return img, x, y


def create_test_image_and_flow(H, W, C):
    # Create a test image and flow field
    img = np.random.rand(1, H, W, C).astype(np.float32)
    flow = np.random.rand(1, H, W, 2).astype(np.float32) * 2 - 1  # Flow values in range [-1, 1]
    return img, flow


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

def warp_test():
    H, W, C = 256, 256, 3  # Example size

    # Create test data
    img_np, flow_np = create_test_image_and_flow(H, W, C)

    # TensorFlow operations
    img_tf = tf.convert_to_tensor(img_np)
    flow_tf = tf.convert_to_tensor(flow_np)
    warped_img_tf = tensorflow_warp(img_tf, flow_tf, H, W)

    # Convert to PyTorch tensors
    img_torch = torch.from_numpy(np.transpose(img_np, (0, 3, 1, 2)))  # Change to PyTorch format
    flow_torch = torch.from_numpy(np.transpose(flow_np, (0, 3, 1, 2)))  # Change to PyTorch format
    warped_img_torch = torch_warp(img_torch, flow_torch, 256, 256)
    # warped_image_util = flow_warp(img_torch, flow_torch)

    # Convert the PyTorch output to numpy for comparison
    warped_img_torch_np = warped_img_torch.numpy()
    # warped_image_util_np = warped_image_util.numpy().transpose(0, 2, 3, 1)

    # Compare TensorFlow and PyTorch results
    if np.allclose(warped_img_tf.numpy(), warped_img_torch_np, atol=1e-6):
        print("The outputs are similar!")
    else:
        print("The outputs are different.")

        print(warped_img_tf)
        print(warped_img_torch_np)


if __name__ == "__main__":
    # img = torch.rand(B, H, W, C) # Your image tensor
    # flow = torch.rand(B, 2, H, W) # Your flow tensor (note the channel dimension is 2 for x and y flow components)
    # warped_img = torch_warp(img, flow)
    get_pixel_value_test()
    warp_test()
