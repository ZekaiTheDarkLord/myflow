import numpy as np
import torch

from loss_function import MyFLow_Loss
import tensorflow as tf

from loss_function_tensor import TensorFlowLoss


def test_abs_robust_loss():
    # Test parameters
    size = (10, 12)  # Example tensor size
    q = 0.4  # Power value
    tolerance = 1e-6  # Tolerance for comparison

    # Create random inputs
    diff = np.random.rand(*size).astype(np.float32)
    mask = np.random.rand(*size).astype(np.float32)

    # TensorFlow loss
    tf_loss_val = TensorFlowLoss().abs_robust_loss(tf.constant(diff), tf.constant(mask), q).numpy()

    # PyTorch loss
    torch_loss_val = MyFLow_Loss().abs_robust_loss(torch.tensor(diff), torch.tensor(mask), q).detach().numpy()

    # Check if the losses are close
    assert np.allclose(tf_loss_val, torch_loss_val, atol=tolerance), "Losses do not match"
    print("Test passed: TensorFlow and PyTorch losses are equivalent.")


def test_create_mask():
    # Test parameters
    tensor_size = (1, 10, 12, 3)  # Example size (batch, height, width, channels)
    paddings = ((1, 1), (1, 1))  # Padding for height and width

    # Create random input tensors
    tensor_tf = tf.random.normal(tensor_size)
    tensor_torch = torch.rand(tensor_size)

    # TensorFlow mask
    tf_mask = TensorFlowLoss().create_mask(tensor_tf, paddings).numpy()

    # PyTorch mask
    torch_mask = MyFLow_Loss().create_mask(tensor_torch, paddings).numpy()

    # Check if the masks are identical
    assert np.array_equal(tf_mask, torch_mask), "Masks do not match"
    print("Test passed: TensorFlow and PyTorch masks are equivalent.")


def test_census_loss():
    # Test parameters
    img_size = (1, 3, 256, 128)  # Example size (batch, channels, height, width)
    mask_size = (1, 1, 256, 128)  # Mask size
    tolerance = 1e-6  # Tolerance for comparison

    # Create random inputs
    img1 = np.random.rand(*img_size).astype(np.float32)
    img2 = np.random.rand(*img_size).astype(np.float32)
    mask = np.random.rand(*mask_size).astype(np.float32)

    # TensorFlow census loss
    tf_img1 = tf.constant(img1.transpose(0, 2, 3, 1))
    tf_img2 = tf.constant(img2.transpose(0, 2, 3, 1))
    tf_mask = tf.constant(mask.transpose(0, 2, 3, 1))
    tf_loss = TensorFlowLoss().census_loss(tf_img1, tf_img2, tf_mask).numpy()

    # PyTorch census loss
    torch_img1 = torch.tensor(img1)
    torch_img2 = torch.tensor(img2)
    torch_mask = torch.tensor(mask)
    torch_loss = MyFLow_Loss().census_loss(torch_img1, torch_img2, torch_mask).detach().numpy()

    # Check if the losses are close
    # print(tf_loss, torch_loss)
    assert np.allclose(tf_loss, torch_loss, atol=tolerance), "Losses do not match"
    print("Test passed: TensorFlow and PyTorch census losses are equivalent.")


def test_compute_losses():
    # Test parameters
    img_size = (1, 3, 256, 256)  # Example size (batch, channels, height, width)
    flow_size = (1, 2, 256, 256)  # Flow vector size
    mask_size = (1, 1, 256, 256)  # Mask size

    # Create random inputs
    batch_img1 = torch.rand(img_size)
    batch_img2 = torch.rand(img_size)
    flow_fw = {'full_res': torch.rand(flow_size)}
    flow_bw = {'full_res': torch.rand(flow_size)}
    mask_fw = torch.rand(mask_size)
    mask_bw = torch.rand(mask_size)

    # Convert inputs to TensorFlow tensors
    batch_img1_tf = tf.convert_to_tensor(batch_img1.numpy().transpose(0, 2, 3, 1))
    batch_img2_tf = tf.convert_to_tensor(batch_img2.numpy().transpose(0, 2, 3, 1))
    flow_fw_tf = {'full_res': tf.convert_to_tensor(flow_fw['full_res'].numpy().transpose(0, 2, 3, 1))}
    flow_bw_tf = {'full_res': tf.convert_to_tensor(flow_bw['full_res'].numpy().transpose(0, 2, 3, 1))}
    mask_fw_tf = tf.convert_to_tensor(mask_fw.numpy().transpose(0, 2, 3, 1))
    mask_bw_tf = tf.convert_to_tensor(mask_bw.numpy().transpose(0, 2, 3, 1))

    # Compute losses for both frameworks
    losses_torch = MyFLow_Loss().compute_losses(batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw)
    losses_tf = TensorFlowLoss().compute_losses(batch_img1_tf, batch_img2_tf, flow_fw_tf,
                                                flow_bw_tf, mask_fw_tf, mask_bw_tf)

    print("torch: ", losses_torch)
    print("tfL ", losses_tf)


if __name__ == "__main__":
    test_abs_robust_loss()
    test_create_mask()
    test_census_loss()
    test_compute_losses()
