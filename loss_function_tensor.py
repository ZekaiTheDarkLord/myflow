import os

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tf_warp


class TensorFlowLoss:
    def get_shape(x, train=True):
        if train:
            x_shape = x.get_shape().as_list()
        else:
            x_shape = tf.shape(x)
        return x_shape

    def abs_robust_loss(self, diff, mask, q=0.4):
        diff = tf.pow((tf.abs(diff) + 0.01), q)
        diff = tf.multiply(diff, mask)
        diff_sum = tf.reduce_sum(diff)
        loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
        return loss_mean

    def create_mask(self, tensor, paddings):
        # shape = tf.shape(tensor)
        # inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        # inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        # inner = tf.ones([inner_width, inner_height])
        #
        # mask2d = tf.pad(inner, paddings)
        # mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        # mask4d = tf.expand_dims(mask3d, 3)
        # return tf.stop_gradient(mask4d)
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return mask4d

    def census_loss(self, img1, img2_warped, mask, max_distance=3):
        patch_size = 2 * max_distance + 1

        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
            weights = tf.constant(w, dtype=tf.float32)
            # (1, 256, 256, 1), (7, 7, 1, 49)
            patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
            return dist_sum

        t1 = _ternary_transform(img1)
        t2 = _ternary_transform(img2_warped)
        dist = _hamming_distance(t1, t2)

        transform_mask = self.create_mask(mask, [[max_distance, max_distance],
                                                 [max_distance, max_distance]])
        return self.abs_robust_loss(dist, mask * transform_mask)

    def compute_losses(self, batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=True, is_scale=True):
        img_size = tf.shape(batch_img1)
        img1_warp = tf_warp.tensorflow_warp(batch_img1, flow_bw['full_res'], img_size[1], img_size[2])
        img2_warp = tf_warp.tensorflow_warp(batch_img2, flow_fw['full_res'], img_size[1], img_size[2])

        losses = {}

        abs_robust_mean = {}
        abs_robust_mean['no_occlusion'] = self.abs_robust_loss(batch_img1 - img2_warp,
                                                               tf.ones_like(mask_fw)) + self.abs_robust_loss(
            batch_img2 - img1_warp, tf.ones_like(mask_bw))
        abs_robust_mean['occlusion'] = self.abs_robust_loss(batch_img1 - img2_warp, mask_fw) + self.abs_robust_loss(
            batch_img2 - img1_warp, mask_bw)
        losses['abs_robust_mean'] = abs_robust_mean

        census_loss = {}
        census_loss['no_occlusion'] = self.census_loss(batch_img1, img2_warp, tf.ones_like(mask_fw), max_distance=3) + \
                                      self.census_loss(batch_img2, img1_warp, tf.ones_like(mask_bw), max_distance=3)
        census_loss['occlusion'] = self.census_loss(batch_img1, img2_warp, mask_fw, max_distance=3) + \
                                   self.census_loss(batch_img2, img1_warp, mask_bw, max_distance=3)
        losses['census'] = census_loss

        return losses

# def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
#     for key in keys:
#         for loss_key, loss_value in losses[key].items():
#             if prefix:
#                 loss_name = '%s/%s/%s' % (prefix, key, loss_key)
#             else:
#                 loss_name = '%s/%s' % (key, loss_key)
#             tf.summary.scalar(loss_name, loss_value)
#
#     abs_robust_mean = {}
#     abs_robust_mean['no_occlusion'] = self.abs_robust_loss(batch_img1 - img2_warp,
#                                                            tf.ones_like(mask_fw)) + self.abs_robust_loss(
#         batch_img2 - img1_warp, tf.ones_like(mask_bw))
#     abs_robust_mean['occlusion'] = self.abs_robust_loss(batch_img1 - img2_warp, mask_fw) + self.abs_robust_loss(
#         batch_img2 - img1_warp, mask_bw)
#     losses['abs_robust_mean'] = abs_robust_mean
#
#     census_loss = {}
#     census_loss['no_occlusion'] = self.census_loss(batch_img1, img2_warp, tf.ones_like(mask_fw), max_distance=3) + \
#                                   self.census_loss(batch_img2, img1_warp, tf.ones_like(mask_bw), max_distance=3)
#     census_loss['occlusion'] = self.census_loss(batch_img1, img2_warp, mask_fw, max_distance=3) + \
#                                self.census_loss(batch_img2, img1_warp, mask_bw, max_distance=3)
#     losses['census'] = census_loss
#
#     return losses
