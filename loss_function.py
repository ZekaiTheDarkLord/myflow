import torch
import torch.nn.functional as F
from torchvision import transforms
from torch_warp import torch_warp


class MyFLow_Loss:
    def abs_robust_loss(self, diff, mask, q=0.4):
        diff = torch.pow((torch.abs(diff) + 0.01), q)
        diff = diff * mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(mask) * 2 + 1e-6)
        return loss_mean

    def create_mask(self, tensor, paddings):
        shape = tensor.shape
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = torch.ones((inner_width, inner_height), dtype=torch.float32)

        # Flatten the padding tuple
        flat_paddings = (paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1])

        mask2d = torch.nn.functional.pad(inner, flat_paddings)
        mask3d = mask2d.unsqueeze(0).expand(shape[0], -1, -1)
        mask4d = mask3d.unsqueeze(3)
        return mask4d

    def census_loss(self, img1, img2_warped, mask, max_distance=3):
        patch_size = 2 * max_distance + 1

        def _ternary_transform(image):
            # intensities = torch.mean(image, dim=1, keepdim=True) * 255
            grayscale_transform = transforms.Grayscale(num_output_channels=1)
            image_gray = grayscale_transform(image)
            intensities = image_gray * 255

            out_channels = patch_size * patch_size
            w = torch.eye(out_channels).reshape((out_channels, 1, patch_size, patch_size))
            weights = torch.tensor(w, dtype=torch.float32)
            patches = F.conv2d(intensities, weights, padding=max_distance)

            transf = patches - intensities
            transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = (t1 - t2) ** 2
            dist_norm = dist / (0.1 + dist)
            dist_sum = torch.sum(dist_norm, dim=3, keepdim=True)
            return dist_sum

        t1 = _ternary_transform(img1)
        t2 = _ternary_transform(img2_warped)
        dist = _hamming_distance(t1.permute(0, 2, 3, 1), t2.permute(0, 2, 3, 1))

        permute_mask = mask.permute(0, 2, 3, 1)
        transform_mask = self.create_mask(permute_mask, [(max_distance, max_distance), (max_distance, max_distance)])
        return self.abs_robust_loss(dist, permute_mask * transform_mask)

    def compute_losses(self, batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=True, is_scale=True):
        img_size = batch_img1.size()
        img1_warp = torch_warp(batch_img1, flow_bw['full_res'], img_size[2], img_size[3]).permute(0, 3, 1, 2)
        img2_warp = torch_warp(batch_img2, flow_fw['full_res'], img_size[2], img_size[3]).permute(0, 3, 1, 2)

        losses = {}

        abs_robust_mean = {}
        abs_robust_mean['no_occlusion'] = (self.abs_robust_loss(batch_img1 - img2_warp, torch.ones_like(mask_fw))
                                           + self.abs_robust_loss(batch_img2 - img1_warp, torch.ones_like(mask_bw)))
        abs_robust_mean['occlusion'] = (self.abs_robust_loss(batch_img1 - img2_warp, mask_fw)
                                        + self.abs_robust_loss(batch_img2 - img1_warp, mask_bw))
        losses['abs_robust_mean'] = abs_robust_mean

        census_loss = {}
        census_loss['no_occlusion'] = self.census_loss(batch_img1, img2_warp, torch.ones_like(mask_fw),
                                                       max_distance=3) + self.census_loss(batch_img2, img1_warp,
                                                                                          torch.ones_like(mask_bw),
                                                                                          max_distance=3)
        census_loss['occlusion'] = self.census_loss(batch_img1, img2_warp, mask_fw, max_distance=3) + self.census_loss(
            batch_img2, img1_warp, mask_bw, max_distance=3)
        losses['census'] = census_loss

        return losses

    def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
        # Add loss summaries using your preferred logging method (e.g., TensorBoard)
        pass
