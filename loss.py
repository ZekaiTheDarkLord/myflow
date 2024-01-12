import torch


# kl_loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)

def flow_loss_func(flow_preds, flow_gt_list, valid_list,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt_list[-1] ** 2, dim=1).sqrt()  # [B, H, W]
    # valid = (valid >= 0.5) & (mag < max_flow)

    scales = [2, 2, 1]
    weights = [0.6, 0.8, 1]

    for i in range(n_predictions):
        # i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt_list[i]).abs() * scales[i]

        flow_loss += weights[i] * (valid_list[i][:, None] * i_loss).mean()

    # flatten_valid = valid.view(valid.shape[0], -1, 1).repeat(1,1,prob_gt.shape[-2])
    # corr = flow_preds[-2] * flatten_valid
    # corr = torch.log_softmax(corr, dim=-1).transpose(-1,-2)

    # prob_loss = kl_loss(corr, prob_gt) / torch.count_nonzero(valid)

    # flow_loss += 1.0 * (valid[:, None] * (flow_preds[-1] - flow_gt).abs()).mean()

    # print(torch.nonzero(valid.view(valid.shape[0], -1)))
    # for b_i, f_i in torch.nonzero(torch.nonzero(valid.view(valid.shape[0], -1))):
    #     print(torch.nonzero(corr[b_i][f_i] > 0.01))
    #     print(torch.nonzero(prob_gt[b_i][f_i] > -10))
    #     print()

    epe = torch.sum((flow_preds[-1] - flow_gt_list[-1]) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_list[-1].view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        # '1px': (torch.count_nonzero(epe<1)/len(epe)).item(),
        # '3px': (torch.count_nonzero(epe<3)/len(epe)).item(),
        # '5px': (torch.count_nonzero(epe<5)/len(epe)).item(),
        'mag': mag[valid_list[-1]].float().mean().item()
    }

    return flow_loss, metrics
