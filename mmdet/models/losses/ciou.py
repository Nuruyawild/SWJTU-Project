def ciou_loss(pred, target, reduce='mean', eps=1e-6):
    """
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction: "mean" or "sum"
    return: loss
    """
    # get the area of pred, target
    pred_widths = (pred[:, 2] - pred[:, 0] + 1.).clamp(0)
    pred_heights = (pred[:, 3] - pred[:, 1] + 1.).clamp(0)
    target_widths = (target[:, 2] - target[:, 0] + 1.).clamp(0)
    target_heights = (target[:, 3] - target[:, 1] + 1.).clamp(0)
    pred_areas = pred_widths * pred_heights
    target_areas = target_widths * target_heights

    # get the intersecting area between pred and target 
    inter_xmins = torch.maximum(pred[:, 0], target[:, 0])
    inter_ymins = torch.maximum(pred[:, 1], target[:, 1])
    inter_xmaxs = torch.minimum(pred[:, 2], target[:, 2])
    inter_ymaxs = torch.minimum(pred[:, 3], target[:, 3])
    inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
    inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
    inter_areas = inter_widths * inter_heights

    # iou
    unions = pred_areas + target_areas - inter_areas + eps
    ious = torch.clamp(inter_areas / unions, min=eps)

    # Find the distance between the diagonals of the smallest external rectangle
    outer_xmins = torch.minimum(pred[:, 0], target[:, 0])
    outer_ymins = torch.minimum(pred[:, 1], target[:, 1])
    outer_xmaxs = torch.maximum(pred[:, 2], target[:, 2])
    outer_ymaxs = torch.maximum(pred[:, 3], target[:, 3])
    outer_diag = torch.clamp((outer_xmaxs - outer_xmins + 1.), min=0.) ** 2 + \
        torch.clamp((outer_ymaxs - outer_ymins + 1.), min=0.) ** 2 + eps

    # Find the distance between the centre of the pred and target box
    c_pred = ((pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2)
    c_target = ((target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2)
    distance = (c_pred[0] - c_target[0] + 1.) ** 2 + (c_pred[1] - c_target[1] + 1.) ** 2

    # Find the loss on the shape of the prediction frame
    w_pred, h_pred = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1] + eps
    w_target, h_target = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1] + eps
    factor = 4 / (math.pi ** 2)
    v = factor * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_target / h_target), 2)
    alpha = v / (1 - ious + v)

    # ciou loss
    cious = ious - distance / outer_diag - alpha * v
    if reduce == 'mean':
        loss = torch.mean(1 - cious)
    elif reduce == 'sum':
        loss = torch.sum(1 - cious)
    else:
        raise NotImplementedError

    return loss