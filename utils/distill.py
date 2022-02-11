import torch


def class_logits_from_subclass_logits(sc_logits, num_classes):
    num_subclasses = int(sc_logits.shape[-1])
    if num_subclasses % num_classes != 0:
        raise ValueError("Number of subclasses must be a multiple of number of classes")

    sc_logits = torch.reshape(sc_logits, (-1, num_classes, num_subclasses // num_classes))
    result = torch.logsumexp(sc_logits, -1)
    # result = torch.sum(sc_logits, -1)
    return result


def aux_loss(subclass_logits,device,temp=1):
    mean = torch.mean(subclass_logits, dim=1, keepdim=True)
    std = torch.std(subclass_logits, dim=1, keepdim=True)

    normalized_logits = (subclass_logits - mean) / std

    scaled_res = torch.matmul(normalized_logits, torch.transpose(normalized_logits, 0, 1))
    scaled_res = scaled_res / temp
    batch_size = torch.Tensor([normalized_logits.shape[0]]).to(device)
    return torch.mean(torch.logsumexp(scaled_res, -1)) - 1 / temp - torch.log(batch_size)
