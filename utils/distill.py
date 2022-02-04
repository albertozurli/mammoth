import torch


def class_logits_from_subclass_logits(sc_logits, num_classes):
    num_subclasses = int(sc_logits.shape[-1])
    if num_subclasses % num_classes != 0:
        raise ValueError("Number of subclasses must be a multiple of number of classes")

    sc_logits = torch.reshape(sc_logits, (-1, num_classes, num_subclasses // num_classes))
    result = torch.logsumexp(sc_logits, -1)
    # result = torch.sum(sc_logits, -1)
    return result