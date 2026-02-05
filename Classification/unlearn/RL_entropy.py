import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import utils



def _infer_class_weights_from_dataset(ds, device):
    """Infer balanced CE weights [w0, w1] from a dataset-like object.

    Assumes binary labels {0,1}. Ignores marked samples with negative labels.
    Works with objects that expose `.labels` or `.targets`.
    """
    labels = None
    if hasattr(ds, "labels"):
        labels = ds.labels
    elif hasattr(ds, "targets"):
        labels = ds.targets

    if labels is None:
        return None

    labels = torch.as_tensor(labels).long().view(-1)
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return None

    n0 = int((labels == 0).sum().item())
    n1 = int((labels == 1).sum().item())
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        return None

    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def _max_entropy_loss(logits):
    """Compute the negative entropy of the softmax distribution.

    Minimizing this loss pushes the model output towards a uniform
    distribution over classes, i.e. maximum uncertainty.

    L = sum_c p_c * log(p_c)   (negative entropy, so minimizing => max entropy)
    """
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    entropy = -(p * log_p).sum(dim=1)  # per-sample entropy
    # We want to MAXIMIZE entropy, so we MINIMIZE negative entropy
    return -entropy.mean()


from .impl import iterative_unlearn


@iterative_unlearn
def RL_entropy(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    # Class-weighted CE for retain set (helpful for imbalanced medical datasets)
    device = torch.device("cuda")

    # infer from retain distribution (do not use forget, since it may have randomized labels)
    cw = _infer_class_weights_from_dataset(retain_loader.dataset, device)
    if cw is not None:
        print(f"[AutoClassWeights][RL] Using inferred CE weights: {cw.detach().cpu().tolist()}")

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)

    if epoch < args.warmup:
        utils.warmup_lr(epoch, 1, optimizer,
                        one_epoch_step=loader_len, args=args)

    # --- Phase 1: Forget set  -> maximize entropy (unlearn) ---
    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        # target is not used for the loss, but we keep it for logging
        output_clean = model(image)
        loss = _max_entropy_loss(output_clean)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Forget-Entropy {loss:.4f}\t'
                  'Time {time:.2f}'.format(
                      epoch, i, len(forget_loader),
                      loss=loss.item(), time=end - start))
            start = time.time()

    # --- Phase 2: Retain set -> standard CE (preserve knowledge) ---
    for i, (image, target) in enumerate(retain_loader):
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        if cw is not None:
            loss = F.cross_entropy(output_clean, target, weight=cw)
        else:
            loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                      epoch, i + len(forget_loader), loader_len,
                      end - start, loss=losses, top1=top1))
            start = time.time()

    return top1.avg