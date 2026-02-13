import sys
import time

import torch
import utils
import torch.nn.functional as F

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

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


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, with_l1=False
):
    train_loader = data_loaders["retain"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            loss = criterion(output_clean, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:

        # infer from retain distribution (do not use forget, since it may have randomized labels)
        cw = _infer_class_weights_from_dataset(train_loader.dataset, device)
        if cw is not None:
            print(f"[AutoClassWeights][RL] Using inferred CE weights: {cw.detach().cpu().tolist()}")

        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            # loss = criterion(output_clean, target)

            if cw is not None:
                loss = F.cross_entropy(output_clean, target, weight=cw)
            else:
                loss = criterion(output_clean, target)

            if with_l1:
                loss += current_alpha * l1_regularization(model)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(
        data_loaders, model, criterion, optimizer, epoch, args, mask, with_l1=True
    )
