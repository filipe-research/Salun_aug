import copy
import os
import time

import torch
import utils
import torch.nn.functional as F
from imagenet import get_x_y_from_data_dict

def _infer_class_weights_from_train_loader(train_loader, device):
    """Infer class weights from the dataset behind the loader.

    Assumes binary labels {0,1}. Ignores marked samples with negative labels.
    Returns a tensor [w0, w1] on `device`, or None if it cannot infer.
    """
    ds = getattr(train_loader, "dataset", None)
    if ds is None:
        return None

    labels = None
    if hasattr(ds, "labels"):
        labels = ds.labels
    elif hasattr(ds, "targets"):
        labels = ds.targets
    elif hasattr(ds, "_labels"):
        labels = ds._labels

    if labels is None:
        return None

    labels = torch.as_tensor(labels).long().view(-1)
    # ignore marked/removed samples
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return None

    n0 = int((labels == 0).sum().item())
    n1 = int((labels == 1).sum().item())
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        return None

    # Balanced CE weights: N/(2*Nc)
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)



def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False):
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

            loss = criterion(output_clean, target)

            if l1:
                loss = loss + args.alpha * l1_regularization(model)
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
    else:

        #code filipe
        
        cw = _infer_class_weights_from_train_loader(train_loader, torch.device("cuda"))
        if cw is not None:
            print(f"[AutoClassWeights] Using inferred CE weights: {cw.detach().cpu().tolist()}")

        #enc code filipe
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)

            # loss = criterion(output_clean, target) #old code

            #[filipe] weighted loss for medical datasets            
            if cw is not None:
                # Class-weighted CE (binary: [w_benign, w_malignant])
                loss = F.cross_entropy(output_clean, target, weight=cw)
            else:
                loss = criterion(output_clean, target)

            #end[filipe]
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
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
