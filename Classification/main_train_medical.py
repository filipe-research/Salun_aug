import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate_medical
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


def main():
    global args, best_sa
    args = arg_parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    else:
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
        ) = setup_model_dataset(args)
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.imagenet_arch:
        lambda0 = (
            lambda cur_iter: (cur_iter + 1) / args.warmup
            if cur_iter < args.warmup
            else (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
                    )
                )
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )  # 0.1 is fixed
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint.get("best_bac", checkpoint.get("best_sa", 0))
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]
        best_epoch = checkpoint.get("best_epoch", -1)

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initalization = checkpoint["init_weight"]
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)

    else:
        all_result = {}
        all_result["train_ta"] = []
        all_result["val_bac"] = []
        all_result["test_metrics"] = {}

        best_epoch = -1
        start_epoch = 0
        state = 0

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        metrics = validate_medical(val_loader, model, criterion, args)
        scheduler.step()

        # Track best epoch by validation BAC
        cur_bac = float(metrics.get('bac', 0.0))
        is_best_sa = cur_bac > best_sa
        if is_best_sa:
            best_sa = cur_bac
            best_epoch = epoch
            # Save an explicit best checkpoint (independent of save_checkpoint naming)
            best_ckpt_path = os.path.join(args.save_dir, "best_bac_checkpoint.pth.tar")
            torch.save(
                {
                    "result": all_result,
                    "epoch": epoch + 1,
                    "best_bac": best_sa,
                    "best_epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                best_ckpt_path,
            )

        all_result["train_ta"].append(acc)
        all_result["val_bac"].append(cur_bac)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_bac": best_sa,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )
        print("one epoch duration:{}".format(time.time() - start_time))

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    # plt.plot(all_result["val_ta"], label="val_acc")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
    plt.close()

    print("Performance on the test data set")

    # Load best checkpoint (by validation BAC) and evaluate on test set
    best_ckpt_path = os.path.join(args.save_dir, "best_bac_checkpoint.pth.tar")
    if os.path.isfile(best_ckpt_path):
        print(f"Loading best checkpoint: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location=torch.device("cuda:" + str(args.gpu)))
        model.load_state_dict(best_ckpt["state_dict"], strict=False)
        best_sa_loaded = float(best_ckpt.get("best_bac", best_sa))
        best_epoch_loaded = int(best_ckpt.get("best_epoch", best_epoch))
    else:
        print(f"WARNING: best checkpoint not found at {best_ckpt_path}. Using last epoch weights.")
        best_sa_loaded = float(best_sa)
        best_epoch_loaded = int(best_epoch)

    model.eval()
    test_metrics = validate_medical(test_loader, model, criterion, args)
    # store and print
    all_result["test_metrics"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()}

    print("\n==== Best-by-BAC summary ====")
    print(f"Best val BAC: {best_sa_loaded:.6f} | Best epoch: {best_epoch_loaded + 1 if best_epoch_loaded >= 0 else 'N/A'}")
    print("==== Test metrics (best checkpoint) ====")
    for k in sorted(test_metrics.keys()):
        try:
            print(f"{k}: {float(test_metrics[k]):.6f}")
        except Exception:
            print(f"{k}: {test_metrics[k]}")

    # Save metrics to disk
    with open(os.path.join(args.save_dir, "test_metrics.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    with open(os.path.join(args.save_dir, "test_metrics.txt"), "w") as f:
        f.write("Best checkpoint (by val BAC)\n")
        f.write(f"best_val_bac: {best_sa_loaded:.6f}\n")
        f.write(f"best_epoch: {best_epoch_loaded + 1 if best_epoch_loaded >= 0 else -1}\n")
        f.write("\nTest metrics\n")
        for k in sorted(test_metrics.keys()):
            try:
                f.write(f"{k}: {float(test_metrics[k]):.8f}\n")
            except Exception:
                f.write(f"{k}: {test_metrics[k]}\n")

    # Save training summary
    # with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
    #     f.write(f"Best val BAC: {best_sa:.6f}\n")
    #     f.write(f"Best epoch (1-based): {best_epoch + 1 if best_epoch >= 0 else -1}\n")
    #     if len(all_result.get("val_bac", [])):
    #         f.write(f"Final val BAC: {all_result['val_bac'][-1]:.6f}\n")
    

if __name__ == "__main__":
    main()