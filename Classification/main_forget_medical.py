import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate_medical

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    elif args.dataset in ["bloodmnist", "pathmnist", "organamnist", "octmnist", "dermamnist_bin"]:
        
        marked = forget_dataset.labels < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.labels = -forget_dataset.labels[marked] - 1
        
        forget_dataset.info["n_samples"]["train"] = forget_dataset.imgs.shape[0]
        
        forget_loader = replace_loader_dataset(
            forget_dataset, seed=seed, shuffle=True
        )
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        marked = retain_dataset.labels >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.labels = retain_dataset.labels[marked]
        
        retain_dataset.info["n_samples"]["train"] = retain_dataset.imgs.shape[0]
        retain_loader = replace_loader_dataset(
            retain_dataset, seed=seed, shuffle=True
        )
        
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        # accuracy = {}
        results = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            # val_acc = validate(loader, model, criterion, args)
            metrics = validate_medical(loader, model, criterion, args)
            # accuracy[name] = val_acc
            results[name] = metrics
            #print(f"{name} acc: {val_acc}")
            print(f"{name} bac: {metrics['bac']}")

        # evaluation_result["accuracy"] = accuracy
        # evaluation_result["bac"] = results
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    print("==== Test metrics (best checkpoint) ====")
    test_metrics = results["test"]
    for k in sorted(test_metrics.keys()):
        try:
            print(f"{k}: {float(test_metrics[k]):.6f}")
        except Exception:
            print(f"{k}: {test_metrics[k]}")
    
    #UA = 100 - evaluation_result["accuracy"]["forget"]
    #UBAC = 1 - evaluation_result["bac"]["forget"]
    UBAC = 1 - results["forget"]["bac"]
    print(f"UBAC (Unlearning Accuracy): {UBAC:.2f}%")

    # RA = evaluation_result["saccuracy"]["retain"]
    #RBAC = evaluation_result["bac"]["retain"]
    RBAC = results["retain"]["bac"]
    print(f"RBAC (Remaining Accuracy): {RBAC:.2f}%")

    # TA = evaluation_result["accuracy"]["test"]
    # TBAC = evaluation_result["bac"]["test"]
    TBAC = results["test"]["bac"]
    print(f"TBAC (Testing Accuracy): {TBAC:.2f}%")

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        # shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(min(retain_len, test_len)))) #Filipe's update
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    MIA = evaluation_result["SVC_MIA_forget_efficacy"]['confidence'] * 100
    print(f"MIA (Membership Inference Attack): {MIA:.2f}")
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
        f.write(f"UBAC (Unlearning Accuracy): {UBAC:.2f}%")
        f.write(f"RBAC (Remaining Accuracy): {RBAC:.2f}%")
        f.write(f"TBAC (Testing Accuracy): {TBAC:.2f}%")
        f.write(f"MIA (Membership Inference Attack): {MIA:.2f}")


if __name__ == "__main__":
    main()