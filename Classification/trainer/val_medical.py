import torch
import torch.nn.functional as F
import math
import utils
from imagenet import get_x_y_from_data_dict


# --- Helper functions for binary classification metrics ---

def _as_positive_probs(output: torch.Tensor) -> torch.Tensor:
    """Return P(y=1) for binary classification.

    Supports outputs shaped (N,), (N,1), or (N,2). Assumes class 1 is the
    positive (e.g., malignant) class.
    """
    if output.dim() == 1:
        # logits for positive class
        return torch.sigmoid(output)
    if output.dim() == 2:
        if output.size(1) == 1:
            return torch.sigmoid(output[:, 0])
        if output.size(1) == 2:
            return F.softmax(output, dim=1)[:, 1]
    raise ValueError(f"Unsupported output shape for binary probs: {tuple(output.shape)}")


def _confusion_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor):
    """y_true/y_pred are 0/1 tensors."""
    y_true = y_true.long()
    y_pred = y_pred.long()
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return tp, tn, fp, fn


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def _roc_auc_score(y_true, y_score) -> float:
    """Pure-python ROC AUC using rank statistic (equivalent to Mann–Whitney U).

    Returns 0.0 if only one class is present.
    """
    y_true = [int(v) for v in y_true]
    y_score = [float(v) for v in y_score]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by score ascending
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])

    # Compute average ranks for ties
    ranks = [0.0] * len(pairs)
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        rank += (j - i)
        i = j

    sum_ranks_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true, y_score) -> float:
    """Average Precision (area under PR curve) in pure python.

    Returns 0.0 if no positives.
    """
    y_true = [int(v) for v in y_true]
    y_score = [float(v) for v in y_score]
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    # Sort by score descending
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, n_pos)
        # Step-wise area (like sklearn's average_precision_score)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)


def _compute_binary_report(y_true: torch.Tensor, y_prob_pos: torch.Tensor, threshold: float = 0.5,
                           cost_fp: float = 1.0, cost_fn_1: float = 1.0, cost_fn_2: float = 20.0):
    """Compute binary classification + clinical risk metrics.

    Assumes y_true in {0,1} and y_prob_pos in [0,1]. Positive class is 1 (e.g., malignant).
    """
    y_true = y_true.detach().cpu().long()
    y_prob_pos = y_prob_pos.detach().cpu().float().clamp(0.0, 1.0)

    y_pred = (y_prob_pos >= threshold).long()
    tp, tn, fp, fn = _confusion_from_preds(y_true, y_pred)

    precision = _safe_div(tp, tp + fp)  # PPV
    npv = _safe_div(tn, tn + fn)
    recall = _safe_div(tp, tp + fn)  # Sensitivity / TPR
    fnr = _safe_div(fn, tp + fn)
    specificity = _safe_div(tn, tn + fp)  # TNR
    bac = 0.5 * (recall + specificity)
    f1 = _safe_div(2 * precision * recall, precision + recall)  # optional

    y_true_list = y_true.tolist()
    y_score_list = y_prob_pos.tolist()
    auc = _roc_auc_score(y_true_list, y_score_list)
    auprc = _average_precision(y_true_list, y_score_list)

    n_total = len(y_true_list)
    n_ben = int((y_true == 0).sum().item())
    n_mal = int((y_true == 1).sum().item())

    # Risk I (FN cost = 1)
    risk_global_1 = _safe_div(cost_fp * fp + cost_fn_1 * fn, n_total)
    risk_ben_1 = _safe_div(cost_fp * fp, n_ben)
    risk_mal_1 = _safe_div(cost_fn_1 * fn, n_mal)

    # Risk II (FN cost = 20)
    risk_global_2 = _safe_div(cost_fp * fp + cost_fn_2 * fn, n_total)
    risk_ben_2 = _safe_div(cost_fp * fp, n_ben)
    risk_mal_2 = _safe_div(cost_fn_2 * fn, n_mal)

    report = {
        "precision_ppv": precision,
        "npv": npv,
        "recall": recall,
        "fnr": fnr,
        "specificity": specificity,
        "bac": bac,
        "auc": auc,
        "auprc": auprc,
        "risk_global_I": risk_global_1,
        "risk_benign_I": risk_ben_1,
        "risk_malign_I": risk_mal_1,
        "risk_global_II": risk_global_2,
        "risk_benign_II": risk_ben_2,
        "risk_malign_II": risk_mal_2,
        # helpful extras
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
        "f1": f1,
    }
    return report


def validate_medical(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    all_probs = []
    all_targets = []

    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # accumulate probabilities and targets for metrics
            all_probs.append(_as_positive_probs(output).detach().cpu())
            all_targets.append(target.detach().cpu())

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # accumulate probabilities and targets for metrics
            all_probs.append(_as_positive_probs(output).detach().cpu())
            all_targets.append(target.detach().cpu())

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        #print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
        print("Avg Test accuracy {top1.avg:.3f}".format(top1=top1))

    # --- aggregate metrics (binary classification) ---
    y_prob = torch.cat(all_probs, dim=0) if len(all_probs) else torch.tensor([])
    y_true = torch.cat(all_targets, dim=0) if len(all_targets) else torch.tensor([])

    if y_true.numel() == 0:
        return {
            "precision_ppv": 0.0,
            "npv": 0.0,
            "recall": 0.0,
            "fnr": 0.0,
            "specificity": 0.0,
            "bac": 0.0,
            "auc": 0.0,
            "auprc": 0.0,
            "risk_global_I": 0.0,
            "risk_benign_I": 0.0,
            "risk_malign_I": 0.0,
            "risk_global_II": 0.0,
            "risk_benign_II": 0.0,
            "risk_malign_II": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "threshold": 0.5,
        }

    # NOTE: assumes positive class is label 1 (e.g., malignant)
    metrics = _compute_binary_report(y_true=y_true, y_prob_pos=y_prob, threshold=0.5,
                                     cost_fp=1.0, cost_fn_1=1.0, cost_fn_2=20.0)

    # Print a concise summary
    print(
        "Metrics: "
        f"PPV={metrics['precision_ppv']:.4f} "
        f"NPV={metrics['npv']:.4f} "
        f"TPR={metrics['recall']:.4f} "
        f"FNR={metrics['fnr']:.4f} "
        f"TNR={metrics['specificity']:.4f} "
        f"BAC={metrics['bac']:.4f} "
        f"AUC={metrics['auc']:.4f} "
        f"AUPRC={metrics['auprc']:.4f} "
        f"R1={metrics['risk_global_I']:.6f} "
        f"R2={metrics['risk_global_II']:.6f}"
    )

    return metrics
