"""
SalUn Híbrido com Consciência de Risco Clínico
===============================================

Modificação do SalUn original para reduzir risco clínico em bases médicas binárias.

Problema do SalUn original:
- Randomiza labels de TODOS os exemplos do forget set
- ~50% dos malignos recebem label "benigno"
- Modelo aprende ativamente que malignos são benignos → aumenta FN

Solução proposta:
- Exemplos MALIGNOS (classe 1) do forget set → entropia máxima
  (modelo fica incerto, mas não aprende que é benigno)
- Exemplos BENIGNOS (classe 0) do forget set → random labeling tradicional
  (pode virar maligno, o que até ajuda sensibilidade)
- Retain set → treino normal com CE ponderada

Isso preserva o mecanismo de desaprendizado enquanto evita a "contaminação"
da fronteira de decisão que aumenta falsos negativos.
"""

import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import utils


def _infer_class_weights_from_dataset(ds, device):
    """Infer balanced CE weights [w0, w1] from a dataset-like object."""
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
    """Compute negative entropy loss.
    
    Minimizing this pushes model output towards uniform distribution,
    i.e., maximum uncertainty.
    """
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    entropy = -(p * log_p).sum(dim=1)
    # Minimize negative entropy = maximize entropy
    return -entropy.mean()


def _get_original_labels(dataset):
    """Extract original labels from dataset before any modification."""
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels).copy()
    elif hasattr(dataset, "targets"):
        if isinstance(dataset.targets, np.ndarray):
            return dataset.targets.copy()
        else:
            return np.array(dataset.targets).copy()
    return None


from .impl import iterative_unlearn


@iterative_unlearn
def RL_hibrid(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    """
    SalUn Híbrido: Entropia para malignos + Random Label para benignos
    
    Para bases médicas binárias onde classe 1 = maligno e classe 0 = benigno.
    """
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    
    device = torch.device("cuda")
    
    # Infer class weights from retain distribution
    cw = _infer_class_weights_from_dataset(retain_loader.dataset, device)
    if cw is not None:
        print(f"[HybridSalUn] CE weights: {cw.detach().cpu().tolist()}")
    
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet" or args.dataset in ["bloodmnist", "pathmnist", "organamnist", "octmnist", "dermamnist_bin", "pneumoniamnist", "breastmnist"]:
        
        # ============================================================
        # MODIFICAÇÃO PRINCIPAL: Tratamento assimétrico por classe
        # ============================================================
        
        # Obter labels originais ANTES de qualquer modificação
        if args.dataset in ["bloodmnist", "pathmnist", "organamnist", "octmnist", "dermamnist_bin", "pneumoniamnist", "breastmnist"]:
            original_labels = _get_original_labels(forget_dataset)
            
            # Criar máscara: True para malignos (classe 1), False para benignos (classe 0)
            # Flatten para garantir 1D
            if original_labels.ndim > 1:
                original_labels = original_labels.flatten()
            
            malignant_mask = (original_labels == 1)
            benign_mask = (original_labels == 0)
            
            n_malignant = malignant_mask.sum()
            n_benign = benign_mask.sum()
            print(f"[HybridSalUn] Forget set: {n_malignant} malignos (entropia), {n_benign} benignos (random label)")
            
            # Para benignos: random labeling tradicional
            # Para malignos: manter label original (será tratado com entropia no loop)
            new_targets = original_labels.copy()
            new_targets[benign_mask] = np.random.randint(0, args.num_classes, size=n_benign)
            # Malignos mantêm label original - serão identificados no loop
            
            forget_dataset.targets = new_targets
            
            # Salvar máscara de malignos para uso no loop
            # Precisamos saber quais índices são malignos para aplicar entropia
            forget_dataset._malignant_mask = malignant_mask
            
        else:
            try:
                forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
            except:
                forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
    
        retain_dataset = retain_loader.dataset
        
        # ============================================================
        # Criar loaders separados para forget e retain
        # Isso permite tratamento diferenciado no forget set
        # ============================================================
        
        forget_train_loader = torch.utils.data.DataLoader(
            forget_dataset, batch_size=args.batch_size, shuffle=False  # shuffle=False para manter alinhamento com máscara
        )
        retain_train_loader = torch.utils.data.DataLoader(
            retain_dataset, batch_size=args.batch_size, shuffle=True
        )
        
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        model.train()
      
        start = time.time()
        loader_len = len(forget_train_loader) + len(retain_train_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, 1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        # ============================================================
        # FASE 1: Forget set com tratamento híbrido
        # ============================================================
        
        # Obter máscara de malignos
        if hasattr(forget_dataset, '_malignant_mask'):
            malignant_mask_full = forget_dataset._malignant_mask
        else:
            malignant_mask_full = None
        
        sample_idx = 0
        for i, (image, target) in enumerate(forget_train_loader):
            batch_size = image.size(0)
            image = image.cuda()
            target = target.cuda()
            
            output_clean = model(image)
            
            if malignant_mask_full is not None:
                # Obter máscara para este batch
                batch_mask = malignant_mask_full[sample_idx:sample_idx + batch_size]
                batch_mask = torch.tensor(batch_mask, dtype=torch.bool, device=device)
                
                # Separar malignos e benignos no batch
                n_malig_batch = batch_mask.sum().item()
                n_benig_batch = (~batch_mask).sum().item()
                
                loss = 0.0
                
                # Loss de entropia para malignos
                if n_malig_batch > 0:
                    malig_logits = output_clean[batch_mask]
                    entropy_loss = _max_entropy_loss(malig_logits)
                    loss = loss + entropy_loss * (n_malig_batch / batch_size)
                
                # Loss CE com random labels para benignos
                if n_benig_batch > 0:
                    benig_logits = output_clean[~batch_mask]
                    benig_targets = target[~batch_mask]
                    if cw is not None:
                        ce_loss = F.cross_entropy(benig_logits, benig_targets, weight=cw)
                    else:
                        ce_loss = F.cross_entropy(benig_logits, benig_targets)
                    loss = loss + ce_loss * (n_benig_batch / batch_size)
                
                sample_idx += batch_size
            else:
                # Fallback: comportamento original
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
            
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Forget Loss {loss:.4f}\t'
                      'Time {time:.2f}'.format(
                          epoch, i, len(forget_train_loader),
                          loss=loss.item() if torch.is_tensor(loss) else loss,
                          time=end - start))
                start = time.time()
        
        # ============================================================
        # FASE 2: Retain set com CE normal
        # ============================================================
        
        for i, (image, target) in enumerate(retain_train_loader):
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
                          epoch, i + len(forget_train_loader), loader_len,
                          end - start, loss=losses, top1=top1))
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        # Mantém comportamento original para datasets não-médicos
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, 1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
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
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

    return top1.avg