import csv
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from collections import Counter
from torch.cuda import amp
from tqdm import tqdm

from dataset import IPMCoeffDualDataset
from model import DualBranchEnsemble

import torch.nn.functional as F

# 使用 GPU
device = torch.device("cuda")
print(f"Using device: {device}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class WarmupCosineScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, warmup_epochs,
                 total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.after_warmup = False
        super().__init__(
            optimizer,
            T_max=total_epochs -
            warmup_epochs,
            eta_min=min_lr,
            last_epoch=-
            1)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch < self.warmup_epochs:
            warmup_lr = [(epoch + 1) / self.warmup_epochs *
                         base_lr for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if not self.after_warmup:
                self.last_epoch = -1
                self.after_warmup = True
            super().step(epoch=epoch - self.warmup_epochs)


class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=0.7):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, inputs, targets):
        return self.weight * \
            self.focal(inputs, targets) + (1 - self.weight) * \
            self.ce(inputs, targets)


class DynamicLossScheduler:
    def __init__(self, switch_epoch=10, alpha=1.0, gamma=2.0, weight=0.7):
        self.switch_epoch = switch_epoch
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss(alpha, gamma)
        self.hybrid = HybridLoss(alpha, gamma, weight)

    def get_loss(self, epoch):
        if epoch < self.switch_epoch:
            return self.ce
        elif self.switch_epoch <= epoch < self.switch_epoch + 10:
            return self.hybrid
        else:
            return self.focal  # 末期强化训练


def train_one_fold(fold_id, train_json, val_json, test_json, data_root,
                   batch_size=4, epochs=50, lr=1e-4,
                   dataset_tag="", use_attention=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'best_model_fold{fold_id}{dataset_tag}_{timestamp}.pth'
    model = DualBranchEnsemble().to(device)
    pretrained_path = None
    model_dir = '.'  # 当前目录保存 .pth 文件
    for file in os.listdir(model_dir):
        if file.startswith(
                f'best_model_fold{fold_id}_') and file.endswith('.pth'):
            pretrained_path = os.path.join(model_dir, file)
            break

    if pretrained_path:
        print(
            f"Fold {fold_id} already trained. Loading model from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
        model.eval()
        # 加载测试集，仅执行测试
        test_dataset = IPMCoeffDualDataset(
            data_root, test_json, use_attention=use_attention)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=0)

        all_labels, all_probs = [], []
        with torch.no_grad():
            for ipm_input, coeff_input, labels in tqdm(test_loader, desc=f"fold {fold_id}/1 [Test]",
                                                       leave=False):
                ipm_input, coeff_input = ipm_input.to(
                    device), coeff_input.to(device)
                outputs = model(ipm_input, coeff_input)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except BaseException:
            auc = -1
        report = classification_report(all_labels, preds, digits=4)
        print(
            f"Fold {fold_id} Test Accuracy: {acc * 100:.2f}%\nFold {fold_id} Test AUC: {auc:.4f}\nClassification Report:\n{report}")
        return acc, auc

    # === 数据集 ===
    print(f"===== Fold {fold_id} =====")
    train_dataset = IPMCoeffDualDataset(
        data_root, train_json, use_attention=use_attention)
    val_dataset = IPMCoeffDualDataset(
        data_root, val_json, use_attention=use_attention)
    test_dataset = IPMCoeffDualDataset(
        data_root, test_json, use_attention=use_attention)
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    #   统计标签分布（看是否平衡）
    train_labels = [label for _, label in train_dataset.samples]
    print(f"Train Labels: {Counter(train_labels)}")

    # === 类别权重 ===
    counter = Counter(train_labels)
    num_cover = counter.get(0, 1)
    num_stego = counter.get(1, 1)
    total = num_cover + num_stego
    class_weights = torch.tensor([
        total / (2 * num_cover + 1e-6),
        total / (2 * num_stego + 1e-6)
    ], dtype=torch.float32).to(device)

    # === 优化器 & Loss ===
    optimizer = optim.Adam(model.parameters(), lr=lr * 1.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = amp.GradScaler(enabled=True)
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None

    # === DataLoader ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * 4,
        shuffle=True,
        num_workers=0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0)

    # === 训练循环 ===
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0.0, 0
        for ipm_input, coeff_input, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]",
                                                   leave=False):
            ipm_input, coeff_input, labels = ipm_input.to(
                device), coeff_input.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                outputs = model(ipm_input, coeff_input)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * ipm_input.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
        avg_train_loss = total_loss / len(train_dataset)
        avg_train_acc = total_correct / len(train_dataset) * 100

        # === 验证 ===
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for ipm_input, coeff_input, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]",
                                                       leave=False):
                ipm_input, coeff_input, labels = ipm_input.to(
                    device), coeff_input.to(device), labels.to(device)
                with amp.autocast(enabled=True):
                    outputs = model(ipm_input, coeff_input)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * ipm_input.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / len(val_dataset) * 100

        print(f"Epoch [{epoch + 1}/{epochs}]  LR: {optimizer.param_groups[0]['lr']:.2e} "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")
        scheduler.step(avg_val_loss)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict()

    torch.save(best_model_state, model_path)
    print(f"Best model saved to {model_path}")

    # === 测试 ===
    model.load_state_dict(best_model_state)
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for ipm_input, coeff_input, labels in tqdm(test_loader, desc=f"fold {fold_id}/1 [Test]",
                                                   leave=False):
            ipm_input, coeff_input = ipm_input.to(
                device), coeff_input.to(device)
            outputs = model(ipm_input, coeff_input)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except BaseException:
        auc = -1
    report = classification_report(all_labels, preds, digits=4)
    print(
        f"Fold {fold_id} Test Accuracy: {acc * 100:.2f}%\nFold {fold_id} Test AUC: {auc:.4f}\nClassification Report:\n{report}")
    return acc, auc


if __name__ == '__main__':
    data_root = 'yourPath/data_0.5_QP'
    split_dir = 'yourPath/splits_0.5_QP'
    dataset_tag = os.path.basename(data_root).split(
        'data')[-1]  # 提取 "_payload_qp"
    results = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'yourPath/fold_results{dataset_tag}_{timestamp}.csv'
    summary_file = f'yourPath/fold_results_summary{dataset_tag}_{timestamp}.csv'

    for fold_id in range(1, 2):
        train_json = os.path.join(
            split_dir, f'label_map_train_split{fold_id}.json')
        val_json = os.path.join(
            split_dir, f'label_map_val_split{fold_id}.json')
        test_json = os.path.join(
            split_dir, f'label_map_test_split{fold_id}.json')
        acc, auc = train_one_fold(fold_id, train_json, val_json, test_json, data_root, dataset_tag=dataset_tag,
                                  use_attention=True)
        results.append({'fold': fold_id, 'accuracy': acc, 'auc': auc})

    with open(result_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'accuracy', 'auc'])
        writer.writeheader()
        writer.writerows(results)

    acc = sum([r['accuracy'] for r in results]) / len(results)
    valid_auc = [r['auc'] for r in results if r['auc'] >= 0]

    print(f"\n=== Summary ===\nAccuracy: {acc * 100:.2f}%")
    print(f"AUC: {valid_auc:.4f}" if valid_auc >= 0 else "AUC: N/A")
