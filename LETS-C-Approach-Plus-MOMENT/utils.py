"""
Utilidades y funciones auxiliares para el proyecto
"""

import torch
import numpy as np
import random
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def setup_logger(log_dir='logs', name='training'):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)


def save_checkpoint(state, save_path, is_best=False):
    torch.save(state, save_path)
    if is_best:
        best_path = str(save_path).replace('.pt', '_best.pt')
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.inf
        else:
            self.monitor_op = np.less
            self.best_score = np.inf

    def __call__(self, score):
        if self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def freeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True


def print_model_summary(model):
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"\nTotal parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Frozen parameters:     {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    print("\nParameter breakdown by module:")
    print("-" * 80)

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = module_params - trainable

        print(f"{name:25s} | Total: {module_params:>12,} | "
              f"Trainable: {trainable:>12,} | Frozen: {frozen:>12,}")

    print("="*80 + "\n")


def calculate_class_weights(labels, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(labels))

    class_counts = np.bincount(labels, minlength=n_classes)
    class_weights = len(labels) / (n_classes * class_counts)

    return torch.FloatTensor(class_weights)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def time_series_augmentation(x, methods=['jitter', 'scaling', 'rotation']):
    x_aug = x.clone()

    if 'jitter' in methods:
        noise = torch.randn_like(x_aug) * 0.05
        x_aug = x_aug + noise

    if 'scaling' in methods:
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2).to(x.device)
        x_aug = x_aug * scale

    if 'rotation' in methods:
        shift = np.random.randint(0, x_aug.size(2))
        x_aug = torch.roll(x_aug, shifts=shift, dims=2)

    if 'magnitude_warp' in methods:
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(x_aug.size(2))
        random_warps = np.random.normal(loc=1.0, scale=0.2, size=(x_aug.size(1), 4))
        warp_steps = np.linspace(0, x_aug.size(2)-1, 4)

        for i in range(x_aug.size(1)):
            warper = CubicSpline(warp_steps, random_warps[i])
            warp = warper(orig_steps)
            x_aug[:, i, :] = x_aug[:, i, :] * torch.FloatTensor(warp).to(x.device)

    return x_aug


def save_predictions(predictions, labels, save_path='predictions.json'):
    results = {
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'labels': labels.tolist() if isinstance(labels, np.ndarray) else labels,
        'timestamp': datetime.now().isoformat()
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def moving_average(values, window=10):
    if len(values) < window:
        return values

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')


class ProgressTracker:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0
        self.best_epoch = 0

    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

    def save(self, save_path='training_history.json'):
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, load_path='training_history.json'):
        with open(load_path, 'r') as f:
            self.history = json.load(f)


def print_training_config(args):
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)

    for key, value in vars(args).items():
        print(f"{key:25s} : {value}")

    print("="*80 + "\n")
