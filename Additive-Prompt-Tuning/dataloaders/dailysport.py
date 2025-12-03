import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class iDailySport(Dataset):

    def __init__(self, root, train=True, tasks=None, download_flag=False,
                 transform=None, seed=0, rand_split=False, validation=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.validation = validation
        self.transform = transform
        self.seed = seed
        self.num_classes = 19
        self.tasks = tasks
        self.t = -1
        self.coreset = []
        self._load_data()
        print(f"DailySport loaded: {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels: {np.unique(self.labels)}")

    def _load_data(self):
        if self.train:
            data_file = os.path.join(self.root, 'x_train.pkl')
            label_file = os.path.join(self.root, 'state_train.pkl')
        else:
            data_file = os.path.join(self.root, 'x_test.pkl')
            label_file = os.path.join(self.root, 'state_test.pkl')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)

        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)

        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        if len(self.labels.shape) > 1:
            self.labels = self.labels.flatten()

        unique_labels = np.unique(self.labels)
        if unique_labels.min() == 1:
            print("Converting labels from 1-indexed to 0-indexed")
            self.labels = self.labels - 1

        print(f"Loaded {self.data.shape[0]} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Label range: {self.labels.min()} to {self.labels.max()}")

    def load_dataset(self, t, train=True):
        self.t = t
        self.train = train
        task_classes = self.tasks[t]
        mask = np.isin(self.labels, task_classes)
        self.task_data = self.data[mask]
        self.task_labels = self.labels[mask]
        print(f"Task {t}: Loaded {len(self.task_data)} samples for classes {task_classes}")

    def append_coreset(self, only=False):
        if len(self.coreset) == 0:
            return

        coreset_data = []
        coreset_labels = []

        for data, label in self.coreset:
            coreset_data.append(data)
            coreset_labels.append(label)

        coreset_data = np.array(coreset_data)
        coreset_labels = np.array(coreset_labels)

        if only:
            self.task_data = coreset_data
            self.task_labels = coreset_labels
        else:
            self.task_data = np.concatenate([self.task_data, coreset_data], axis=0)
            self.task_labels = np.concatenate([self.task_labels, coreset_labels], axis=0)

        print(f"Coreset: {len(coreset_data)} samples added")

    def update_coreset(self, coreset_size, seen_classes):
        if coreset_size == 0:
            return

        num_classes_seen = len(seen_classes)
        samples_per_class = coreset_size // num_classes_seen

        self.coreset = []

        for class_idx in seen_classes:
            class_mask = self.labels == class_idx
            class_data = self.data[class_mask]
            class_labels = self.labels[class_mask]

            if len(class_data) > samples_per_class:
                indices = np.random.choice(len(class_data), samples_per_class, replace=False)
                class_data = class_data[indices]
                class_labels = class_labels[indices]

            for data, label in zip(class_data, class_labels):
                self.coreset.append((data, label))

        print(f"Coreset updated: {len(self.coreset)} samples, {samples_per_class} per class")

    def __len__(self):
        return len(self.task_data)

    def __getitem__(self, idx):
        data = self.task_data[idx]
        label = self.task_labels[idx]
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label])[0]

        if self.transform is not None:
            data = self.transform(data)

        return data, label, self.t


class TimeSeriesTransform:
    def __init__(self, jitter_ratio=0.05, scale_ratio=0.1):
        self.jitter_ratio = jitter_ratio
        self.scale_ratio = scale_ratio

    def __call__(self, x):
        if self.jitter_ratio > 0:
            noise = torch.randn_like(x) * self.jitter_ratio * x.std()
            x = x + noise

        if self.scale_ratio > 0:
            scale = 1 + (torch.rand(1) - 0.5) * 2 * self.scale_ratio
            x = x * scale

        return x


def get_dailysport_dataloader(args):
    if args.first_split_size == 10 and args.other_split_size == 3:
        tasks = [
            list(range(0, 10)),
            list(range(10, 13)),
            list(range(13, 16)),
            list(range(16, 19)),
        ]
    elif args.first_split_size == 5 and args.other_split_size == 2:
        tasks = [
            list(range(0, 5)),
            list(range(5, 7)),
            list(range(7, 9)),
            list(range(9, 11)),
            list(range(11, 13)),
            list(range(13, 15)),
            list(range(15, 17)),
            list(range(17, 19)),
        ]
    elif args.first_split_size == 1 and args.other_split_size == 1:
        tasks = [[i] for i in range(19)]
    else:
        raise ValueError(f"Unsupported split: {args.first_split_size}-{args.other_split_size}")

    transform = TimeSeriesTransform() if args.train_aug else None

    train_dataset = iDailySport(
        root=args.dataroot,
        train=True,
        tasks=tasks,
        transform=transform,
        seed=args.seed,
        rand_split=args.rand_split,
        validation=args.validation
    )

    test_dataset = iDailySport(
        root=args.dataroot,
        train=False,
        tasks=tasks,
        transform=None,
        seed=args.seed,
        rand_split=args.rand_split,
        validation=args.validation
    )

    return train_dataset, test_dataset


class iDailyAndSports(iDailySport):
    pass


def test_dataloader():
    import argparse

    args = argparse.Namespace()
    args.dataroot = './data/dailysport'
    args.first_split_size = 10
    args.other_split_size = 3
    args.train_aug = False
    args.seed = 42
    args.rand_split = False
    args.validation = False

    print("Testing DailySport dataloader...")

    try:
        train_dataset, test_dataset = get_dailysport_dataloader(args)
        train_dataset.load_dataset(0, train=True)
        test_dataset.load_dataset(0, train=False)
        data, label, task = train_dataset[0]
        print(f"\nSample data shape: {data.shape}")
        print(f"Sample label: {label}")
        print(f"Task ID: {task}")

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for batch_data, batch_labels, batch_tasks in train_loader:
            print(f"\nBatch data shape: {batch_data.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            print(f"Batch labels: {batch_labels[:10]}")
            break

        print("\n✓ Dataloader test successful!")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False