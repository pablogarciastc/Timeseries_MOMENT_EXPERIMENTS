import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle


def prepare_dataset(x_path, y_path):
    """Load and prepare dataset from pickle files."""
    with open(x_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)

    # Convert to tensors if not already
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data)
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data)

    return x_data, y_data


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.seed = seed

        # Load data from pickle files
        self._setup_data(dataset_name, shuffle, seed)

        # Setup task increments
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

        logging.info(f"Task increments: {self._increments}")

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # No transforms needed for tensor data
        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)

        if ret_data:
            return data, targets, TensorDataset(data, targets)
        else:
            return TensorDataset(data, targets)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        train_data, train_targets = [], []
        val_data, val_targets = [], []

        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )

            # Convert to numpy for indexing if needed
            class_data_np = class_data.numpy() if isinstance(class_data, torch.Tensor) else class_data
            class_targets_np = class_targets.numpy() if isinstance(class_targets, torch.Tensor) else class_targets

            val_indx = np.random.choice(
                len(class_data_np), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data_np))) - set(val_indx))

            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(torch.max(appendent_targets).item()) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )

                append_data_np = append_data.numpy() if isinstance(append_data, torch.Tensor) else append_data

                val_indx = np.random.choice(
                    len(append_data_np), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data_np))) - set(val_indx))

                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data = torch.cat(train_data, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        val_data = torch.cat(val_data, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        return TensorDataset(train_data, train_targets), TensorDataset(val_data, val_targets)

    def _setup_data(self, dataset_name, shuffle, seed):
        """Load data from pickle files."""
        print("Loading data from pickle files...")

        base_path = Path('../data') / dataset_name

        # Load training and test data
        self.x_train, self.y_train = prepare_dataset(
            base_path / 'x_train.pkl',
            base_path / 'state_train.pkl'
        )
        self.x_test, self.y_test = prepare_dataset(
            base_path / 'x_test.pkl',
            base_path / 'state_test.pkl'
        )

        print(f"Train: {self.x_train.shape}, Test: {self.x_test.shape}")

        # Store as internal data
        self._train_data = self.x_train
        self._train_targets = self.y_train
        self._test_data = self.x_test
        self._test_targets = self.y_test

        # No image transforms needed for tensor data
        self.use_path = False

        # Setup class order
        n_classes = len(torch.unique(self.y_train))
        order = [i for i in range(n_classes)]

        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()

        self._class_order = order
        logging.info(f"Class order: {self._class_order}")

        # Map indices to new order
        self._train_targets = self._map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = self._map_new_class_index(
            self._test_targets, self._class_order
        )

        print(f"\nTotal classes: {n_classes}")
        print(f"Train samples: {len(self._train_targets)}, Test samples: {len(self._test_targets)}")

    def _select(self, x, y, low_range, high_range):
        """Select samples within a class range."""
        mask = (y >= low_range) & (y < high_range)
        return x[mask], y[mask]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        """Select samples with random missing modality rate."""
        assert m_rate is not None
        mask = (y >= low_range) & (y < high_range)
        indices = torch.where(mask)[0]

        if m_rate != 0:
            n_select = int((1 - m_rate) * len(indices))
            selected_indices = torch.randperm(len(indices))[:n_select]
            selected_indices = torch.sort(selected_indices)[0]
            new_indices = indices[selected_indices]
        else:
            new_indices = indices

        return x[new_indices], y[new_indices]

    def _map_new_class_index(self, y, order):
        """Map class indices to new order."""
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        mapped = np.array([order.index(int(label)) for label in y_np])
        return torch.tensor(mapped, dtype=y.dtype)

    def getlen(self, index):
        """Get number of samples for a specific class."""
        y = self._train_targets
        return torch.sum(y == index).item()


class TensorDataset(Dataset):
    """Dataset wrapper for tensor data."""

    def __init__(self, data, labels):
        assert len(data) == len(labels), "Data size error!"
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]


class DummyDataset(Dataset):
    """Legacy dataset class for compatibility with image-based workflows."""

    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def pil_loader(path):
    """Load image from path."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")