import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PabloNet

from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import random

num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = PabloNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"]  # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc_proxy(self):
        model = self._network
        model = model.eval()
        model.fc.weight.data[self._known_classes:self._total_classes, :] = model.proxy_fc.weight.data
        model.fc.bias.data[self._known_classes:self._total_classes] = model.proxy_fc.bias.data

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)

                    class_index = int(class_index)
                    index = int(index)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index + 1) * self._network.out_dim:(
                                                                                                          index + 2) * self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[
                            class_index, index * self._network.out_dim:(index + 1) * self._network.out_dim] = proto
        return

    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim

        start_cls, end_cls = self.get_cls_range(task_id)

        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim: end_dim]
        # A = self._network.fc.weight.data[0:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        # B = self._network.fc.weight.data[0:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls: end_cls, start_dim: end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        for task_id in range(self._cur_task):
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)

            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]

            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls: end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def _save_task_features(self, train_loader, test_loader, task_id):
        """Guarda features y labels para entrenamiento offline"""
        import pickle
        import os

        save_dir = "saved_features"
        os.makedirs(save_dir, exist_ok=True)

        self._network.eval()

        # Train features
        train_features, train_labels = [], []
        print(f"[Task {task_id}] Extracting train features...")
        with torch.no_grad():
            for batch in train_loader:
                _, inputs, targets = batch[:3]
                inputs = inputs.to(self._device)

                # IMPORTANTE: usar test=True para obtener TODAS las features
                features = self._network.backbone(inputs, test=True)

                train_features.append(features.cpu())
                train_labels.append(targets.cpu())

        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # Test features
        test_features, test_labels = [], []
        print(f"[Task {task_id}] Extracting test features...")
        with torch.no_grad():
            for batch in test_loader:
                _, inputs, targets = batch[:3]
                inputs = inputs.to(self._device)
                features = self._network.backbone(inputs, test=True)

                test_features.append(features.cpu())
                test_labels.append(targets.cpu())

        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        # Guardar
        save_data = {
            'task_id': task_id,
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'feature_dim': features.shape[1],
            'num_classes': self._total_classes,
        }

        save_path = os.path.join(save_dir, f'task_{task_id}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"✅ Saved: {save_path}")
        print(f"   Train: {train_features.shape}, Test: {test_features.shape}")
        print(f"   Feature dim: {features.shape[1]}\n")

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1)  # dim=1, not dim=0

                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]

                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls: end_cls, start_dim: end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        # self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._network._cur_task + 1  # +1 porque update_fc lo incrementará
        )
        self._network.update_fc(self._total_classes)


        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args["init_lr"],
            weight_decay=self.args["weight_decay"]
        )

        # Sincronizar después
        self._cur_task = self._network._cur_task
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                                   source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._save_task_features(self.train_loader, self.test_loader, self._cur_task)

        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            if hasattr(self._network.backbone, "cur_adapter") and len(self._network.backbone.cur_adapter) > 0:
                if not self._network.backbone.msa_adapt:
                    for name, param in self._network.backbone.cur_adapter[0].named_parameters():
                        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                else:
                    for name, param in self._network.backbone.cur_adapter[0][1].named_parameters():
                        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                    for name, param in self._network.backbone.cur_adapter[-1][1].named_parameters():
                        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
            else:
                print("[DEBUG] No adapters initialized (msa_adapt=False)")

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()

                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                output = self._network(inputs, test=False)
                if isinstance(output, dict):
                    logits = output["logits"] if isinstance(output, dict) else output
                else:
                    logits = output

                aux_targets = aux_targets.long()
                loss = F.cross_entropy(logits, aux_targets)

                # Compute orthogonality loss before any backward passes
                # Detach to prevent conflicts between multiple forward/backward passes
                if self._cur_task > 0 and self._network.backbone.use_block_weight:
                    with torch.no_grad():
                        orth_loss_specific = compute_orthogonality_loss(
                            self._network.backbone.block_weight_list,
                            self._network.backbone.block_weight
                        )

                    if not isinstance(orth_loss_specific, torch.Tensor):
                        orth_loss_specific = torch.tensor(orth_loss_specific, device=self._device, dtype=loss.dtype)

                    loss = loss + 0.0001 * orth_loss_specific.detach()

                if self._cur_task > 0:
                    kd_ratio = 5.
                    Temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)

                    # Add KD loss to total loss instead of separate backward
                    loss = loss + loss_kd

                    # Now do a single backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Apply gradient manipulation after backward
                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])

                        # Skip gradient manipulation if adapters or MSA are disabled
                        if not hasattr(self._network.backbone, "msa") or not getattr(self._network.backbone,
                                                                                     "msa_adapt", False):
                            continue

                        for jj in range(len(self._network.backbone.msa)):

                            if self._network.backbone.msa[jj] == 1:
                                w = self._network.backbone.old_adapter_list[self._cur_task - 1][pos][jj].lora_A.weight

                                temp_weights = 1. * torch.norm(
                                    self._network.backbone.old_adapter_list[self._cur_task - 1][pos][jj].lora_A.weight,
                                    dim=1)
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)

                                # Create a new tensor instead of modifying in-place
                                original_grad = self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                                if original_grad is not None:
                                    weighted_grad = temp_weights.unsqueeze(1) * original_grad
                                    self._network.backbone.cur_adapter[pos][
                                        jj].lora_A.weight.grad = weighted_grad.clone()
                    optimizer.step()
                else:
                    # For first task, just do normal backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    task_id = int(task_id.item()) if torch.is_tensor(task_id) else int(task_id)
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id - 1) * self.inc
                        end_cls = self.init_cls + task_id * self.inc

                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]