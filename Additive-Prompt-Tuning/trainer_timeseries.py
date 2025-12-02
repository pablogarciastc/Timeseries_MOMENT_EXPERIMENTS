"""
Modified Trainer for Time Series Continual Learning
Adapted from APT's trainer_timeseries.py for DailySport dataset
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
from collections import OrderedDict
from torch.utils.data import DataLoader

# Import your DailySport dataloader
from dataloaders.dailysport import iDailySport, get_dailysport_dataloader

# Import learners
import learners

# Import utils
from utils.calc_forgetting import calc_coda_forgetting, calc_general_forgetting


class TimeSeriesTrainer:
    """
    Trainer for time series continual learning with MOMENT+APT
    """

    def __init__(self, args, seed, cur_iter, metric_keys, save_keys):

        # Process inputs
        self.seed = seed
        self.cur_iter = cur_iter
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers

        # Model load directory
        self.model_top_dir = args.log_dir

        # DailySport dataset configuration
        if args.dataset == 'DailySport':
            num_classes = 19
            self.dataset_size = [125, 45]  # [seq_len, n_channels]
            # Adjust based on your data format
        else:
            raise ValueError(f'Dataset {args.dataset} not implemented for time series!')

        self.top_k = 1

        # Upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # Load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()

        if self.seed >= 0 and args.rand_split:
            print('=' * 45)
            print('Shuffling.... seed is', self.seed)
            print('pre-shuffle:', str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:', str(class_order))
            print('=' * 45)

        # Create task splits
        self.tasks = []
        self.tasks_logits = []
        p = 0

        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p + inc])
            self.tasks_logits.append(class_order_logits[p:p + inc])
            p += inc

        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]

        # Number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # Create datasets
        # No transforms for time series by default
        # You can add time series augmentation if needed
        from dataloaders.dailysport import TimeSeriesTransform

        train_transform = TimeSeriesTransform() if args.train_aug else None
        test_transform = None

        self.train_dataset = iDailySport(
            args.dataroot,
            train=True,
            tasks=self.tasks,
            download_flag=True,
            transform=train_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation
        )

        self.test_dataset = iDailySport(
            args.dataroot,
            train=False,
            tasks=self.tasks,
            download_flag=False,
            transform=test_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation
        )

        # Oracle flag
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare learner (model)
        self.learner_config = {
            'num_classes': num_classes,
            'lr': args.lr,
            'debug_mode': args.debug_mode == 1,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'schedule': [args.schedule],
            'schedule_type': args.schedule_type,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'gpuid': args.gpuid,
            'memory': args.memory,
            'temp': args.temp,
            'out_dim': num_classes,
            'overwrite': args.overwrite == 1,
            'DW': args.DW,
            'batch_size': args.batch_size,
            'upper_bound_flag': args.upper_bound_flag,
            'tasks': self.tasks_logits,
            'top_k': self.top_k,
            'prompt_param': [self.num_tasks, args.prompt_param],
            'ema_coeff': args.ema_coeff
        }

        self.learner_type = args.learner_type
        self.learner_name = args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](
            self.learner_config
        )

    def task_eval(self, t_index, local=False, task='acc'):
        """
        Evaluate on a specific task
        """
        val_name = self.task_names[t_index]
        print('Validation split name:', val_name, f"local = {local}")

        # Load test data for this task
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,  # Don't drop last for evaluation
            num_workers=self.workers
        )

        if local:
            return self.learner.validation(
                test_loader,
                task_in=self.tasks_logits[t_index],
                task_metric=task
            )
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
        """
        Train on all tasks sequentially
        """
        # Temporary results saving
        temp_table = {}
        for mkey in self.metric_keys:
            temp_table[mkey] = []

        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # For each task
        for i in range(self.max_task):

            # Save current task index
            self.current_t_index = i

            # Print task name
            train_name = self.task_names[i]
            print('=' * 22, train_name, '=' * 23)

            # Load dataset for task
            task = self.tasks_logits[i]

            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[
                    self.learner_name
                ](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # Set task ID for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # Add valid output dimension to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # Load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # Create dataloader
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=int(self.workers)
            )

            # Increment task ID in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # Learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers
            )

            model_save_dir = (self.model_top_dir + '/models/repeat-' +
                              str(self.cur_iter + 1) + '/task-' +
                              self.task_names[i] + '/')

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            avg_train_time = self.learner.learn_batch(
                train_loader,
                self.train_dataset,
                model_save_dir
            )

            # Save model
            self.learner.save_model(model_save_dir)

            # Evaluate accuracy on all tasks seen so far
            acc_table = []
            self.reset_cluster_labels = True

            for j in range(i + 1):
                acc_table.append(self.task_eval(j))

            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # Save temporary accuracy results
            save_file = temp_dir + 'acc.csv'
            np.savetxt(
                save_file,
                np.asarray(temp_table['acc']),
                delimiter=",",
                fmt='%.2f'
            )

            if avg_train_time is not None:
                avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table):
        """
        Summarize accuracy across all tasks
        """
        # Unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']

        if self.max_task > 1:
            forgetting_table = np.zeros((1, self.max_task, self.max_task))

        # Calculate average performance across tasks
        avg_acc_history = [0] * self.max_task

        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0

            for j in range(i + 1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j, i, self.cur_iter] = acc_table[val_name][train_name]

                if self.max_task > 1:
                    forgetting_table[0][i][j] = acc_table[val_name][train_name]

            avg_acc_history[i] = cls_acc_sum / (i + 1)

        avg_acc_all[:, self.cur_iter] = avg_acc_history

        # Calculate forgetting metrics
        if self.max_task > 1:
            coda_forgetting = calc_coda_forgetting(forgetting_table)
            general_forgetting = calc_general_forgetting(forgetting_table)
            print("coda_forgetting =", coda_forgetting)
            print("general_forgetting =", general_forgetting)

        # Calculate drop matrix
        drop_array = []
        print("acctable:", acc_table)

        for i in range(self.max_task):
            train_name = self.task_names[i]
            drop_i = []
            for j in range(i + 1, self.max_task + 1):
                val_name = self.task_names[j - 1]
                drop_i.append(acc_table[train_name][val_name])
            drop_array.append(drop_i)

        print("dropmatrix:", drop_array)

        return {'global': avg_acc_all, 'pt': avg_acc_pt}

    def evaluate(self, avg_metrics):
        """
        Evaluate saved models on all tasks
        """
        self.learner = learners.__dict__[self.learner_type].__dict__[
            self.learner_name
        ](self.learner_config)

        # Store results
        metric_table = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}

        for i in range(self.max_task):

            # Increment task ID in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # Load model
            model_save_dir = (self.model_top_dir + '/models/repeat-' +
                              str(self.cur_iter + 1) + '/task-' +
                              self.task_names[i] + '/')

            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # Set task ID for model
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # Evaluate accuracy
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True

            print("=== Global validation ===")
            for j in range(i + 1):
                val_name = self.task_names[j]
                print(f"Test task {val_name}, using model {self.task_names[i]}")
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)

        # Summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'])

        return avg_metrics