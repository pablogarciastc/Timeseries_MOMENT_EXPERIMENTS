"""
Advanced Debugging Script for CL-LoRA Catastrophic Forgetting
This script systematically isolates potential causes of forgetting
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import copy
import pickle

class CLLoRADebugger:
    def __init__(self, config_file):
        """Initialize debugger with config"""
        with open(config_file, 'r') as f:
            self.args = json.load(f)

        # Convert device strings to torch devices
        if isinstance(self.args["device"], list) and isinstance(self.args["device"][0], str):
            self.args["device"] = [torch.device(f"cuda:{d}" if d.isdigit() else d)
                                  for d in self.args["device"]]

        self.device = self.args["device"][0]
        self.debug_results = {}

    def setup_model_and_data(self):
        """Initialize model and data manager"""
        from utils.data_manager import DataManager
        from utils import factory

        self.data_manager = DataManager(
            self.args["dataset"],
            self.args["shuffle"],
            self.args["seed"],
            self.args["init_cls"],
            self.args["increment"],
            self.args
        )

        self.model = factory.get_model(self.args["model_name"], self.args)
        return self.model, self.data_manager

    def test_1_adapter_isolation(self):
        """Test 1: Check if adapters are properly isolated between tasks"""
        print("\n" + "="*80)
        print("TEST 1: ADAPTER ISOLATION BETWEEN TASKS")
        print("="*80)

        model, data_manager = self.setup_model_and_data()
        backbone = model._network.backbone

        # Train on Task 0
        print("\n[Task 0] Training...")
        data_manager.get_dataset(np.arange(0, self.args["init_cls"]), source="train", mode="train")
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        # Save adapter weights after Task 0
        task0_adapters = {}
        if hasattr(backbone, 'cur_adapter'):
            for i, adapter_list in enumerate(backbone.cur_adapter):
                task0_adapters[f'block_{i}'] = []
                for j, adapter in enumerate(adapter_list):
                    if hasattr(adapter, 'lora_A'):
                        task0_adapters[f'block_{i}'].append({
                            'lora_A': adapter.lora_A.weight.data.clone(),
                            'lora_B': adapter.lora_B.weight.data.clone()
                        })

        # Move to Task 1
        print("\n[Task 1] After task transition...")
        model.after_task()
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        # Check if old adapters are frozen
        frozen_count = 0
        total_count = 0
        if hasattr(backbone, 'old_adapter_list') and len(backbone.old_adapter_list) > 0:
            for adapter_set in backbone.old_adapter_list:
                for adapter_list in adapter_set:
                    for adapter in adapter_list:
                        if hasattr(adapter, 'lora_A'):
                            total_count += 1
                            if not adapter.lora_A.weight.requires_grad:
                                frozen_count += 1

        print(f"✅ Frozen adapters: {frozen_count}/{total_count}")

        # Check if new adapters are different
        if hasattr(backbone, 'cur_adapter'):
            different = True
            for i, adapter_list in enumerate(backbone.cur_adapter[:len(task0_adapters)]):
                for j, adapter in enumerate(adapter_list):
                    if hasattr(adapter, 'lora_A'):
                        if f'block_{i}' in task0_adapters and j < len(task0_adapters[f'block_{i}']):
                            old = task0_adapters[f'block_{i}'][j]
                            if torch.allclose(adapter.lora_A.weight.data, old['lora_A']):
                                different = False
                                print(f"⚠️ WARNING: New adapter at block {i}, MSA {j} is same as old!")

        if different:
            print("✅ New adapters are properly initialized")

        self.debug_results['adapter_isolation'] = {
            'frozen_ratio': f"{frozen_count}/{total_count}",
            'properly_isolated': different
        }

    def test_2_feature_stability(self):
        """Test 2: Check feature stability across tasks"""
        print("\n" + "="*80)
        print("TEST 2: FEATURE STABILITY ANALYSIS")
        print("="*80)

        model, data_manager = self.setup_model_and_data()

        # Get Task 0 data
        train_dataset = data_manager.get_dataset(
            np.arange(0, self.args["init_cls"]),
            source="train", mode="test"
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        # Extract features before training
        model._network.eval()
        features_before = []
        labels_all = []

        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                # Get features with all adapters
                features = model._network.backbone(inputs, test=True)
                features_before.append(features.cpu())
                labels_all.append(targets)
                if len(features_before) >= 5:  # Sample subset
                    break

        features_before = torch.cat(features_before, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        # Train on Task 0
        print("\nTraining on Task 0...")
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        # Simulate minimal training
        model._network.train()
        optimizer = torch.optim.AdamW(
            [p for p in model._network.parameters() if p.requires_grad],
            lr=1e-4
        )

        for epoch in range(2):
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).long()

                output = model._network(inputs, test=False)
                logits = output["logits"] if isinstance(output, dict) else output
                logits = logits.float()
                loss = nn.CrossEntropyLoss()(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

        # Extract features after Task 0
        model._network.eval()
        features_task0 = []

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                if i >= 5:
                    break
                inputs = inputs.to(self.device).float()
                features = model._network.backbone(inputs, test=True)
                features_task0.append(features.cpu())

        features_task0 = torch.cat(features_task0, dim=0)

        # Move to Task 1 and train
        print("\nMoving to Task 1...")
        model.after_task()
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        # Get Task 1 data
        train_dataset_t1 = data_manager.get_dataset(
            np.arange(self.args["init_cls"], self.args["init_cls"] + self.args["increment"]),
            source="train", mode="train"
        )
        train_loader_t1 = DataLoader(train_dataset_t1, batch_size=16, shuffle=True)

        # Train on Task 1
        model._network.train()
        optimizer = torch.optim.AdamW(
            [p for p in model._network.parameters() if p.requires_grad],
            lr=1e-4
        )

        for epoch in range(2):
            for _, inputs, targets in train_loader_t1:
                inputs = inputs.to(self.device).float()
                targets = (targets.to(self.device) - self.args["init_cls"]).long()

                output = model._network(inputs, test=False)
                logits = output["logits"] if isinstance(output, dict) else output
                logits = logits.float()
                loss = nn.CrossEntropyLoss()(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

        # Extract features after Task 1 (on Task 0 data)
        model._network.eval()
        features_task1 = []

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                if i >= 5:
                    break
                inputs = inputs.to(self.device)
                features = model._network.backbone(inputs, test=True)
                features_task1.append(features.cpu())

        features_task1 = torch.cat(features_task1, dim=0)

        min_dim_0_1 = min(features_task0.size(1), features_before.size(1))
        min_dim_1_0 = min(features_task1.size(1), features_task0.size(1))

        drift_0_to_1 = torch.norm(
            features_task0[:, :min_dim_0_1] - features_before[:, :min_dim_0_1],
            dim=1
        ).mean().item()

        drift_1_to_0 = torch.norm(
            features_task1[:, :min_dim_1_0] - features_task0[:, :min_dim_1_0],
            dim=1
        ).mean().item()

        print(f"\nFeature Drift Analysis:")
        print(f"  Before → Task 0: {drift_0_to_1:.4f}")
        print(f"  Task 0 → Task 1: {drift_1_to_0:.4f}")

        if drift_1_to_0 > drift_0_to_1 * 2:
            print("⚠️ WARNING: Significant feature drift detected after Task 1!")
        else:
            print("✅ Feature drift within acceptable range")

        self.debug_results['feature_stability'] = {
            'initial_drift': drift_0_to_1,
            'cross_task_drift': drift_1_to_0,
            'excessive_drift': drift_1_to_0 > drift_0_to_1 * 2
        }

    def test_3_adapter_mapping(self):
        """Test 3: Verify adapter mapping consistency"""
        print("\n" + "="*80)
        print("TEST 3: ADAPTER MAPPING VERIFICATION")
        print("="*80)

        model, data_manager = self.setup_model_and_data()
        backbone = model._network.backbone

        # Setup for two tasks
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        # After Task 0
        model.after_task()
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        # Get sample input
        train_dataset = data_manager.get_dataset(
            np.arange(0, self.args["init_cls"]),
            source="train", mode="test"
        )
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        _, inputs, _ = next(iter(train_loader))
        inputs = inputs.to(self.device)

        # Test adapter activation in train mode
        model._network.train()
        print("\nTrain mode (test=False):")
        with torch.no_grad():
            _ = model._network.backbone(inputs, test=False)

        if hasattr(backbone, '_active_adapter_mapping'):
            print(f"  Active mapping: {list(backbone._active_adapter_mapping.keys())}")
            for block_idx, adapters in backbone._active_adapter_mapping.items():
                if adapters is not None:
                    print(f"    Block {block_idx}: {len(adapters)} adapters")

        # Test adapter activation in test mode
        model._network.eval()
        print("\nTest mode (test=True):")
        with torch.no_grad():
            _ = model._network.backbone(inputs, test=True)

        if hasattr(backbone, '_active_adapter_mapping'):
            print(f"  Active mapping: {list(backbone._active_adapter_mapping.keys())}")
            for block_idx, adapters in backbone._active_adapter_mapping.items():
                if adapters is not None:
                    print(f"    Block {block_idx}: {len(adapters)} adapters")

        # Verify adapter list lengths
        print("\nAdapter list structure:")
        if hasattr(backbone, 'old_adapter_list'):
            print(f"  Old adapter lists: {len(backbone.old_adapter_list)}")
            for i, adapter_set in enumerate(backbone.old_adapter_list):
                print(f"    Task {i}: {len(adapter_set)} blocks")

        if hasattr(backbone, 'cur_adapter'):
            print(f"  Current adapters: {len(backbone.cur_adapter)} blocks")

        self.debug_results['adapter_mapping'] = {
            'old_tasks': len(backbone.old_adapter_list) if hasattr(backbone, 'old_adapter_list') else 0,
            'current_blocks': len(backbone.cur_adapter) if hasattr(backbone, 'cur_adapter') else 0
        }

    def test_4_gradient_flow(self):
        """Test 4: Check gradient flow through adapters"""
        print("\n" + "="*80)
        print("TEST 4: GRADIENT FLOW ANALYSIS")
        print("="*80)

        model, data_manager = self.setup_model_and_data()
        backbone = model._network.backbone

        # Setup training
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        train_dataset = data_manager.get_dataset(
            np.arange(0, self.args["init_cls"]),
            source="train", mode="train"
        )
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        model._network.train()
        optimizer = torch.optim.AdamW(
            [p for p in model._network.parameters() if p.requires_grad],
            lr=1e-3
        )

        # Track gradients
        gradient_norms = {'task_0': {}, 'task_1': {}}

        # Task 0 gradients
        print("\nTask 0 gradient flow:")
        for batch_idx, (_, inputs, targets) in enumerate(train_loader):
            if batch_idx >= 2:
                break

            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).long()

            output = model._network(inputs, test=False)
            logits = output["logits"] if isinstance(output, dict) else output
            logits = logits.float()
            loss = nn.CrossEntropyLoss()(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            # Check adapter gradients
            if hasattr(backbone, 'cur_adapter'):
                for i, adapter_list in enumerate(backbone.cur_adapter):
                    for j, adapter in enumerate(adapter_list):
                        if hasattr(adapter, 'lora_A') and adapter.lora_A.weight.grad is not None:
                            grad_norm = adapter.lora_A.weight.grad.norm().item()
                            key = f'block_{i}_msa_{j}'
                            if key not in gradient_norms['task_0']:
                                gradient_norms['task_0'][key] = []
                            gradient_norms['task_0'][key].append(grad_norm)

            optimizer.step()

        # Report Task 0 gradients
        for key, norms in gradient_norms['task_0'].items():
            mean_norm = np.mean(norms)
            print(f"  {key}: {mean_norm:.6f}")
            if mean_norm < 1e-6:
                print(f"    ⚠️ WARNING: Very small gradients!")

        # Move to Task 1
        model.after_task()
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        train_dataset_t1 = data_manager.get_dataset(
            np.arange(self.args["init_cls"], self.args["init_cls"] + self.args["increment"]),
            source="train", mode="train"
        )
        train_loader_t1 = DataLoader(train_dataset_t1, batch_size=4, shuffle=True)

        model._network.train()
        optimizer = torch.optim.AdamW(
            [p for p in model._network.parameters() if p.requires_grad],
            lr=1e-3
        )

        # Task 1 gradients
        print("\nTask 1 gradient flow:")
        for batch_idx, (_, inputs, targets) in enumerate(train_loader_t1):
            if batch_idx >= 2:
                break

            inputs = inputs.to(self.device).float()
            targets = (targets.to(self.device) - self.args["init_cls"]).long()

            output = model._network(inputs, test=False)
            logits = output["logits"] if isinstance(output, dict) else output
            logits = logits.float()
            loss = nn.CrossEntropyLoss()(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            # Check new adapter gradients
            if hasattr(backbone, 'cur_adapter'):
                for i, adapter_list in enumerate(backbone.cur_adapter):
                    for j, adapter in enumerate(adapter_list):
                        if hasattr(adapter, 'lora_A') and adapter.lora_A.weight.grad is not None:
                            grad_norm = adapter.lora_A.weight.grad.norm().item()
                            key = f'block_{i}_msa_{j}'
                            if key not in gradient_norms['task_1']:
                                gradient_norms['task_1'][key] = []
                            gradient_norms['task_1'][key].append(grad_norm)

            # Check if old adapters get gradients (they shouldn't!)
            if hasattr(backbone, 'old_adapter_list') and len(backbone.old_adapter_list) > 0:
                for task_idx, adapter_set in enumerate(backbone.old_adapter_list):
                    for block_idx, adapter_list in enumerate(adapter_set):
                        for msa_idx, adapter in enumerate(adapter_list):
                            if hasattr(adapter, 'lora_A') and adapter.lora_A.weight.grad is not None:
                                print(f"    ⚠️ WARNING: Old adapter (task={task_idx}, block={block_idx}, msa={msa_idx}) has gradients!")

            optimizer.step()

        # Report Task 1 gradients
        for key, norms in gradient_norms['task_1'].items():
            mean_norm = np.mean(norms)
            print(f"  {key}: {mean_norm:.6f}")

        self.debug_results['gradient_flow'] = gradient_norms

    def test_5_knowledge_distillation(self):
        """Test 5: Verify knowledge distillation is working"""
        print("\n" + "="*80)
        print("TEST 5: KNOWLEDGE DISTILLATION VERIFICATION")
        print("="*80)

        model, data_manager = self.setup_model_and_data()

        # Train Task 0
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        # Move to Task 1
        model.after_task()
        model._cur_task = 1
        model._network._cur_task = 1
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        # Get Task 1 data
        train_dataset = data_manager.get_dataset(
            np.arange(self.args["init_cls"], self.args["init_cls"] + self.args["increment"]),
            source="train", mode="train"
        )
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        _, inputs, _ = next(iter(train_loader))
        inputs = inputs.to(self.device)

        # Test KD forward
        model._network.train()
        if hasattr(model._network, 'forward_kd'):
            try:
                out_new, out_teacher = model._network.forward_kd(inputs, model._cur_task)

                new_logits = out_new["logits"] if isinstance(out_new, dict) else out_new
                teacher_logits = out_teacher["logits"] if isinstance(out_teacher, dict) else out_teacher

                print(f"New model logits shape: {new_logits.shape}")
                print(f"Teacher logits shape: {teacher_logits.shape}")

                # Check if outputs are different
                diff = torch.abs(new_logits - teacher_logits).mean().item()
                print(f"Mean difference: {diff:.6f}")

                if diff < 1e-6:
                    print("⚠️ WARNING: Teacher and student outputs are identical!")
                else:
                    print("✅ Knowledge distillation produces different outputs")

                self.debug_results['kd_working'] = diff > 1e-6

            except Exception as e:
                print(f"⚠️ ERROR in forward_kd: {e}")
                self.debug_results['kd_working'] = False
        else:
            print("⚠️ forward_kd method not found")
            self.debug_results['kd_working'] = False

    def test_6_classifier_weights(self):
        """Test 6: Analyze classifier weight organization"""
        print("\n" + "="*80)
        print("TEST 6: CLASSIFIER WEIGHT ANALYSIS")
        print("="*80)

        model, data_manager = self.setup_model_and_data()

        # Train Task 0
        model._known_classes = 0
        model._total_classes = self.args["init_cls"]
        model._network.update_fc(self.args["init_cls"])

        # Get classifier weights after Task 0
        fc_weights_t0 = model._network.fc.weight.data.clone()
        print(f"Task 0 classifier shape: {fc_weights_t0.shape}")

        # Move to Task 1
        model.after_task()
        model._network.update_fc(self.args["init_cls"] + self.args["increment"])

        fc_weights_t1 = model._network.fc.weight.data.clone()
        print(f"Task 1 classifier shape: {fc_weights_t1.shape}")

        # Check weight structure
        if hasattr(model._network, 'out_dim'):
            out_dim = model._network.out_dim
            print(f"Output dimension per adapter: {out_dim}")

            # Analyze weight blocks
            if model.args.get('use_diagonal', False):
                print("\nDiagonal mode detected - analyzing weight blocks:")
                for task_id in range(2):  # Check first 2 tasks
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.args["init_cls"]
                    else:
                        start_cls = self.args["init_cls"]
                        end_cls = start_cls + self.args["increment"]

                    if end_cls <= fc_weights_t1.shape[0]:
                        for adapter_idx in range(min(3, fc_weights_t1.shape[1] // out_dim)):
                            weight_block = fc_weights_t1[start_cls:end_cls,
                                                        adapter_idx * out_dim:(adapter_idx + 1) * out_dim]
                            norm = weight_block.norm().item()
                            print(f"  Task {task_id}, Adapter {adapter_idx}: norm={norm:.4f}")

                            if norm < 1e-6:
                                print(f"    ⚠️ WARNING: Zero weights detected!")

        self.debug_results['classifier_structure'] = {
            'task0_shape': list(fc_weights_t0.shape),
            'task1_shape': list(fc_weights_t1.shape)
        }

    def test_7_forward_path_comparison(self):
        """Test 7: Compare forward paths offline vs online"""
        print("\n" + "="*80)
        print("TEST 7: FORWARD PATH COMPARISON (OFFLINE VS ONLINE)")
        print("="*80)

        # Load saved features if available
        saved_features_dir = "saved_features"
        if not os.path.exists(saved_features_dir):
            print("⚠️ No saved features found. Run training first to generate them.")
            return

        # Load Task 0 features
        task0_file = os.path.join(saved_features_dir, "task_0.pkl")
        if os.path.exists(task0_file):
            with open(task0_file, 'rb') as f:
                task0_data = pickle.load(f)

            print(f"Loaded Task 0 offline features: {task0_data['train_features'].shape}")

            # Setup model for online comparison
            model, data_manager = self.setup_model_and_data()

            # Configure for Task 1 (after Task 0 is complete)
            model._known_classes = self.args["init_cls"]
            model._total_classes = self.args["init_cls"] + self.args["increment"]
            model._cur_task = 1
            model._network._cur_task = 1
            model._network.update_fc(model._total_classes)


            # Get Task 0 test data for online feature extraction
            test_dataset = data_manager.get_dataset(
                np.arange(0, self.args["init_cls"]),
                source="test", mode="test"
            )
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # Extract online features
            model._network.eval()
            online_features = []
            online_labels = []

            print("\nExtracting online features for Task 0 data...")
            with torch.no_grad():
                for _, inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    features = model._network.backbone.forward_proto(inputs,
                                                                     adapt_index=0)  # Task 0 only
                    online_features.append(features.cpu())
                    online_labels.append(targets)
                    if len(online_features) >= 5:  # Match offline sample size
                        break

            online_features = torch.cat(online_features, dim=0)
            online_labels = torch.cat(online_labels, dim=0)

            # Compare features
            offline_sample = task0_data['test_features'][:online_features.shape[0]]

            # Compute similarity
            cosine_sim = nn.CosineSimilarity(dim=1)
            similarities = cosine_sim(offline_sample, online_features).mean().item()

            # Compute L2 distance
            l2_dist = torch.norm(offline_sample - online_features, dim=1).mean().item()

            print(f"\nFeature comparison (Task 0 data):")
            print(f"  Cosine similarity: {similarities:.4f}")
            print(f"  L2 distance: {l2_dist:.4f}")

            if similarities < 0.9:
                print("⚠️ WARNING: Low similarity between offline and online features!")
                print("   This indicates features are changing between tasks.")
            else:
                print("✅ Features are consistent between offline and online")

            # Test classifier accuracy with offline features
            print("\nTesting classifier with offline features...")
            with torch.no_grad():
                # Use offline features with current classifier
                offline_logits = model._network.fc(offline_sample.to(self.device))
                if isinstance(offline_logits, dict):
                    offline_logits = offline_logits['logits']

                offline_preds = offline_logits.argmax(dim=1).cpu()
                offline_acc = (offline_preds == online_labels[:offline_preds.shape[0]]).float().mean()

                # Compare with online forward pass
                online_logits = model._network.fc(online_features.to(self.device))
                if isinstance(online_logits, dict):
                    online_logits = online_logits['logits']

                online_preds = online_logits.argmax(dim=1).cpu()
                online_acc = (online_preds == online_labels[:online_preds.shape[0]]).float().mean()

                print(f"  Offline features → Classifier accuracy: {offline_acc:.4f}")
                print(f"  Online features → Classifier accuracy: {online_acc:.4f}")

                if abs(offline_acc - online_acc) > 0.1:
                    print("⚠️ WARNING: Significant accuracy difference!")

            self.debug_results['offline_online_comparison'] = {
                'feature_similarity': similarities,
                'feature_l2_dist': l2_dist,
                'offline_accuracy': offline_acc.item(),
                'online_accuracy': online_acc.item()
            }

    def run_all_tests(self):
        """Run all debugging tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CL-LORA DEBUGGING SUITE")
        print("="*80)

        # Run each test
        self.test_1_adapter_isolation()
        self.test_2_feature_stability()
        self.test_3_adapter_mapping()
        self.test_4_gradient_flow()
        #self.test_5_knowledge_distillation()
        self.test_6_classifier_weights()
        self.test_7_forward_path_comparison()

        # Generate summary report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive debugging report"""
        print("\n" + "="*80)
        print("DEBUGGING SUMMARY REPORT")
        print("="*80)

        issues_found = []

        # Check each test result
        if 'adapter_isolation' in self.debug_results:
            if not self.debug_results['adapter_isolation']['properly_isolated']:
                issues_found.append("Adapters not properly isolated between tasks")

        if 'feature_stability' in self.debug_results:
            if self.debug_results['feature_stability']['excessive_drift']:
                issues_found.append("Excessive feature drift detected")

        if 'kd_working' in self.debug_results:
            if not self.debug_results['kd_working']:
                issues_found.append("Knowledge distillation not functioning")

        if 'offline_online_comparison' in self.debug_results:
            comp = self.debug_results['offline_online_comparison']
            if comp['feature_similarity'] < 0.9:
                issues_found.append("Feature inconsistency between offline/online")
            if abs(comp['offline_accuracy'] - comp['online_accuracy']) > 0.1:
                issues_found.append("Significant accuracy gap offline vs online")

        # Report findings
        if issues_found:
            print("\n🔴 CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues_found, 1):
                print(f"  {i}. {issue}")

            print("\n📝 RECOMMENDATIONS:")
            print("  1. Check adapter freezing mechanism in after_task()")
            print("  2. Verify forward_kd() implementation matches original")
            print("  3. Ensure old_adapter_list is properly maintained")
            print("  4. Check if test=True flag correctly activates all adapters")
            print("  5. Verify gradient manipulation in training loop")

        else:
            print("\n✅ No critical issues detected")
            print("   Catastrophic forgetting may be due to:")
            print("   - Hyperparameter tuning needed")
            print("   - Insufficient adapter capacity")
            print("   - Learning rate scheduling issues")

        # Save results to file
        report_file = "cllora_debug_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.debug_results, f, indent=2, default=str)
        print(f"\n📊 Detailed results saved to: {report_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python isolate_forgetting_debug.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    debugger = CLLoRADebugger(config_file)
    debugger.run_all_tests()