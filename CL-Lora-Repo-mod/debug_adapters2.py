"""
Focused Test: Adapter Activation Mechanism in MOMENT CL-LoRA
This script specifically tests if adapters are being activated correctly
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
from torch.utils.data import DataLoader


def test_adapter_activation(config_file):
    """Test if adapters are activated correctly during forward passes"""

    print("\n" + "=" * 80)
    print("ADAPTER ACTIVATION MECHANISM TEST")
    print("=" * 80)

    # Load config
    with open(config_file, 'r') as f:
        args = json.load(f)

    if isinstance(args["device"], list) and isinstance(args["device"][0], str):
        args["device"] = [torch.device(f"cuda:{d}" if d.isdigit() else d)
                          for d in args["device"]]

    device = args["device"][0]

    # Setup model and data
    from utils.data_manager import DataManager
    from utils import factory

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )

    model = factory.get_model(args["model_name"], args)
    backbone = model._network.backbone

    # Prepare for Task 0
    model._known_classes = 0
    model._total_classes = args["init_cls"]
    model._network.update_fc(args["init_cls"])

    # Get sample data
    train_dataset = data_manager.get_dataset(
        np.arange(0, args["init_cls"]),
        source="train", mode="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    _, inputs, _ = next(iter(train_loader))
    inputs = inputs.to(device)

    print("\n" + "-" * 40)
    print("PHASE 1: SINGLE TASK (Task 0 only)")
    print("-" * 40)

    # Track adapter activations
    activation_tracker = {}

    def hook_fn(name, is_adapter=False):
        def hook(module, input, output):
            if name not in activation_tracker:
                activation_tracker[name] = {
                    'count': 0,
                    'is_active': False,
                    'output_norm': []
                }
            activation_tracker[name]['count'] += 1

            if isinstance(output, tuple):
                output = output[0]

            if torch.is_tensor(output):
                norm = output.norm().item()
                activation_tracker[name]['output_norm'].append(norm)
                activation_tracker[name]['is_active'] = norm > 1e-8

                if is_adapter and norm > 1e-8:
                    print(f"  ✓ {name} activated: norm={norm:.6f}")

        return hook

    # Register hooks for Task 0
    hooks = []

    # Hook current adapters
    if hasattr(backbone, 'cur_adapter'):
        for i, adapter_list in enumerate(backbone.cur_adapter):
            for j, adapter in enumerate(adapter_list):
                if hasattr(adapter, 'lora_A'):
                    name = f'cur_adapter_b{i}_m{j}'
                    h = adapter.register_forward_hook(hook_fn(name, True))
                    hooks.append(h)

    # Test 1: Train mode (test=False)
    print("\n1. Train Mode (test=False, cur_task=0):")
    model._network.train()
    activation_tracker.clear()

    with torch.no_grad():
        _ = model._network(inputs, test=False)

    active_count = sum(1 for v in activation_tracker.values() if v['is_active'])
    print(f"   Active adapters: {active_count}/{len(activation_tracker)}")

    # Test 2: Test mode (test=True)
    print("\n2. Test Mode (test=True, cur_task=0):")
    model._network.eval()
    activation_tracker.clear()

    with torch.no_grad():
        _ = model._network(inputs, test=True)

    active_count = sum(1 for v in activation_tracker.values() if v['is_active'])
    print(f"   Active adapters: {active_count}/{len(activation_tracker)}")

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Move to Task 1
    print("\n" + "-" * 40)
    print("PHASE 2: MULTI-TASK (After Task 0 → Task 1)")
    print("-" * 40)

    model.after_task()
    model._cur_task = 1
    model._network._cur_task = 1
    model._network.update_fc(args["init_cls"] + args["increment"])

    # Re-register hooks for both old and current adapters
    hooks = []
    activation_tracker.clear()

    # Hook old adapters
    if hasattr(backbone, 'old_adapter_list'):
        for task_idx, adapter_set in enumerate(backbone.old_adapter_list):
            for block_idx, adapter_list in enumerate(adapter_set):
                for msa_idx, adapter in enumerate(adapter_list):
                    if hasattr(adapter, 'lora_A'):
                        name = f'old_t{task_idx}_b{block_idx}_m{msa_idx}'
                        h = adapter.register_forward_hook(hook_fn(name, True))
                        hooks.append(h)

    # Hook current adapters
    if hasattr(backbone, 'cur_adapter'):
        for i, adapter_list in enumerate(backbone.cur_adapter):
            for j, adapter in enumerate(adapter_list):
                if hasattr(adapter, 'lora_A'):
                    name = f'cur_adapter_b{i}_m{j}'
                    h = adapter.register_forward_hook(hook_fn(name, True))
                    hooks.append(h)

    # Test 3: Train mode after task switch
    print("\n3. Train Mode (test=False, cur_task=1):")
    print("   Should activate ONLY current task adapters")
    model._network.train()
    activation_tracker.clear()

    with torch.no_grad():
        _ = model._network(inputs, test=False)

    # Analyze activations
    old_active = [k for k, v in activation_tracker.items()
                  if k.startswith('old_') and v['is_active']]
    cur_active = [k for k, v in activation_tracker.items()
                  if k.startswith('cur_') and v['is_active']]

    print(f"   Old adapters active: {len(old_active)}")
    print(f"   Current adapters active: {len(cur_active)}")

    if old_active:
        print("   ⚠️ WARNING: Old adapters active in train mode!")
        for name in old_active[:3]:  # Show first 3
            print(f"      - {name}")

    # Test 4: Test mode after task switch
    print("\n4. Test Mode (test=True, cur_task=1):")
    print("   Should activate ALL adapters (old + current)")
    model._network.eval()
    activation_tracker.clear()

    with torch.no_grad():
        _ = model._network(inputs, test=True)

    old_active = [k for k, v in activation_tracker.items()
                  if k.startswith('old_') and v['is_active']]
    cur_active = [k for k, v in activation_tracker.items()
                  if k.startswith('cur_') and v['is_active']]

    print(f"   Old adapters active: {len(old_active)}")
    print(f"   Current adapters active: {len(cur_active)}")

    if len(old_active) == 0 and model._cur_task > 0:
        print("   ⚠️ CRITICAL: Old adapters NOT active in test mode!")
        print("      This will cause catastrophic forgetting!")

    # Clean up
    for h in hooks:
        h.remove()

    # Test adapter mapping mechanism
    print("\n" + "-" * 40)
    print("PHASE 3: ADAPTER MAPPING VERIFICATION")
    print("-" * 40)

    if hasattr(backbone, '_active_adapter_mapping'):
        print("\n_active_adapter_mapping found")

        # Force a forward pass and check mapping
        model._network.eval()
        with torch.no_grad():
            _ = backbone(inputs, test=True)

        if hasattr(backbone, '_active_adapter_mapping'):
            mapping = backbone._active_adapter_mapping
            print(f"Mapping keys: {list(mapping.keys())}")

            for block_idx in sorted(mapping.keys()):
                adapters = mapping[block_idx]
                if adapters is not None:
                    print(f"  Block {block_idx}: {len(adapters)} adapters")
                    for i, adapter in enumerate(adapters[:2]):  # Show first 2
                        if adapter is not None:
                            adapter_type = type(adapter).__name__
                            print(f"    MSA {i}: {adapter_type}")
    else:
        print("⚠️ No _active_adapter_mapping attribute found")

    # Final diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if len(old_active) == 0 and model._cur_task > 0:
        print("\n🔴 CRITICAL ISSUE FOUND:")
        print("   Old task adapters are NOT being activated during test mode.")
        print("   This is the primary cause of catastrophic forgetting.\n")
        print("   SOLUTION:")
        print("   1. Check the forward() method in MOMENTWithCLLoRA")
        print("   2. Ensure test=True triggers all adapter activation")
        print("   3. Verify _set_adapter_mapping() includes old_adapter_list")
        print("   4. Check if old adapters are being added to mapping correctly")
    else:
        print("\n✅ Adapter activation mechanism appears to be working correctly")
        print("   Look for other sources of forgetting (hyperparameters, capacity, etc.)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_adapter_activation.py <config_file>")
        sys.exit(1)

    test_adapter_activation(sys.argv[1])