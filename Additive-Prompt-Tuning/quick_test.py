#!/usr/bin/env python3
"""
Quick test to see what's happening with data loading
"""

import pickle
import numpy as np
import sys

print("=" * 60)
print("Testing Direct Data Load")
print("=" * 60)

# Test 1: Load files directly
print("\n1. Loading pickle files directly...")
try:
    with open('./data/dailysport/x_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    print(f"✓ x_train.pkl loaded")
    print(f"  Type: {type(X_train)}")
    if isinstance(X_train, np.ndarray):
        print(f"  Shape: {X_train.shape}")
    else:
        print(f"  Length: {len(X_train)}")
        if len(X_train) > 0:
            print(
                f"  First element: {type(X_train[0])}, shape: {X_train[0].shape if hasattr(X_train[0], 'shape') else 'N/A'}")

    with open('./data/dailysport/state_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    print(f"✓ state_train.pkl loaded")
    print(f"  Type: {type(y_train)}")
    if isinstance(y_train, np.ndarray):
        print(f"  Shape: {y_train.shape}")
        print(f"  Unique labels: {np.unique(y_train)}")
    else:
        print(f"  Length: {len(y_train)}")

except Exception as e:
    print(f"✗ Error loading files: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Try the dataloader
print("\n" + "=" * 60)
print("2. Testing Dataloader")
print("=" * 60)

try:
    sys.path.insert(0, '.')
    from dataloaders.dailysport import iDailySport

    print("✓ Imported iDailySport")

    # Define tasks
    tasks = [
        list(range(0, 10)),
        list(range(10, 13)),
        list(range(13, 16)),
        list(range(16, 19)),
    ]

    print(f"✓ Tasks: {tasks}")

    # Create dataset
    dataset = iDailySport(
        root='./data/dailysport',
        train=True,
        tasks=tasks,
        download_flag=False,
        transform=None,
        seed=3,
        rand_split=False,
        validation=False
    )

    print(f"✓ Dataset created")
    print(f"  Full dataset size: {len(dataset.data)}")

    # Load first task
    print(f"\n3. Loading Task 0...")
    dataset.load_dataset(0, train=True)

    print(f"✓ Task 0 loaded")
    print(f"  Task dataset size: {len(dataset.task_data)}")
    print(f"  Task labels: {np.unique(dataset.task_labels)}")

    # Get a sample
    if len(dataset) > 0:
        data, label, task_id = dataset[0]
        print(f"\n✓ Sample retrieved:")
        print(f"  Data shape: {data.shape}")
        print(f"  Label: {label}")
        print(f"  Task ID: {task_id}")
    else:
        print(f"✗ Task has 0 samples!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Check what trainer is doing
print("\n" + "=" * 60)
print("3. Checking Trainer Setup")
print("=" * 60)

try:
    # Check which trainer is being used
    with open('run.py', 'r') as f:
        run_content = f.read()

    if 'TimeSeriesTrainer' in run_content:
        print("✓ run.py imports TimeSeriesTrainer")
    elif 'from trainer import Trainer' in run_content:
        print("✗ run.py imports wrong Trainer!")
        print("  Should import: from trainer_timeseries import TimeSeriesTrainer")
    else:
        print("? Could not determine trainer import")

    # Check if trainer_timeseries exists
    import os

    if os.path.exists('trainer_timeseries.py'):
        print("✓ trainer_timeseries.py exists")
    else:
        print("✗ trainer_timeseries.py NOT FOUND")

except Exception as e:
    print(f"? Error checking trainer: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)
print("\nIf all checks above passed, the issue is likely:")
print("1. run.py is importing the WRONG trainer")
print("2. The trainer is not using the DailySport dataloader")
print("\nNext step: Check what dataset trainer_timeseries.py is creating")