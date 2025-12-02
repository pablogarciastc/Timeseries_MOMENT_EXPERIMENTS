"""
Script de debug para identificar por qué el modelo no aprende
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def debug_single_batch(model, x_batch, y_batch, task_id):
    """
    Debuggea un batch completo para ver qué está pasando
    """
    print("\n" + "="*80)
    print("🔍 DEBUG SINGLE BATCH")
    print("="*80)

    device = x_batch.device
    model.train()

    # 1. Check input
    print(f"\n1. INPUT:")
    print(f"   x_batch shape: {x_batch.shape}")
    print(f"   y_batch: {y_batch.tolist()}")
    print(f"   y_batch unique: {torch.unique(y_batch).tolist()}")
    print(f"   task_id: {task_id}")

    # 2. Forward pass con hooks
    print(f"\n2. FORWARD PASS:")

    # Text encoder
    with torch.no_grad():
        base_feats = model.text_encoder(x_batch)
    print(f"   text_encoder output: {base_feats.shape}, mean={base_feats.mean().item():.4f}, std={base_feats.std().item():.4f}")

    # Extract frozen features
    text_feats, moment_feats = model._extract_frozen_features(x_batch)
    print(f"   text_feats: {text_feats.shape}, mean={text_feats.mean().item():.4f}, std={text_feats.std().item():.4f}")
    print(f"   moment_feats: {moment_feats.shape}, mean={moment_feats.mean().item():.4f}, std={moment_feats.std().item():.4f}")

    # Projection
    moment_projected = model.moment_to_work(moment_feats)
    print(f"   moment_to_work output: {moment_projected.shape}, mean={moment_projected.mean().item():.4f}, std={moment_projected.std().item():.4f}")
    print(f"   moment_to_work.weight grad: {model.moment_to_work.weight.requires_grad}")

    # Combine
    combined = text_feats + moment_projected
    print(f"   combined: mean={combined.mean().item():.4f}, std={combined.std().item():.4f}")

    # Adapter
    adapted = model.feature_adapter(combined)
    print(f"   feature_adapter output: {adapted.shape}, mean={adapted.mean().item():.4f}, std={adapted.std().item():.4f}")
    print(f"   feature_adapter[0].weight grad: {list(model.feature_adapter.parameters())[0].requires_grad}")

    # Prompts
    selected_prompts, prompt_info = model.l2prompt.select_prompts_from_query(adapted.unsqueeze(1))
    print(f"   L2Prompt selected: {selected_prompts.shape}")
    if isinstance(prompt_info, dict) and 'top_k_indices' in prompt_info:
        print(f"   L2Prompt indices: {prompt_info['top_k_indices'][0].tolist()}")

    prompt_feats = selected_prompts.mean(dim=1)
    print(f"   prompt_feats: mean={prompt_feats.mean().item():.4f}, std={prompt_feats.std().item():.4f}")
    print(f"   L2Prompt.prompts grad: {model.l2prompt.prompts.requires_grad}")

    # Combine
    pooled = adapted + prompt_feats
    print(f"   pooled (adapted+prompts): mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")

    # Task prediction
    task_info = model.task_predictor(pooled, training=True, task_id=task_id)
    print(f"   task_logits: {task_info['task_logits'][0].tolist()}")
    print(f"   task_pred: {task_info['predicted_task']}")

    # Classification
    logits = model.classifier(pooled, task_id)
    print(f"   classifier logits: {logits[0].tolist()}")
    print(f"   classifier head grad: {list(model.classifier.heads[task_id].parameters())[0].requires_grad}")

    # 3. Loss & backward
    print(f"\n3. LOSS & BACKWARD:")

    # 🔥 CRITICAL: Check if labels are already local or global
    print(f"   y_batch range: [{y_batch.min().item()}, {y_batch.max().item()}]")
    print(f"   Expected local range: [0, {model.classes_per_task-1}]")
    print(f"   Expected global range for task {task_id}: [{task_id * model.classes_per_task}, {(task_id+1) * model.classes_per_task - 1}]")

    # Determine if labels are local or global
    if y_batch.min() >= task_id * model.classes_per_task and y_batch.max() < (task_id + 1) * model.classes_per_task:
        print(f"   ✓ Labels are GLOBAL → converting to local")
        local_labels = y_batch - (task_id * model.classes_per_task)
    elif y_batch.min() >= 0 and y_batch.max() < model.classes_per_task:
        print(f"   ✓ Labels are already LOCAL → no conversion needed")
        local_labels = y_batch
    else:
        print(f"   ❌ WARNING: Label range is ambiguous!")
        local_labels = y_batch - (task_id * model.classes_per_task)

    print(f"   local_labels: {local_labels.tolist()}")

    if local_labels.min() < 0 or local_labels.max() >= model.classes_per_task:
        print(f"   ❌ ERROR: local_labels out of range! Min={local_labels.min()}, Max={local_labels.max()}")
        print(f"   This will cause training to fail!")

    cls_loss = nn.CrossEntropyLoss()(logits, local_labels)
    print(f"   cls_loss: {cls_loss.item():.4f}")

    task_targets = torch.full((x_batch.size(0),), task_id, dtype=torch.long, device=device)
    task_loss = nn.CrossEntropyLoss()(task_info['task_logits'], task_targets)
    print(f"   task_loss: {task_loss.item():.4f}")

    total_loss = cls_loss + 1.0 * task_loss
    print(f"   total_loss: {total_loss.item():.4f}")

    # Backward
    total_loss.backward()

    # 4. Check gradients
    print(f"\n4. GRADIENTS:")

    # Classifier
    cls_head_first_param = list(model.classifier.heads[task_id].parameters())[0]
    if cls_head_first_param.grad is not None:
        print(f"   ✓ classifier.heads[{task_id}][0].weight.grad: mean={cls_head_first_param.grad.mean().item():.6f}, norm={cls_head_first_param.grad.norm().item():.6f}")
    else:
        print(f"   ❌ classifier.heads[{task_id}][0].weight.grad: None")

    # Prompts
    if model.l2prompt.prompts.grad is not None:
        print(f"   ✓ l2prompt.prompts.grad: mean={model.l2prompt.prompts.grad.mean().item():.6f}, norm={model.l2prompt.prompts.grad.norm().item():.6f}")
    else:
        print(f"   ❌ l2prompt.prompts.grad: None")

    # Feature adapter
    adapter_first_param = list(model.feature_adapter.parameters())[0]
    if adapter_first_param.grad is not None:
        print(f"   ✓ feature_adapter[0].weight.grad: mean={adapter_first_param.grad.mean().item():.6f}, norm={adapter_first_param.grad.norm().item():.6f}")
    else:
        print(f"   ❌ feature_adapter[0].weight.grad: None")

    # Moment projection
    if model.moment_to_work.weight.grad is not None:
        print(f"   ✓ moment_to_work.weight.grad: mean={model.moment_to_work.weight.grad.mean().item():.6f}, norm={model.moment_to_work.weight.grad.norm().item():.6f}")
    else:
        print(f"   ❌ moment_to_work.weight.grad: None")

    # Task keys
    if model.task_predictor.task_keys[task_id].grad is not None:
        print(f"   ✓ task_keys[{task_id}].grad: mean={model.task_predictor.task_keys[task_id].grad.mean().item():.6f}, norm={model.task_predictor.task_keys[task_id].grad.norm().item():.6f}")
    else:
        print(f"   ❌ task_keys[{task_id}].grad: None")

    # 5. Predictions
    print(f"\n5. PREDICTIONS:")
    _, predicted = logits.max(1)
    predicted_global = predicted + (task_id * model.classes_per_task)
    print(f"   predicted (local): {predicted.tolist()}")
    print(f"   predicted (global): {predicted_global.tolist()}")
    print(f"   ground truth: {y_batch.tolist()}")
    print(f"   correct: {predicted.eq(local_labels).tolist()}")

    acc = predicted.eq(local_labels).sum().item() / local_labels.size(0)
    print(f"   batch accuracy: {acc*100:.2f}%")

    print("="*80 + "\n")

    return {
        'loss': total_loss.item(),
        'cls_loss': cls_loss.item(),
        'task_loss': task_loss.item(),
        'accuracy': acc
    }


def debug_training_loop(trainer, task_id, num_debug_batches=3):
    """
    Ejecuta debug en las primeras batches del training
    """
    print("\n" + "="*80)
    print(f"🔍 DEBUG TRAINING - TASK {task_id}")
    print("="*80)

    x_train, y_train = trainer.task_train_data[task_id]

    print(f"\n📊 DATASET INFO:")
    print(f"   x_train shape: {x_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_train unique: {torch.unique(y_train).tolist()}")
    print(f"   y_train range: [{y_train.min().item()}, {y_train.max().item()}]")
    print(f"   Expected for task {task_id}: classes {trainer.task_classes[task_id]}")

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=trainer.args.batch_size, shuffle=True)

    # Setup optimizer
    optimizer_params = [
        {'params': trainer.model.classifier.heads[task_id].parameters(), 'lr': trainer.args.lr},
        {'params': trainer.model.l2prompt.parameters(), 'lr': trainer.args.lr * 0.1},
        {'params': [trainer.model.task_predictor.task_keys[task_id]], 'lr': trainer.args.lr * 0.05},
        {'params': trainer.model.moment_to_work.parameters(), 'lr': trainer.args.lr * 0.1},
        {'params': trainer.model.feature_adapter.parameters(), 'lr': trainer.args.lr * 0.1},
    ]
    optimizer = optim.Adam(optimizer_params)

    # Check trainable params
    print("\n📊 TRAINABLE PARAMETERS:")
    total_params = 0
    trainable_params = 0
    for name, param in trainer.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'heads.0' in name or 'l2prompt' in name or 'moment_to_work' in name or 'feature_adapter' in name or 'task_keys.0' in name:
                print(f"   ✓ {name}: {param.shape}")
    print(f"\nTotal: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Debug first few batches
    trainer.model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        if batch_idx >= num_debug_batches:
            break

        x_batch = x_batch.to(trainer.device)
        y_batch = y_batch.to(trainer.device)

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_debug_batches}")
        print(f"{'='*80}")

        optimizer.zero_grad()
        result = debug_single_batch(trainer.model, x_batch, y_batch, task_id)
        optimizer.step()

        print(f"\n6. AFTER OPTIMIZER STEP:")
        print(f"   Loss: {result['loss']:.4f}")
        print(f"   Accuracy: {result['accuracy']*100:.2f}%")

    print("\n" + "="*80)
    print("🏁 DEBUG COMPLETE")
    print("="*80)


# Añade esto a tu continual_learning.py
def main_with_debug():
    """
    Versión de main() con debugging activado
    """
    import argparse
    from pathlib import Path
    from continual_learning import ContinualLearningTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dailysport')
    parser.add_argument('--x_train', type=str, default='x_train.pkl')
    parser.add_argument('--x_test', type=str, default='x_test.pkl')
    parser.add_argument('--state_train', type=str, default='state_train.pkl')
    parser.add_argument('--state_test', type=str, default='state_test.pkl')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--n_tasks', type=int, default=6)
    parser.add_argument('--task_loss_weight', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='continual_results')
    parser.add_argument('--save_checkpoints', action='store_true')
    args = parser.parse_args()

    # Create trainer
    trainer = ContinualLearningTrainer(args)

    # 🔥 DEBUG FIRST TASK ONLY
    print("\n" + "="*80)
    print("🚨 RUNNING IN DEBUG MODE - TASK 0 ONLY")
    print("="*80)

    debug_training_loop(trainer, task_id=0, num_debug_batches=3)

    print("\n✅ Debug complete. Check the output above to identify the issue.")


if __name__ == "__main__":
    main_with_debug()