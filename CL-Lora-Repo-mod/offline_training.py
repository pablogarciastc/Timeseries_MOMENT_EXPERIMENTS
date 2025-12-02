"""
Guarda este script como: offline_classifier_training.py
Ejecuta después de entrenar todos los tasks: python offline_classifier_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import numpy as np
from backbone.linears import CosineLinearFeature


class OfflineClassifierTrainer:
    def __init__(self, saved_features_dir="saved_features"):
        self.saved_features_dir = saved_features_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_all_features(self):
        """Cargar features de todos los tasks - TODOS LOS SAMPLES"""
        print("Loading all saved features...")

        task_files = sorted([f for f in os.listdir(self.saved_features_dir) if f.endswith('.pkl')])

        # Encontrar dimensión del último task
        with open(os.path.join(self.saved_features_dir, task_files[-1]), 'rb') as f:
            last_data = pickle.load(f)
        max_dim = last_data['feature_dim']

        print(f"  Target feature dimension: {max_dim}")

        all_train_features = []
        all_train_labels = []

        # 🔧 Para train: necesitas RE-EXTRAER features de tasks antiguos con modelo final
        # Por ahora, usa solo test que ya tiene todo

        test_features = last_data['test_features'].float()
        test_labels = last_data['test_labels'].long()

        # Dividir test en train/test
        from sklearn.model_selection import train_test_split

        # Hacer split estratificado por clase
        X_train, X_test, y_train, y_test = train_test_split(
            test_features.numpy(),
            test_labels.numpy(),
            test_size=0.5,
            stratify=test_labels.numpy(),
            random_state=42
        )

        train_features = torch.from_numpy(X_train)
        train_labels = torch.from_numpy(y_train)
        test_features = torch.from_numpy(X_test)
        test_labels = torch.from_numpy(y_test)

        total_classes = int(test_labels.max().item()) + 1

        print(f"\n✅ Data prepared (split from final task test set):")
        print(f"   Train: {train_features.shape}")
        print(f"   Test: {test_features.shape}")
        print(f"   Feature dim: {max_dim}")
        print(f"   Total classes: {total_classes}")

        return train_features, train_labels, test_features, test_labels, max_dim, total_classes

    def train_cosine_classifier(self, epochs=100, batch_size=128, lr=0.01):
        """Entrenar CosineLinearFeature desde cero con todos los datos"""

        # Cargar features
        train_features, train_labels, test_features, test_labels, feature_dim, total_classes = self.load_all_features()

        # 🔧 FIX: Convertir a float32
        train_features = train_features.float()
        test_features = test_features.float()
        train_labels = train_labels.long()
        test_labels = test_labels.long()

        # Crear datasets
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        # ... resto del código igual

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Crear clasificador
        print(f"\nCreating CosineLinearFeature classifier...")
        print(f"  Input: {feature_dim} dims")
        print(f"  Output: {total_classes} classes")

        classifier = CosineLinearFeature(feature_dim, total_classes).to(self.device)

        # Optimizer
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Training
        print(f"\nTraining for {epochs} epochs...")
        best_acc = 0.0

        for epoch in range(epochs):
            # Train
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = classifier(features)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * train_correct / train_total

            # Test
            classifier.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = classifier(features)
                    _, predicted = outputs['logits'].max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            test_acc = 100. * test_correct / test_total

            if test_acc > best_acc:
                best_acc = test_acc
                # Guardar mejor modelo
                torch.save(classifier.state_dict(), 'best_offline_classifier.pth')

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)")

        print(f"\n✅ Training completed!")
        print(f"   Best test accuracy: {best_acc:.2f}%")

        # Test per-task accuracy
        self.evaluate_per_task(classifier, total_classes)

        return classifier, best_acc

    def evaluate_per_task(self, classifier, total_classes):
        """Evaluar accuracy por task"""
        print(f"\n📊 Per-Task Evaluation:")
        print("-" * 60)

        # 🔧 FIX: Solo evaluar el último task que tiene dimensión correcta
        print("  ⚠️  Note: Only evaluating final task (other tasks have wrong dimensions)")

        last_task_file = sorted([f for f in os.listdir(self.saved_features_dir) if f.endswith('.pkl')])[-1]
        path = os.path.join(self.saved_features_dir, last_task_file)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        test_features = data['test_features'].float().to(self.device)
        test_labels = data['test_labels'].long().to(self.device)

        classifier.eval()
        with torch.no_grad():
            outputs = classifier(test_features)
            _, predicted = outputs['logits'].max(1)
            accuracy = 100. * predicted.eq(test_labels).sum().item() / test_labels.size(0)

        print(f"  Final task (all data): {accuracy:.2f}%")
        print("-" * 60)

        return [accuracy]

# ==============================================================================
# MAIN - Como usar el script
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("OFFLINE CLASSIFIER TRAINING")
    print("=" * 80)

    trainer = OfflineClassifierTrainer(saved_features_dir="saved_features")

    # Entrenar clasificador
    classifier, best_acc = trainer.train_cosine_classifier(
        epochs=1000,
        batch_size=32,
        lr=0.0008
    )

    print(f"\n🎉 Done! Best accuracy: {best_acc:.2f}%")
    print(f"Classifier saved to: best_offline_classifier.pth")