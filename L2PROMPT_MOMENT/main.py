import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

# Importar MOMENT (asumiendo que estÃ¡ instalado)
try:
    from momentfm import MOMENTPipeline
except ImportError:
    print("WARNING: MOMENT no estÃ¡ instalado. Instala con: pip install momentfm")
    print("Usando mock para demostraciÃ³n")
    MOMENTPipeline = None

try:
    from momentfm.utils.utils import control_randomness
except ImportError:
    control_randomness = None


class L2PromptLayer(nn.Module):
    """Capa de prompts aprendibles para series temporales"""

    def __init__(self, n_prompts, prompt_length, d_model):
        super().__init__()
        self.n_prompts = n_prompts
        self.prompt_length = prompt_length
        self.d_model = d_model

        # Prompts aprendibles - inicializados aleatoriamente
        self.prompts = nn.Parameter(
            torch.randn(n_prompts, prompt_length, d_model) * 0.01
        )

    def forward(self, x, prompt_idx=None):
        """
        Args:
            x: embeddings de MOMENT [batch, n_channels, n_patches, d_model] o [batch, n_patches, d_model]
            prompt_idx: Ã­ndice del prompt a usar [batch] (opcional, si None usa prompt 0)
        Returns:
            x con prompts concatenados
        """
        # Verificar dimensiones de x
        if len(x.shape) == 4:
            # [batch, n_channels, n_patches, d_model]
            # Necesitamos aplanar primero: combinar n_channels y n_patches
            batch_size, n_channels, n_patches, d_model = x.shape
            x = x.reshape(batch_size, n_channels * n_patches, d_model)
        elif len(x.shape) == 3:
            # [batch, n_patches, d_model] - ya estÃ¡ en el formato correcto
            batch_size = x.size(0)
        else:
            raise ValueError(f"Expected x to have 3 or 4 dimensions, got {len(x.shape)}")

        # Si no se proporciona prompt_idx, usar el primero para todos
        if prompt_idx is None:
            prompt_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # Seleccionar prompts segÃºn el Ã­ndice
        selected_prompts = self.prompts[prompt_idx]  # [batch, prompt_length, d_model]

        # Concatenar prompts al inicio de la secuencia
        x_prompted = torch.cat([selected_prompts, x], dim=1)

        return x_prompted


class MOMENTWithL2Prompt(nn.Module):
    """MOMENT backbone congelado + L2Prompt + clasificador"""

    def __init__(self, n_classes, n_prompts=18, prompt_length=5, d_model=None, moment_model='small'):
        super().__init__()

        # MOMENT backbone (congelado)
        if MOMENTPipeline:
            # Usar modelo small por defecto
            model_name = f"AutonLab/MOMENT-1-{moment_model}"
            print(f"Loading {model_name}...")

            # Cargar modelo MOMENT para embedding
            self.moment = MOMENTPipeline.from_pretrained(
                model_name,
                model_kwargs={
                    'task_name': 'embedding',
                    'enable_gradient_checkpointing': False,
                }
            )
            self.moment.init()

            # Obtener dimensiÃ³n real de MOMENT desde config
            if hasattr(self.moment, 'config'):
                moment_d_model = self.moment.config.d_model
            else:
                # Fallback: MOMENT-1-small=512, large=1024
                moment_d_model = 512 if moment_model == 'small' else 1024

            d_model = moment_d_model
            print(f"MOMENT d_model: {d_model}")

        else:
            # Mock para demostraciÃ³n
            self.moment = nn.Identity()
            d_model = d_model or 512

        # Congelar MOMENT
        if hasattr(self, 'moment') and not isinstance(self.moment, nn.Identity):
            for param in self.moment.parameters():
                param.requires_grad = False
            print(f"âœ“ MOMENT backbone frozen")

        self.d_model = d_model

        # Capa de prompts aprendibles (18 actividades)
        self.l2prompt = L2PromptLayer(n_prompts, prompt_length, d_model)

        # Clasificador sobre las representaciones con prompts
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)  # 18 clases
        )

    def forward(self, x, prompt_idx=None):
        """
        Args:
            x: series temporales [batch, channels, seq_len]
            prompt_idx: Ã­ndice del prompt [batch] - opcional
        """
        # Obtener embeddings de MOMENT (congelado)
        with torch.no_grad():
            if isinstance(self.moment, nn.Identity):
                # Mock: crear embeddings ficticios
                batch_size = x.size(0)
                moment_embeds = torch.randn(batch_size, 100, self.d_model).to(x.device)
            else:
                # MOMENT 0.1.4 API:
                # Input debe ser pasado como keyword argument 'x_enc'
                # x debe tener forma [batch, n_channels, seq_len]
                # usar reduction='none' para obtener embeddings completos

                output = self.moment.embed(x_enc=x, reduction='none')
                moment_embeds = output.embeddings

                # Debug: mostrar shape una vez
                if not hasattr(self, '_shape_printed'):
                    print(f"MOMENT embeddings shape: {moment_embeds.shape}")
                    self._shape_printed = True

        # Aplicar L2Prompt
        prompted_embeds = self.l2prompt(moment_embeds, prompt_idx)

        # Pooling temporal (mean over sequence dimension)
        pooled = prompted_embeds.mean(dim=1)  # [batch, d_model]

        # Clasificar
        logits = self.classifier(pooled)

        return logits


def load_dataset(file_path):
    """Carga datos de DailySport desde pickle"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_dataset(x_path, state_path):
    """
    Prepara dataset de DailySport
    DailySport tiene 19 actividades con datos de 45 sensores
    """
    # Cargar datos
    x_data = load_dataset(x_path)
    states = load_dataset(state_path)

    # Convertir a tensores
    if isinstance(x_data, np.ndarray):
        x_tensor = torch.FloatTensor(x_data)
    else:
        x_tensor = x_data

    if isinstance(states, np.ndarray):
        y_tensor = torch.LongTensor(states)
    else:
        y_tensor = states

    # DailySport: asegurar formato [batch, channels, seq_len]
    if len(x_tensor.shape) == 2:
        x_tensor = x_tensor.unsqueeze(1)

    return x_tensor, y_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena una Ã©poca"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Usar la etiqueta como Ã­ndice de prompt
        prompt_idx = y_batch

        # Forward
        optimizer.zero_grad()
        logits = model(x_batch, prompt_idx)
        loss = criterion(logits, y_batch)

        # Backward
        loss.backward()
        optimizer.step()

        # MÃ©tricas
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """EvalÃºa el modelo"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            prompt_idx = y_batch

            logits = model(x_batch, prompt_idx)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Cargar datos
    print("Loading DailySport dataset...")
    x_train, y_train = prepare_dataset(args.x_train, args.state_train)
    x_test, y_test = prepare_dataset(args.x_test, args.state_test)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Number of classes: {len(torch.unique(y_train))}")

    # Crear dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Crear modelo
    n_classes = len(torch.unique(y_train))
    model = MOMENTWithL2Prompt(
        n_classes=n_classes,
        n_prompts=n_classes,  # Un prompt por clase
        prompt_length=args.prompt_length,
        d_model=args.d_model,
        moment_model=args.moment_model  # 'small' o 'large'
    ).to(device)

    print(f"\nModel architecture:")
    print(f"MOMENT model: {args.moment_model}")
    print(model)

    # Contar parÃ¡metros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Optimizer y loss (solo parÃ¡metros entrenables)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Guardar mejor modelo
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, args.save_path)
            print(f"âœ“ Saved best model (acc: {best_acc:.2f}%)")

    print(f"\n{'=' * 50}")
    print(f"Training completed! Best test accuracy: {best_acc:.2f}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MOMENT + L2Prompt on DailySport')

    # Data paths
    parser.add_argument('--x_train', type=str, default='x_train.pkl')
    parser.add_argument('--x_test', type=str, default='x_test.pkl')
    parser.add_argument('--state_train', type=str, default='state_train.pkl')
    parser.add_argument('--state_test', type=str, default='state_test.pkl')

    # Model hyperparameters
    parser.add_argument('--moment_model', type=str, default='small', choices=['small', 'large'],
                        help='MOMENT model size (small or large)')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=None,
                        help='Model dimension (auto-detected from MOMENT if not specified)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_tasks', type=int, default=6)

    # Save path
    parser.add_argument('--save_path', type=str, default='best_model.pt')

    args = parser.parse_args()
    main(args)