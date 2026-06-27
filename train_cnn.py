"""
train_cnn.py
============
Defines the Dual-View 1D CNN architecture in PyTorch, trains it on
the prepared dataset, and saves the weights to models/cnn_classifier.pt.
"""

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "cnn_classifier.pt"
DATA_PATH = Path(__file__).parent / "models" / "cnn_dataset.npz"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class DualViewCNN(nn.Module):
    """
    AstroNet-inspired Dual-View 1D CNN.
    Processes global and local folded light curves in parallel,
    concatenates representations, and incorporates scalar features for final prediction.
    """
    def __init__(self):
        super(DualViewCNN, self).__init__()
        
        # --- Global View Stream ---
        # Input shape: (Batch, 1, 2001)
        self.global_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten()
        )
        # 2001 // 4 = 500 // 4 = 125 // 4 = 31. Shape: 31 * 64 = 1984
        
        # --- Local View Stream ---
        # Input shape: (Batch, 1, 201)
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        # 201 // 2 = 100 // 2 = 50 // 2 = 25. Shape: 25 * 64 = 1600
        
        # --- Dense Fusion Layers ---
        # Flattened global (1984) + local (1600) + scalars (5 features) = 3589
        self.fc = nn.Sequential(
            nn.Linear(1984 + 1600 + 5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # outputs PC=0, AFP=1, NTP=2 logits
        )

    def forward(self, x_global, x_local, x_scalars):
        # Ensure correct channel dimension: (Batch, 1, SequenceLength)
        if len(x_global.shape) == 2:
            x_global = x_global.unsqueeze(1)
        if len(x_local.shape) == 2:
            x_local = x_local.unsqueeze(1)
            
        g_feat = self.global_conv(x_global)
        l_feat = self.local_conv(x_local)
        
        # Concatenate features
        fused = torch.cat((g_feat, l_feat, x_scalars), dim=1)
        logits = self.fc(fused)
        return logits

def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run prepare_cnn_data.py first.")
        
    data = np.load(DATA_PATH)
    x_global = torch.tensor(data["global_view"], dtype=torch.float32)
    x_local = torch.tensor(data["local_view"], dtype=torch.float32)
    x_scalars = torch.tensor(data["scalars"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.int64)
    
    # 80/20 train/test split
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(n_samples * 0.8)
    
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_dataset = TensorDataset(x_global[train_idx], x_local[train_idx], x_scalars[train_idx], y[train_idx])
    test_dataset = TensorDataset(x_global[test_idx], x_local[test_idx], x_scalars[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = DualViewCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    logger.info("Starting CNN training for %d epochs...", epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for g, l, s, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(g, l, s)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * g.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for g, l, s, labels in test_loader:
                outputs = model(g, l, s)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * g.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(test_dataset)
        val_acc = val_correct / val_total * 100.0
        
        logger.info(
            "Epoch %d/%d: Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch + 1, epochs, epoch_loss, epoch_acc, val_epoch_loss, val_acc
        )
        
    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("CNN model saved successfully to %s", MODEL_PATH)

if __name__ == "__main__":
    train_model()
