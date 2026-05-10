import torch
import torch.nn as nn
import wandb
from models.mlp import MNISTModel
from dataset.mnist_loader import get_mnist_loaders

def train():
    # 1. Inizializza WandB
    wandb.init(
        project="mnist-professional-sandbox",
        config={
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 64,
            "hidden_size": 512
        }
    )
    config = wandb.config # Accediamo agli iperparametri via config

    # 2. Setup Device, Data e Modello
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_mnist_loaders(batch_size=config.batch_size)
    
    model = MNISTModel(hidden_size=config.hidden_size).to(device)
    
    # 3. Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 4. Training Loop [cite: 562, 563, 591]
    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reset gradienti
            optimizer.zero_grad()
            
            # Forward
            output = model(data)
            loss = criterion(output, target)
            
            # Backward + Step
            loss.backward()
            optimizer.step()
            
            # Statistiche
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        # 5. Log delle metriche su WandB 
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        })
        
        print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}, Accuracy {epoch_acc:.2f}%")

    # 6. Salva i pesi localmente
    torch.save(model.state_dict(), "models/mnist_model.pth")
    wandb.finish()

if __name__ == "__main__":
    train()