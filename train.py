import torch
import torch.nn as nn
import wandb
import argparse
from models.mlp import MNISTModelPiatto, MNISTModelImbuto
from dataset.mnist_loader import get_mnist_loaders

def train(args):
    # 1. Inizializza WandB con la configurazione scelta
    wandb.init(
        project="mnist-comparison",
        config={
            "model_type": args.model,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch,
        }
    )
    config = wandb.config

    # 2. Setup Device e Data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_mnist_loaders(batch_size=config.batch_size)
    
    # 3. Selezione del Modello
    if config.model_type == "imbuto":
        model = MNISTModelImbuto().to(device)
    else:
        model = MNISTModelPiatto().to(device)
    
    print(f"--- Allenamento del modello: {config.model_type} ---")

    # 4. Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 5. Training Loop
    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        })
        
        print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}, Accuracy {epoch_acc:.2f}%")

    # 6. Salva i pesi con un nome specifico per non sovrascriverli
    save_path = f"models/mnist_{config.model_type}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")
    wandb.finish()

if __name__ == "__main__":
    # Parser per gestire gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="MNIST Training Comparison")
    parser.add_argument("--model", type=str, default="piatto", choices=["piatto", "imbuto"], help="Scegli l'architettura")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Numero di epoche")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    train(args)