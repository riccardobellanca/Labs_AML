import torch
from torch import nn

class MNISTModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MNISTModel, self).__init__()
        
        # 1. Appiattiamo l'immagine 28x28 in un vettore 1D da 784 pixel
        self.flatten = nn.Flatten() 
        
        # 2. Definiamo la sequenza di layer (lo "stack")
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, num_classes), 
        )

    def forward(self, x):
        # Definiamo come i dati passano attraverso la rete
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits