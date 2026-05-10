import torch
from torch import nn

import torch
from torch import nn

class MNISTModelImbuto(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MNISTModelImbuto, self).__init__()
        
        # 1. Appiattiamo l'immagine 28x28 in un vettore 1D da 784 pixel
        self.flatten = nn.Flatten()
        
        # 2. Definiamo la sequenza di layer con struttura a imbuto
        self.linear_relu_stack = nn.Sequential(
            # Primo salto: 784 -> 512
            nn.Linear(input_size, 512),
            nn.ReLU(),
            
            # Secondo salto: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # Terzo salto: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Quarto salto finale: 128 -> 10 (classi)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # I dati entrano come batch di immagini [64, 1, 28, 28]
        x = self.flatten(x)               # Diventano [64, 784]
        logits = self.linear_relu_stack(x) # Passano attraverso l'imbuto
        return logits                     # Escono come [64, 10]

class MNISTModelPiatto(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MNISTModelPiatto, self).__init__()
        
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