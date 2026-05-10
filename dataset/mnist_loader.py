from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(data_dir="./data", batch_size=64):
    # 1. Definiamo le trasformazioni: convertiamo in Tensor e normalizziamo
    # MNIST ha immagini in scala di grigio (un solo canale)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Media e deviazione standard di MNIST
    ])

    # 2. Scarichiamo i dati di training e test
    # download=True scaricherà i dati nella cartella 'data' (che è nel .gitignore)
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # 3. Creiamo i DataLoader per iterare sui dati in batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
