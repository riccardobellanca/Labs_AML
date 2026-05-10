import torch
import torch.nn as nn
import argparse
from models.mlp import MNISTModelPiatto, MNISTModelImbuto
from dataset.mnist_loader import get_mnist_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def evaluate_model(model, test_loader, device, model_name="Model"):
    """
    Valuta un modello sul test set e ritorna metriche dettagliate.
    
    Args:
        model: Modello da valutare
        test_loader: DataLoader per il test set
        device: Device su cui eseguire (cuda/cpu)
        model_name: Nome del modello per i log
    
    Returns:
        Dict con tutte le metriche e le predizioni
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Valutazione {model_name} ---")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Conversione a numpy array
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calcolo metriche
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(test_loader)
    
    # Matrice di confusione
    cm = confusion_matrix(all_targets, all_preds)
    
    # Metriche per classe
    class_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'avg_loss': avg_loss,
        'confusion_matrix': cm,
        'class_report': class_report,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    # Stampa risultati
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision_macro:.4f}")
    print(f"Recall:       {recall_macro:.4f}")
    print(f"F1-Score:     {f1_macro:.4f}")
    print(f"Avg Loss:     {avg_loss:.4f}")
    
    return results

def plot_confusion_matrix(cm, model_name, save_path="results/"):
    """Crea e salva una visualizzazione della matrice di confusione."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Matrice di confusione salvata in {save_path}{model_name}_confusion_matrix.png")

def compare_models(results_dict, save_path="results/"):
    """Crea un confronto visivo tra i modelli."""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    data_for_comparison = {metric: [results_dict[model][metric] for model in models] for metric in metrics}
    
    df = pd.DataFrame(data_for_comparison, index=models)
    
    # Grafico a barre
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    plt.title('Confronto Modelli - Metriche di Valutazione')
    plt.ylabel('Score')
    plt.xlabel('Modello')
    plt.legend(title='Metrica')
    plt.tight_layout()
    plt.savefig(f'{save_path}model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Grafico di confronto salvato in {save_path}model_comparison.png")
    
    # Salva anche come CSV
    df.to_csv(f'{save_path}model_comparison.csv')
    print(f"Risultati salvati in {save_path}model_comparison.csv")
    
    return df

def save_detailed_report(results_dict, save_path="results/"):
    """Salva un report dettagliato in formato testo."""
    with open(f'{save_path}evaluation_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MNIST MODELS EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, results in results_dict.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Accuracy:     {results['accuracy']:.4f}\n")
            f.write(f"Precision:    {results['precision_macro']:.4f}\n")
            f.write(f"Recall:       {results['recall_macro']:.4f}\n")
            f.write(f"F1-Score:     {results['f1_macro']:.4f}\n")
            f.write(f"Avg Loss:     {results['avg_loss']:.4f}\n\n")
            
            f.write("Classification Report per classe:\n")
            f.write("-" * 80 + "\n")
            class_df = pd.DataFrame(results['class_report']).transpose()
            f.write(class_df.to_string())
            f.write("\n\n")
    
    print(f"Report dettagliato salvato in {save_path}evaluation_report.txt")

def main(args):
    import os
    
    # Crea la directory per i risultati se non esiste
    os.makedirs('results', exist_ok=True)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizzo device: {device}")
    
    # Carica il test set
    print("\nCaricamento dati MNIST...")
    _, test_loader = get_mnist_loaders(batch_size=args.batch)
    print(f"Test set caricato: {len(test_loader)} batch")
    
    results_dict = {}
    
    # Valuta il modello Piatto se esiste
    piatto_path = "models/mnist_piatto.pth"
    if os.path.exists(piatto_path):
        print(f"\nCaricamento modello Piatto da {piatto_path}...")
        model_piatto = MNISTModelPiatto().to(device)
        model_piatto.load_state_dict(torch.load(piatto_path, map_location=device))
        results_piatto = evaluate_model(model_piatto, test_loader, device, "Piatto")
        results_dict["Piatto"] = results_piatto
        
        if args.plot:
            plot_confusion_matrix(results_piatto['confusion_matrix'], "Piatto")
    else:
        print(f"Attenzione: {piatto_path} non trovato!")
    
    # Valuta il modello Imbuto se esiste
    imbuto_path = "models/mnist_imbuto.pth"
    if os.path.exists(imbuto_path):
        print(f"\nCaricamento modello Imbuto da {imbuto_path}...")
        model_imbuto = MNISTModelImbuto().to(device)
        model_imbuto.load_state_dict(torch.load(imbuto_path, map_location=device))
        results_imbuto = evaluate_model(model_imbuto, test_loader, device, "Imbuto")
        results_dict["Imbuto"] = results_imbuto
        
        if args.plot:
            plot_confusion_matrix(results_imbuto['confusion_matrix'], "Imbuto")
    else:
        print(f"Attenzione: {imbuto_path} non trovato!")
    
    # Confronto dei modelli
    if len(results_dict) > 1:
        print("\n" + "="*80)
        print("CONFRONTO TRA I MODELLI")
        print("="*80)
        comparison_df = compare_models(results_dict)
        print("\n" + comparison_df.to_string())
        
        # Determina il miglior modello
        best_model = max(results_dict, key=lambda x: results_dict[x]['accuracy'])
        print(f"\nMiglior modello: {best_model} (Accuracy: {results_dict[best_model]['accuracy']:.4f})")
    
    # Salva report dettagliato
    save_detailed_report(results_dict)
    
    print("\n" + "="*80)
    print("Valutazione completata! I risultati sono stati salvati in 'results/'")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Models Evaluation")
    parser.add_argument("--batch", type=int, default=64, help="Batch size per la valutazione")
    parser.add_argument("--plot", action="store_true", help="Genera grafici delle matrici di confusione e del confronto")
    
    args = parser.parse_args()
    main(args)
