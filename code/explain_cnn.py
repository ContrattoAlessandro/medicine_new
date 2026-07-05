import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_resnet1d18
from models.resnet1d import resnet1d18

def main():
    print("Setting up paths and loading experiment...")
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'
    plots_dir = os.path.join(outputfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize the superdiagnostic experiment to load and preprocess the validation data
    exp_name = 'exp1.1.1_filtered_aug_balanced'
    e = SCP_Experiment(
        experiment_name=exp_name,
        task='superdiagnostic',
        datafolder=datafolder,
        outputfolder=outputfolder,
        models=[conf_fastai_resnet1d18],
        augment=True,
        filter_type='bandpass',
        class_balancing=True
    )
    
    print("Preparing validation data...")
    e.prepare()
    
    X_val = e.X_val  # Shape: (N, 1000, 12)
    y_val = e.y_val  # Shape: (N, 5)
    n_classes = e.n_classes
    
    # Load MLB to get class names
    mlb_path = os.path.join(outputfolder, exp_name, 'data/mlb.pkl')
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    classes = list(mlb.classes_)
    print(f"Loaded {len(classes)} classes: {classes}")
    
    # Load ResNet1d18 model
    model_checkpoint_path = os.path.join(outputfolder, exp_name, 'models/fastai_resnet1d18/models/fastai_resnet1d18.pth')
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}.")
        sys.exit(1)
        
    print(f"Loading ResNet1d18 model from {model_checkpoint_path}...")
    model = resnet1d18(num_classes=n_classes, input_channels=12, inplanes=128, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded and sent to {device}.")
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Store aggregated absolute gradients per class
    # class_gradients[class_idx] will be a list of lead importances (12,)
    class_gradients = {c: [] for c in range(n_classes)}
    
    print("Computing attribution scores (gradients) on validation set...")
    # Convert X_val to PyTorch tensor and send to device
    # Shape: (N, 12, 1000) for Conv1D
    X_val_torch = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32).to(device)
    
    for i in range(len(X_val)):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"Processing record {i + 1}/{len(X_val)}...")
            
        x_i = X_val_torch[i:i+1].clone()  # Shape: (1, 12, 1000)
        x_i.requires_grad = True
        
        # Forward pass
        logits = model(x_i)
        
        # Backward pass for each active class in the ground truth
        active_classes = np.where(y_val[i] == 1.0)[0]
        for c in active_classes:
            model.zero_grad()
            if x_i.grad is not None:
                x_i.grad.zero_()
                
            logit = logits[0, c]
            logit.backward(retain_graph=True)
            
            # Extract gradient shape (12, 1000)
            grad = x_i.grad.detach().cpu().numpy()[0]
            # Saliency map is the absolute value of gradient
            abs_grad = np.abs(grad)
            # Average over time dimension (axis=1) to get importance per channel
            mean_abs_grad = np.mean(abs_grad, axis=1)  # Shape: (12,)
            
            class_gradients[c].append(mean_abs_grad)
            
    print("Aggregating results...")
    # Compute mean lead importance for each class
    mean_class_importances = {}
    for c in range(n_classes):
        if len(class_gradients[c]) > 0:
            mean_class_importances[classes[c]] = np.mean(class_gradients[c], axis=0)
        else:
            mean_class_importances[classes[c]] = np.zeros(12)
            
    # Universal lead importance: average of class-specific importances to give equal weight to each class
    class_importances_df = pd.DataFrame(mean_class_importances, index=lead_names)
    class_importances_df['Universal'] = class_importances_df.mean(axis=1)
    
    # Sort by Universal importance descending
    class_importances_df = class_importances_df.sort_values(by='Universal', ascending=False)
    
    # Save rankings to CSV
    results_dir = os.path.join(outputfolder, exp_name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'cnn_lead_importance.csv')
    class_importances_df.to_csv(csv_path)
    print(f"Lead importances saved to {csv_path}")
    print("\n--- Universal Lead Importance Ranking ---")
    for idx, (lead, val) in enumerate(zip(class_importances_df.index, class_importances_df['Universal'])):
        print(f"Rank {idx+1}: Lead {lead} (Score: {val:.6f})")
        
    # Generate visualization
    print("Generating premium visualization...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Bar Plot of Universal Importance
    universal_sorted = class_importances_df['Universal'].sort_values(ascending=True)
    norm_color = (universal_sorted - universal_sorted.min()) / (universal_sorted.max() - universal_sorted.min() + 1e-8)
    colors = plt.cm.viridis(norm_color)
    
    axes[0].barh(universal_sorted.index, universal_sorted.values, color=colors, edgecolor='none', alpha=0.9)
    axes[0].set_title("Universal Lead Importance Ranking (Average Across Classes)", fontweight='bold', pad=15)
    axes[0].set_xlabel("Mean Absolute Gradient Score")
    axes[0].set_ylabel("ECG Lead")
    
    # Add value labels to the bars
    for i, v in enumerate(universal_sorted.values):
        axes[0].text(v + (universal_sorted.max() * 0.01), i, f"{v:.5f}", va='center', fontsize=10)
        
    # 2. Heatmap of Class-Specific Importance
    import seaborn as sns
    # Plot class-specific importances (excluding Universal column) sorted by universal importance
    heatmap_data = class_importances_df.drop(columns=['Universal'])
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1], cbar_kws={'label': 'Gradient Score'})
    axes[1].set_title("Lead Importance by Diagnostic Pathology", fontweight='bold', pad=15)
    axes[1].set_ylabel("ECG Lead")
    axes[1].set_xlabel("Diagnostic Class")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'cnn_lead_importance.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {plot_path}")

if __name__ == '__main__':
    main()
