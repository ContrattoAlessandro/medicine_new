import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_resnet1d18
from models.resnet1d import resnet1d18

def parse_exp_settings(exp_name):
    # Determine task
    if exp_name.startswith('exp1.1.1'):
        task = 'superdiagnostic'
    elif exp_name.startswith('exp1.1'):
        task = 'subdiagnostic'
    elif exp_name.startswith('exp1'):
        task = 'diagnostic'
    elif exp_name.startswith('exp0'):
        task = 'all'
    elif exp_name.startswith('exp2'):
        task = 'form'
    elif exp_name.startswith('exp3'):
        task = 'rhythm'
    else:
        raise ValueError(f"Unknown task for experiment: {exp_name}")
        
    # Determine augment, filter_type, class_balancing
    augment = '_aug' in exp_name
    filter_type = 'bandpass' if '_filtered' in exp_name else None
    class_balancing = '_balanced' in exp_name
    
    return task, augment, filter_type, class_balancing

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Explain multiple CNN experiments")
    parser.add_argument('--sample_size', type=int, default=None,
                        help="Number of validation samples to use for gradient computation (default: None, which uses all validation samples)")
    args = parser.parse_args()
    
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'
    plots_dir = os.path.join(outputfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # List of experiments to compare (only complete pipeline experiments)
    experiments = [
        'exp0_filtered_aug_balanced',       # All (Optimized, 71 Classes)
        'exp1_filtered_aug_balanced',       # Diagnostic (Optimized, 44 Classes)
        'exp1.1_filtered_aug_balanced',     # Subdiagnostic (Optimized, 23 Classes)
        'exp1.1.1_filtered_aug_balanced',   # Superdiagnostic (Optimized, 5 Classes)
        'exp2_filtered_aug_balanced',       # Form (Optimized, 19 Classes)
        'exp3_filtered_aug_balanced'        # Rhythm (Optimized, 12 Classes)
    ]
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Dict to store universal importance vector for each experiment
    all_importances = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for exp_name in experiments:
        print(f"\n=============================================================")
        print(f"PROCESSING EXPERIMENT: {exp_name}")
        print(f"=============================================================")
        
        # Check checkpoint first
        model_checkpoint_path = os.path.join(outputfolder, exp_name, 'models/fastai_resnet1d18/models/fastai_resnet1d18.pth')
        if not os.path.exists(model_checkpoint_path):
            print(f"Warning: Checkpoint not found at {model_checkpoint_path}. Skipping.")
            continue
            
        task, augment, filter_type, class_balancing = parse_exp_settings(exp_name)
        print(f"Parsed settings - Task: {task}, Augment: {augment}, Filter: {filter_type}, Balancing: {class_balancing}")
        
        # Initialize experiment
        e = SCP_Experiment(
            experiment_name=exp_name,
            task=task,
            datafolder=datafolder,
            outputfolder=outputfolder,
            models=[conf_fastai_resnet1d18],
            augment=augment,
            filter_type=filter_type,
            class_balancing=class_balancing
        )
        
        print("Preparing data...")
        e.prepare()
        
        X_val = e.X_val
        y_val = e.y_val
        n_classes = e.n_classes
        
        # Load MLB
        mlb_path = os.path.join(outputfolder, exp_name, 'data/mlb.pkl')
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        classes = list(mlb.classes_)
        
        # Load Model
        print(f"Loading ResNet1d18 model from checkpoint...")
        model = resnet1d18(num_classes=n_classes, input_channels=12, inplanes=128, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        
        # Sampling
        if args.sample_size is not None and args.sample_size > 0:
            sample_size = min(len(X_val), args.sample_size)
            np.random.seed(42)
            indices = np.random.choice(len(X_val), sample_size, replace=False)
            print(f"Computing saliency maps on {sample_size} out of {len(X_val)} validation records...")
        else:
            sample_size = len(X_val)
            indices = np.arange(len(X_val))
            print(f"Computing saliency maps on all {sample_size} validation records...")
        
        X_val_torch = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32).to(device)
        
        class_gradients = {c: [] for c in range(n_classes)}
        
        for count, idx in enumerate(indices):
            if (count + 1) % 200 == 0 or count == 0:
                print(f"  Record {count + 1}/{sample_size}...")
                
            x_i = X_val_torch[idx:idx+1].clone()
            x_i.requires_grad = True
            
            logits = model(x_i)
            
            active_classes = np.where(y_val[idx] == 1.0)[0]
            for c in active_classes:
                model.zero_grad()
                if x_i.grad is not None:
                    x_i.grad.zero_()
                    
                logit = logits[0, c]
                logit.backward(retain_graph=True)
                
                grad = x_i.grad.detach().cpu().numpy()[0]
                abs_grad = np.abs(grad)
                mean_abs_grad = np.mean(abs_grad, axis=1) # (12,)
                
                class_gradients[c].append(mean_abs_grad)
                
        # Aggregate
        mean_class_importances = []
        for c in range(n_classes):
            if len(class_gradients[c]) > 0:
                mean_class_importances.append(np.mean(class_gradients[c], axis=0))
        
        # Universal importance for this experiment
        if len(mean_class_importances) > 0:
            universal_importance = np.mean(mean_class_importances, axis=0)
        else:
            universal_importance = np.zeros(12)
            
        all_importances[exp_name] = universal_importance
        print(f"Universal importance computed for {exp_name}.")
        
    if not all_importances:
        print("Error: No experiments were successfully evaluated.")
        sys.exit(1)
        
    # Combine into a DataFrame
    df_comparison = pd.DataFrame(all_importances, index=lead_names)
    
    # Normalize each column to sum to 1 to compare relative importance profiles
    df_normalized = df_comparison.div(df_comparison.sum(axis=0), axis=1)
    
    # Save results
    results_dir = os.path.join(outputfolder, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'lead_importance_comparison.csv')
    df_comparison.to_csv(csv_path)
    df_normalized.to_csv(os.path.join(results_dir, 'lead_importance_comparison_normalized.csv'))
    print(f"\nLead importance comparison saved to: {csv_path}")
    
    # Generate Heatmap
    print("\nGenerating comparison heatmap...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    # Plot normalized heatmap for relative profile comparison
    sns.heatmap(df_normalized, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={'label': 'Relative Saliency Fraction'})
    plt.title("ECG Lead Relative Importance Comparison across Tasks & Pipeline Passes", fontweight='bold', pad=15, fontsize=14)
    plt.ylabel("ECG Lead")
    plt.xlabel("Experiment Configuration")
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'lead_importance_comparison_heatmap.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {plot_path}")
    
    # Print ranked leads for each experiment
    print("\n=============================================================")
    print("RANKED LEADS BY EXPERIMENT CONFIGURATION")
    print("=============================================================")
    for exp_name in all_importances:
        sorted_series = df_comparison[exp_name].sort_values(ascending=False)
        ranks_str = ", ".join([f"{lead} ({val:.5f})" for lead, val in zip(sorted_series.index, sorted_series.values)])
        print(f"\n{exp_name}:")
        print(f"  Ranked: {ranks_str}")

if __name__ == '__main__':
    main()
