import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_resnet1d18

class PrunedSCP_Experiment(SCP_Experiment):
    def __init__(self, *args, lead_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lead_indices = lead_indices
        
    def prepare(self):
        super().prepare()
        if self.lead_indices is not None:
            print(f"Pruning: Slicing dataset to leads indices: {self.lead_indices}")
            self.X_train = self.X_train[:, :, self.lead_indices]
            self.X_val = self.X_val[:, :, self.lead_indices]
            self.X_test = self.X_test[:, :, self.lead_indices]
            self.input_shape = self.X_train[0].shape
            print(f"New dataset shape: train={self.X_train.shape}, val={self.X_val.shape}, test={self.X_test.shape}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pruning CNN experiment")
    parser.add_argument('--task', type=str, default='all', choices=['all', 'superdiagnostic'],
                        help="Task to run pruning for: 'all' (default, exp0) or 'superdiagnostic' (exp1.1.1)")
    args = parser.parse_args()
    target_task = args.task

    print(f"Setting up hard pruning experiment for task: {target_task}...")
    
    if target_task == 'all':
        base_exp_name = 'exp0_filtered_aug_balanced'
        scp_task = 'all'
    else:
        base_exp_name = 'exp1.1.1_filtered_aug_balanced'
        scp_task = 'superdiagnostic'
        
    lead_map = {'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
                'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11}
                
    # Load lead importance
    importance_csv = '../output/results/lead_importance_comparison.csv'
    individual_csv = f'../output/{base_exp_name}/results/cnn_lead_importance.csv'
    
    if os.path.exists(importance_csv):
        df_imp = pd.read_csv(importance_csv, index_col=0)
        if base_exp_name in df_imp.columns:
            ranked_leads = list(df_imp[base_exp_name].sort_values(ascending=False).index)
            print(f"Loaded ranked leads from combined CSV for {base_exp_name}.")
        else:
            print(f"Warning: Column '{base_exp_name}' not found in {importance_csv}. Trying individual CSV.")
            if os.path.exists(individual_csv):
                df_imp = pd.read_csv(individual_csv, index_col=0)
                ranked_leads = list(df_imp.index)
            else:
                print(f"Error: Individual CSV not found at {individual_csv}.")
                sys.exit(1)
    elif os.path.exists(individual_csv):
        df_imp = pd.read_csv(individual_csv, index_col=0)
        ranked_leads = list(df_imp.index)
        print(f"Loaded ranked leads from individual CSV for {base_exp_name}.")
    else:
        print(f"Error: Lead importance CSV not found. Please run explain_all_experiments.py first.")
        sys.exit(1)
        
    print(f"Ranked Leads (from most to least important): {ranked_leads}")
    ranked_indices = [lead_map[name] for name in ranked_leads]
    
    # We will test configurations of 1, 3, 4, and 6 leads
    K_values = [1, 3, 4, 6]
    
    # Store results for comparison
    results_list = []
    
    # Load 12-lead baseline results
    baseline_path = f'../output/{base_exp_name}/models/fastai_resnet1d18/results/te_results.csv'
    if os.path.exists(baseline_path):
        df_base = pd.read_csv(baseline_path, index_col=0)
        # Extract 'point' estimate metrics
        auc = df_base.loc['point', 'macro_auc']
        aupr = df_base.loc['point', 'macro_aupr']
        f1 = df_base.loc['point', 'F1']
        results_list.append({
            'Leads_Count': 12,
            'Leads_List': 'All (12 Leads)',
            'macro_auc': auc,
            'macro_aupr': aupr,
            'F1': f1
        })
        print(f"Loaded 12-lead baseline: AUC={auc:.4f}, AUPR={aupr:.4f}, F1={f1:.4f}")
    else:
        print(f"Warning: Baseline 12-lead results not found at {baseline_path}. Make sure the baseline has been run.")
        
    for K in K_values:
        K_leads = ranked_leads[:K]
        K_indices = ranked_indices[:K]
        
        print(f"\n=============================================================")
        print(f"TRAINING PRUNED CNN MODEL WITH K={K} LEADS: {K_leads}")
        print(f"=============================================================\n")
        
        pruned_exp_name = f"{base_exp_name}_pruned_{K}"
        
        # Instantiate and run experiment
        e = PrunedSCP_Experiment(
            experiment_name=pruned_exp_name,
            task=scp_task,
            datafolder='../data/ptbxl/',
            outputfolder='../output/',
            models=[conf_fastai_resnet1d18],
            augment=True,
            filter_type='bandpass',
            class_balancing=True,
            lead_indices=K_indices
        )
        
        print(f"Preparing pruned data for K={K}...")
        e.prepare()
        
        print(f"Training pruned CNN model for K={K}...")
        e.perform()
        
        print(f"Evaluating pruned CNN model for K={K}...")
        # Note: evaluate needs bootstraping parameters, we'll use defaults as defined in scp_experiment.py
        e.evaluate(n_bootstraping_samples=100, n_jobs=4, bootstrap_eval=False, dumped_bootstraps=False)
        
        # Load the test results
        te_res_path = f"../output/{pruned_exp_name}/models/fastai_resnet1d18/results/te_results.csv"
        if os.path.exists(te_res_path):
            df_k = pd.read_csv(te_res_path, index_col=0)
            auc_k = df_k.loc['point', 'macro_auc']
            aupr_k = df_k.loc['point', 'macro_aupr']
            f1_k = df_k.loc['point', 'F1']
            
            results_list.append({
                'Leads_Count': K,
                'Leads_List': str(K_leads),
                'macro_auc': auc_k,
                'macro_aupr': aupr_k,
                'F1': f1_k
            })
            print(f"Finished K={K}: AUC={auc_k:.4f}, AUPR={aupr_k:.4f}, F1={f1_k:.4f}")
        else:
            print(f"Error: Results file not found for K={K} at {te_res_path}")
            
    # Save the final summary table
    df_summary = pd.DataFrame(results_list)
    summary_path = f'../output/results_pruning_cnn_{target_task}.csv'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    
    print("\n=============================================================")
    print(f"PRUNING EXPERIMENT SUMMARY RESULTS ({target_task.upper()})")
    print("=============================================================")
    print(df_summary.to_string(index=False))
    print(f"Pruning comparison results saved to: {summary_path}")

if __name__ == '__main__':
    main()
