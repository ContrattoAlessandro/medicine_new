import os
import sys

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.raw_configs import *
from configs.patchtst_configs import *



def main():
    print("Starting reproduce_results.py...", flush=True)
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    models = [
        conf_fastai_resnet1d18,
        conf_wavelet_standard_nn,
        conf_raw_standard_nn,
        conf_patchtst_standard,
        ]

    ##########################################
    # PASS 1: WITHOUT DATA AUGMENTATION
    ##########################################
    print("\n==========================================", flush=True)
    print("RUNNING PIPELINE WITHOUT DATA AUGMENTATION", flush=True)
    print("==========================================\n", flush=True)

    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
    ]

    for name, task in experiments:
        print(f"\n--- Running experiment: {name} (task: {task}) ---", flush=True)
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, augment=False)
        print("Preparing data...", flush=True)
        e.prepare()
        print("Performing training and prediction...", flush=True)
        e.perform()
        print("Evaluating results...", flush=True)
        e.evaluate()

    # generate great summary table
    print("\nGenerating PTB-XL summary table...", flush=True)
    utils.generate_ptbxl_summary_table(suffix='')

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    print("\n--- Running experiment on ICBEB data ---", flush=True)
    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models, augment=False)
    print("Preparing data...", flush=True)
    e.prepare()
    print("Performing training and prediction...", flush=True)
    e.perform()
    print("Evaluating results...", flush=True)
    e.evaluate()

    # generate great summary table
    print("\nGenerating ICBEB summary table...", flush=True)
    utils.ICBEBE_table(suffix='')

    ##########################################
    # PASS 2: WITH DATA AUGMENTATION
    ##########################################
    print("\n==========================================", flush=True)
    print("RUNNING PIPELINE WITH DATA AUGMENTATION", flush=True)
    print("==========================================\n", flush=True)

    experiments_aug = [
        ('exp0_aug', 'all'),
        ('exp1_aug', 'diagnostic'),
        ('exp1.1_aug', 'subdiagnostic'),
        ('exp1.1.1_aug', 'superdiagnostic'),
        ('exp2_aug', 'form'),
        ('exp3_aug', 'rhythm')
    ]

    for name, task in experiments_aug:
        print(f"\n--- Running experiment: {name} (task: {task}) ---", flush=True)
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, augment=True)
        print("Preparing data...", flush=True)
        e.prepare()
        print("Performing training and prediction...", flush=True)
        e.perform()
        print("Evaluating results...", flush=True)
        e.evaluate()

    # generate great summary table
    print("\nGenerating PTB-XL summary table with data augmentation...", flush=True)
    utils.generate_ptbxl_summary_table(suffix='_aug')

    ##########################################
    # EXPERIMENT BASED ICBEB DATA WITH DATA AUGMENTATION
    ##########################################

    print("\n--- Running experiment on ICBEB data with data augmentation ---", flush=True)
    e = SCP_Experiment('exp_ICBEB_aug', 'all', datafolder_icbeb, outputfolder, models, augment=True)
    print("Preparing data...", flush=True)
    e.prepare()
    print("Performing training and prediction...", flush=True)
    e.perform()
    print("Evaluating results...", flush=True)
    e.evaluate()

    # generate great summary table
    print("\nGenerating ICBEB summary table with data augmentation...", flush=True)
    utils.ICBEBE_table(suffix='_aug')

    ##########################################
    # PASS 3: WITH FILTERING AND DATA AUGMENTATION
    ##########################################
    print("\n==========================================", flush=True)
    print("RUNNING PIPELINE WITH FILTERING AND DATA AUGMENTATION", flush=True)
    print("==========================================\n", flush=True)

    experiments_filtered_aug = [
        ('exp0_filtered_aug', 'all'),
        ('exp1_filtered_aug', 'diagnostic'),
        ('exp1.1_filtered_aug', 'subdiagnostic'),
        ('exp1.1.1_filtered_aug', 'superdiagnostic'),
        ('exp2_filtered_aug', 'form'),
        ('exp3_filtered_aug', 'rhythm')
    ]

    for name, task in experiments_filtered_aug:
        print(f"\n--- Running experiment: {name} (task: {task}) ---", flush=True)
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, augment=True, filter_type='bandpass')
        print("Preparing data...", flush=True)
        e.prepare()
        print("Performing training and prediction...", flush=True)
        e.perform()
        print("Evaluating results...", flush=True)
        e.evaluate()

    # generate great summary table
    print("\nGenerating PTB-XL summary table with filtering and data augmentation...", flush=True)
    utils.generate_ptbxl_summary_table(suffix='_filtered_aug')

    ##########################################
    # EXPERIMENT BASED ICBEB DATA WITH FILTERING AND DATA AUGMENTATION
    ##########################################

    print("\n--- Running experiment on ICBEB data with filtering and data augmentation ---", flush=True)
    e = SCP_Experiment('exp_ICBEB_filtered_aug', 'all', datafolder_icbeb, outputfolder, models, augment=True, filter_type='bandpass')
    print("Preparing data...", flush=True)
    e.prepare()
    print("Performing training and prediction...", flush=True)
    e.perform()
    print("Evaluating results...", flush=True)
    e.evaluate()

    # generate great summary table
    print("\nGenerating ICBEB summary table with filtering and data augmentation...", flush=True)
    utils.ICBEBE_table(suffix='_filtered_aug')

    ##########################################
    # PASS 4: WITH FILTERING, DATA AUGMENTATION, AND CLASS BALANCING
    ##########################################
    print("\n==========================================", flush=True)
    print("RUNNING PIPELINE WITH FILTERING, DATA AUGMENTATION, AND CLASS BALANCING", flush=True)
    print("==========================================\n", flush=True)

    experiments_filtered_aug_balanced = [
        ('exp0_filtered_aug_balanced', 'all'),
        ('exp1_filtered_aug_balanced', 'diagnostic'),
        ('exp1.1_filtered_aug_balanced', 'subdiagnostic'),
        ('exp1.1.1_filtered_aug_balanced', 'superdiagnostic'),
        ('exp2_filtered_aug_balanced', 'form'),
        ('exp3_filtered_aug_balanced', 'rhythm')
    ]

    for name, task in experiments_filtered_aug_balanced:
        print(f"\n--- Running experiment: {name} (task: {task}) ---", flush=True)
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, augment=True, filter_type='bandpass', class_balancing=True)
        print("Preparing data...", flush=True)
        e.prepare()
        print("Performing training and prediction...", flush=True)
        e.perform()
        print("Evaluating results...", flush=True)
        e.evaluate()

    # generate great summary table
    print("\nGenerating PTB-XL summary table with filtering, data augmentation, and class balancing...", flush=True)
    utils.generate_ptbxl_summary_table(suffix='_filtered_aug_balanced')

    ##########################################
    # EXPERIMENT BASED ICBEB DATA WITH FILTERING, DATA AUGMENTATION, AND CLASS BALANCING
    ##########################################

    print("\n--- Running experiment on ICBEB data with filtering, data augmentation, and class balancing ---", flush=True)
    e = SCP_Experiment('exp_ICBEB_filtered_aug_balanced', 'all', datafolder_icbeb, outputfolder, models, augment=True, filter_type='bandpass', class_balancing=True)
    print("Preparing data...", flush=True)
    e.prepare()
    print("Performing training and prediction...", flush=True)
    e.perform()
    print("Evaluating results...", flush=True)
    e.evaluate()

    # generate great summary table
    print("\nGenerating ICBEB summary table with filtering, data augmentation, and class balancing...", flush=True)
    utils.ICBEBE_table(suffix='_filtered_aug_balanced')

if __name__ == "__main__":
    main()
