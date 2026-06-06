import os
import sys

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.fft_configs import *
from configs.raw_configs import *
from configs.patchtst_configs import *



def main():
    print("Starting reproduce_results.py...", flush=True)
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    models = [
        #conf_fastai_resnet1d18,
        #conf_fastai_resnet1d_wang,
        #conf_wavelet_standard_nn,
        #conf_fft_standard_nn,
        #conf_raw_standard_nn,
        conf_patchtst_standard,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0', 'all')#,
        #('exp1', 'diagnostic'),
        #('exp1.1', 'subdiagnostic'),
        #('exp1.1.1', 'superdiagnostic'),
        #('exp2', 'form'),
        #('exp3', 'rhythm')
       ]

    for name, task in experiments:
        print(f"\n--- Running experiment: {name} (task: {task}) ---", flush=True)
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        print("Preparing data...", flush=True)
        e.prepare()
        print("Performing training and prediction...", flush=True)
        e.perform()
        print("Evaluating results...", flush=True)
        e.evaluate()

    # generate great summary table
    print("\nGenerating PTB-XL summary table...", flush=True)
    utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    print("\n--- Running experiment on ICBEB data ---", flush=True)
    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    print("Preparing data...", flush=True)
    e.prepare()
    print("Performing training and prediction...", flush=True)
    e.perform()
    print("Evaluating results...", flush=True)
    e.evaluate()

    # generate great summary table
    print("\nGenerating ICBEB summary table...", flush=True)
    utils.ICBEBE_table()

if __name__ == "__main__":
    main()
