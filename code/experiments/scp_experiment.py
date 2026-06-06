from utils import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat'):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency
        self.zero_shot = (self.experiment_name == 'exp_ICBEB')
        self.zero_shot_ref_exp = 'exp0'

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def prepare(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        if self.zero_shot:
            scaler_path = self.outputfolder + self.zero_shot_ref_exp + '/data/standard_scaler.pkl'
            with open(scaler_path, 'rb') as ss_file:
                ss = pickle.load(ss_file)
            self.X_train = utils.apply_standardizer(self.X_train, ss)
            self.X_val = utils.apply_standardizer(self.X_val, ss)
            self.X_test = utils.apply_standardizer(self.X_test, ss)
        else:
            self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')

    def perform(self):
        if self.zero_shot:
            # Load MLB from ref_exp to get class names
            with open(self.outputfolder + self.zero_shot_ref_exp + '/data/mlb.pkl', 'rb') as f:
                ref_mlb = pickle.load(f)
            ref_classes = list(ref_mlb.classes_)
            ref_n_classes = len(ref_classes)

            # Load MLB from current exp to get class names
            with open(self.outputfolder + self.experiment_name + '/data/mlb.pkl', 'rb') as f:
                current_mlb = pickle.load(f)
            current_classes = list(current_mlb.classes_)

            # Build class mapping
            mapping = []
            for cls in current_classes:
                query_cls = 'PVC' if cls == 'VPC' else cls
                if query_cls in ref_classes:
                    mapping.append(ref_classes.index(query_cls))
                else:
                    mapping.append(-1)
        else:
            ref_n_classes = self.Y.shape[1]

        for model_description in self.models:
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            # For zero-shot, load from the reference experiment
            if self.zero_shot:
                ref_mpath = self.outputfolder + self.zero_shot_ref_exp + '/models/' + modelname + '/'
                model_dir = ref_mpath
            else:
                model_dir = mpath

            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, ref_n_classes, self.sampling_frequency, model_dir, self.input_shape, **modelparams)
            elif modeltype == 'FFT':
                from models.fft_model import FFTModel
                model = FFTModel(modelname, ref_n_classes, self.sampling_frequency, model_dir, self.input_shape, **modelparams)
            elif modeltype == 'RAW_STATS':
                from models.raw_model import RawModel
                model = RawModel(modelname, ref_n_classes, self.sampling_frequency, model_dir, self.input_shape, **modelparams)
            elif modeltype == 'PATCHTST':
                from models.patchtst_model import PatchTSTModel
                model = PatchTSTModel(modelname, ref_n_classes, self.sampling_frequency, model_dir, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, ref_n_classes, self.sampling_frequency, model_dir, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            if self.zero_shot:
                print(f"Zero-shot evaluation of {modelname} (no training on {self.experiment_name})")
                
                # Predict on the current experiment datasets (returns shape (N, 71))
                preds_train_raw = model.predict(self.X_train)
                preds_val_raw = model.predict(self.X_val)
                preds_test_raw = model.predict(self.X_test)

                # Map 71 classes predictions to 9 classes predictions
                preds_train = np.zeros((len(preds_train_raw), len(current_classes)))
                preds_val = np.zeros((len(preds_val_raw), len(current_classes)))
                preds_test = np.zeros((len(preds_test_raw), len(current_classes)))

                for i, ref_idx in enumerate(mapping):
                    if ref_idx != -1:
                        preds_train[:, i] = preds_train_raw[:, ref_idx]
                        preds_val[:, i] = preds_val_raw[:, ref_idx]
                        preds_test[:, i] = preds_test_raw[:, ref_idx]

                # Dump predictions
                preds_train.dump(mpath+'y_train_pred.npy')
                preds_val.dump(mpath+'y_val_pred.npy')
                preds_test.dump(mpath+'y_test_pred.npy')
            else:
                # fit model
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
                # predict and dump
                model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
                model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
                model.predict(self.X_test).dump(mpath+'y_test_pred.npy')

        # (Ensemble model generation removed as requested)

    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):

        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        #y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                #train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                #val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            #train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            #val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        #train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        #val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            mpath = self.outputfolder+self.experiment_name+'/models/'+m+'/'
            rpath = self.outputfolder+self.experiment_name+'/models/'+m+'/results/'

            # skip if prediction files are missing
            if not os.path.exists(mpath+'y_train_pred.npy') or not os.path.exists(mpath+'y_test_pred.npy'):
                continue

            print(m)
            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            #y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([
                    te_df_point.mean().values, 
                    te_df.mean().values,
                    te_df.quantile(0.05).values,
                    te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')
