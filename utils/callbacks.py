from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from utils.ioutils import generate_output_filename

import numpy as np
import os

# def generate_output_filename(
#     path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension) :
#     file_pattern = '{}/{}/{:02}-{}-{}-{}-{}.{}'
#     return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)

def generate_callbacks(general_configuration, training_configuration, case_name, mean = None, std = None) :
    ## save mean and std
    meanstd_filename = generate_output_filename(
        general_configuration['model_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        training_configuration['dimension'],
        str(training_configuration['patch_shape']),
        str(training_configuration['extraction_step'])+'_meanstd',
        'npz')
    ## check and make folders
        meanstd_foldername = os.path.dirname(meanstd_filename)
        if not os.path.isdir(meanstd_foldername) :
            os.makedirs(meanstd_foldername)
            
    if (mean is None) or (std is None):
        mean = {'input': 0, 'output': 0}
        std = {'input': 1, 'output': 1}
    np.savez(meanstd_filename, mean=mean, std=std)
    
    ## save model
    model_filename = generate_output_filename(
        general_configuration['model_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        training_configuration['dimension'],
        str(training_configuration['patch_shape']),
        str(training_configuration['extraction_step']),
        'h5')
    
    if (os.path.exists(model_filename) == False) or (training_configuration['retrain'] == True):
        ## check and make folders
        model_foldername = os.path.dirname(model_filename)
        if not os.path.isdir(model_foldername) :
            os.makedirs(model_foldername)

        csv_filename = generate_output_filename(
            general_configuration['log_path'],
            training_configuration['dataset'],
            case_name,
            training_configuration['approach'],
            training_configuration['dimension'],
            str(training_configuration['patch_shape']),
            str(training_configuration['extraction_step']),
            'cvs')
        ## check and make folders
        csv_foldername = os.path.dirname(csv_filename)
        if not os.path.isdir(csv_foldername) :
            os.makedirs(csv_foldername)

        stopper = EarlyStopping(
            patience=training_configuration['patience'])

        checkpointer = ModelCheckpoint(
            filepath=model_filename,
            verbose=0,
            save_best_only=True,
            save_weights_only=True)

        csv_logger = CSVLogger(csv_filename, separator=';')

        if training_configuration['optimizer'] == 'SGD' :
            def step_decay(epoch) :
                initial_lr = training_configuration['learning_rate']
                drop = training_configuration['decay']
                epochs_drop = 5.0
                lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
                return lr
            lr_scheduler = LearningRateScheduler(step_decay, verbose=1)
            return [stopper, checkpointer, csv_logger, lr_scheduler]
        else :
            return [stopper, checkpointer, csv_logger]
    else :
        return None
