'''
    main_test.py: Heterogeneous U-net for the IQT contrast enhancement
    todo:
    1. gen_conf, train_conf -> heterogeneous input
    2. read data
'''
from config_iso_IBADAN import general_configuration as gen_conf
from config_iso_IBADAN import training_configuration as train_conf
from config_iso_IBADAN import testing_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.train import training
from workflow.test import testing
from workflow.evaluation import evaluation
from utils.preprocessing_util import preproc_dataset
import os

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# default dataset
dataset_name = train_conf['dataset']
dataset_info = gen_conf['dataset_info'][dataset_name]
dataset_info['training_subjects'] = [
 '200008', '200109', '200210', '200311', '200513', '200614', '200917', '201111',
 '201414', '201515', '201717', '201818', '202113', '202719', '202820']

# 6, 2, 53-61
# gen_conf['dataset_info']['HCP-Wu-Minn-Contrast']['postfix'] = ['_sim036T_ds6_gap2_GM53_WM61_pre', '_sim036T_ds6_gap2_groundtruth', '_sim036T_ds6_gap2_GM53_WM61', '.nii_sim036T_GM50_WM63']
# 6, 2, 40-45
# gen_conf['dataset_info']['HCP-Wu-Minn-Contrast']['postfix'] = ['_sim036T_ds6_gap2_GM40_WM45_pre', '_sim036T_ds6_gap2_groundtruth', '_sim036T_ds6_gap2_GM40_WM45', '.nii_sim036T_GM50_WM63']
# 3, 1, 53-61
# gen_conf['dataset_info']['HCP-Wu-Minn-Contrast']['postfix'] = ['_sim036T_ds3_gap1_GM53_WM61_pre', '_sim036T_ds3_gap1_groundtruth', '_sim036T_ds3_gap1_GM53_WM61', '.nii_sim036T_GM50_WM63']
# 3, 1, 40-45
# gen_conf['dataset_info']['HCP-Wu-Minn-Contrast']['postfix'] = ['_sim036T_ds3_gap1_GM40_WM45_pre', '_sim036T_ds3_gap1_groundtruth', '_sim036T_ds3_gap1_GM40_WM45', '.nii_sim036T_GM50_WM63']

# GPU configuration on the Miller/Armstrong cluster
is_processed = True
is_cmic_cluster = True
gpu_no = opt['gpu']

if is_cmic_cluster == True:
    # GPUs devices:
    ## Marco "CUDA_VISIBLE_DEVICES" defines the working GPU in the CMIC clusters.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()  ## Check the GPU list.

# data pre-processing
if is_processed == False:
    preproc_dataset(gen_conf, train_conf)

# training process
model, mean, std = training(gen_conf, train_conf)


# mean = {'input': [0], 'output': [0]}
# std = {'input': [1], 'output': [1]}

dataset_name = test_conf['dataset']
dataset_info = gen_conf['dataset_info'][dataset_name]
# dataset_info['test_subjects'] = ['PA010']
# dataset_info['test_subjects'] = ['PA006']
dataset_info['test_subjects'] = ['PA004']
# test process
testing(gen_conf, test_conf, mean, std, train_conf)

