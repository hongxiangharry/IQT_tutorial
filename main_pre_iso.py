'''
    main_test.py: Heterogeneous U-net for the IQT contrast enhancement
    todo:
    1. gen_conf, train_conf -> heterogeneous input
    2. read data
'''
from conf_pre_iso import general_configuration as gen_conf
from conf_pre_iso import training_configuration as train_conf
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

dataset_info['test_subjects'] = ['203418', '203721', '203923', '204016', '204218',
'204319', '204420', '204521', '204622', '205119', '205220', '205725', '205826', '206222', '206323']

# # GPU configuration on the Miller/Armstrong cluster
# is_processed = True
# is_cmic_cluster = True
# gpu_no = opt['gpu']
#
# if is_cmic_cluster == True:
#     # GPUs devices:
#     ## Marco "CUDA_VISIBLE_DEVICES" defines the working GPU in the CMIC clusters.
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
#     from tensorflow.python.client import device_lib
#     print device_lib.list_local_devices()  ## Check the GPU list.
#
# # data pre-processing
# if is_processed == False:
#     preproc_dataset(gen_conf, train_conf)
#
# # training process
# model, mean, std = training(gen_conf, train_conf)
#
# # test process
# testing(gen_conf, train_conf, mean, std)

# evaluation
evaluation(gen_conf, train_conf)
