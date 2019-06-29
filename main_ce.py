'''
    main_ce.py: pipeline of IQT contrast enhancement (ce)

'''
from conf_ce import general_configuration as gen_conf
from conf_ce import training_configuration as train_conf
from workflow.data_preparation import data_preparation
from workflow.train import training
from workflow.test import testing
from workflow.evaluation import evaluation
import os

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# GPU configuration on the Miller/Armstrong cluster
is_cmic_cluster = True
gpu_no = opt['gpu']

if is_cmic_cluster == True:
    # GPUs devices:
    ## Marco "CUDA_VISIBLE_DEVICES" defines the working GPU in the CMIC clusters.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()  ## Check the GPU list.

# training process
model = training(gen_conf, train_conf)

# test process
testing(gen_conf, train_conf)

# evaluation
evaluation(gen_conf, train_conf)