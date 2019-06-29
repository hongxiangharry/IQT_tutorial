'''
    main_test.py: Heterogeneous U-net for the IQT contrast enhancement
    todo:
    1. gen_conf, train_conf -> heterogeneous input
    2. read data
'''
from conf_aunet import general_configuration as gen_conf
from conf_aunet import training_configuration as train_conf
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
# dataset_info['training_subjects'] = [
#     'sub-NDARAA075AMK', 'sub-NDARAA536PTU', 'sub-NDARAD481FXF',
#     'sub-NDARAE199TDD', 'sub-NDARAJ366ZFA', 'sub-NDARAR025WX4',
#     'sub-NDARAT100AEQ', 'sub-NDARAT299YRR', 'sub-NDARAV747WVL',
#     'sub-NDARAV894XWD', 'sub-NDARAV945MCQ', 'sub-NDARAY238DFL',
#     'sub-NDARBA507GCT', 'sub-NDARBA521RA8', 'sub-NDARBB854DRN'
# ]

dataset_info['test_subjects'] = ['203418', '203721', '203923', '204016', '204218']
# dataset_info['test_subjects'] = ['sub-NDARBC580HR5', 'sub-NDARBG702FED', 'sub-NDARBK669XJQ', 'sub-NDARBN100LCD', 'sub-NDARBU112XZE']

# GPU configuration on the Miller/Armstrong cluster
is_processed = False
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
