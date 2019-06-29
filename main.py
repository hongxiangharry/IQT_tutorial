from configuration import general_configuration, training_configuration
from workflow.evaluate import run_evaluation_in_dataset
import os

is_cmic_cluster = True
gpu_no = '0'

if is_cmic_cluster == True:
    # GPUs devices:
    ## Marco "CUDA_VISIBLE_DEVICES" defines the working GPU in the CMIC clusters.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()  ## Check the GPU list.

run_evaluation_in_dataset(general_configuration, training_configuration)