import os
import argparse
import csv
from IPython.core.debugger import set_trace

def argument_parse():
    # -------------------------- Set up configurations ----------------------------
    # Basic settings
    ## description: text to display before the argument help
    parser = argparse.ArgumentParser(description='IQT-Keras-version')
    ## dest : The name of the attribute to be added to the object returned by parse_args()
    ## If there is no explicit written "dest" parameter, the key should be "e" in this case.

    # default: None for '?' and [] for '*'
    # list to tuple
    # system conf
    parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')

    ## directory
    parser.add_argument('-dp', '--dataset_path', dest='dataset_path', nargs='?', type=str, help='dataset directory')
    parser.add_argument('-bp', '--base_path', dest='base_path', nargs='?', type=str, help='workplace directory')
    parser.add_argument('-jn', '--job_name', dest='job_name', nargs='?', type=str, help='job name of folder')

    ## dataset info
    parser.add_argument('--dataset', dest='dataset', nargs='?', type=str, help='dataset name')
    parser.add_argument('--no_subject', dest='no_subject', nargs='*', type=int, help='set train/test subjects')
    # patching info
    parser.add_argument('-es', '--extraction_step', dest='extraction_step', nargs='*', type=int,
                        help='stride between patch for training')
    parser.add_argument('-est', '--extraction_step_test', dest='extraction_step_test', nargs='*', type=int,
                        help='stride between patch for testing')
    parser.add_argument('-ip', '--input_patch', dest='input_patch', nargs='*', type=int,
                        help='input patch shape')
    parser.add_argument('-op', '--output_patch', dest='output_patch', nargs='*', type=int,
                        help='output patch shape')

    # network info
    parser.add_argument('--approach', dest='approach', nargs='?', type=str, help='name of network architecture')
    parser.add_argument('-ne', '--no_epochs', dest='no_epochs', nargs='?', type=int, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, help='batch size')
    parser.add_argument('--patience', dest='patience', nargs=1, type=int, help='early stop at patience number')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', nargs='?', type=float, help='learning rate')
    parser.add_argument('-dc', '--decay', dest='decay', nargs='?', type=float, help='decay of learning rate')
    parser.add_argument('-dsf', '--downsize_factor', dest='downsize_factor', nargs='?', type=int, help='downsize factor for CNN')
    parser.add_argument('-nk', '--num_kernels', dest='num_kernels', nargs='?', type=int, help='number of kernels per block')
    parser.add_argument('-nf', '--num_filters', dest='num_filters', nargs='?', type=int,
                        help='number of filters per conv layer')
    parser.add_argument('-mt', '--mapping_times', dest='mapping_times', nargs='?', type=int,
                        help='number of FSRCNN shrinking layers')

    # # action : Turn on the value for the key, i.e. "overwrite=True"
    # parser.add_argument('--overwrite', action='store_true', help='restart the training completelu')
    # parser.add_argument('--continue', action='store_true', help='continue training from previous epoch')
    # parser.add_argument('--is_reset', action='store_true', help='reset the patch library?')
    # parser.add_argument('--not_save', action='store_true', help='invoke if you do not want to save the output')
    # parser.add_argument('--disp', action='store_true', help='save the displayed outputs?')
    #
    # # Directories:
    # parser.add_argument('--base_dir', type=str, default='/home/harrylin/experiments', help='base directory')
    # parser.add_argument('--gt_dir', type=str, default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc',
    #                     help='ground truth directory')
    # parser.add_argument('--subpath', type=str, default='T1w/Diffusion', help='subdirectory in gt_dir')
    # parser.add_argument('--mask_dir', type=str, default='/SAN/vision/hcp/Ryu/miccai2017/hcp_masks',
    #                     help='directory of segmentation masks')
    # parser.add_argument('--mask_subpath', type=str, default='', help='subdirectory in mask_dir')
    #
    # # Network
    # parser.add_argument('-m', '--method', dest='method', type=str, default='espcn', help='network type')
    # parser.add_argument('--no_filters', type=int, default=50, help='number of initial filters')
    # parser.add_argument('--no_layers', type=int, default=2, help='number of hidden layers')
    # parser.add_argument('--is_shuffle', action='store_true',
    #                     help='Needed for ESPCN/DCESPCN. Want to reverse shuffle the HR output into LR space?')
    # parser.add_argument('--is_BN', action='store_true', help='want to use batch normalisation?')
    # parser.add_argument('--optimizer', type=str, default='adam', help='optimization method')
    # parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default='1e-3',
    #                     help='learning rate')
    # parser.add_argument('-dr', '--dropout_rate', dest='dropout_rate', type=float, default='0.0', help='drop-out rate')
    # parser.add_argument('--no_epochs', type=int, default=200, help='number of epochs to train for')
    # parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    # parser.add_argument('--validation_fraction', type=float, default=0.5, help='fraction of validation data')
    # parser.add_argument('--valid', action='store_true', help='pick the best model based on the loss, not the MSE?')
    #
    # # Data/task
    # parser.add_argument('--is_map', action='store_true', help='Want to use MAP-MRI instead?')
    # parser.add_argument('-pl', '--no_patches', dest='no_patches', type=int, default=2250,
    #                     help='number of patches sampled from each train subject')
    # parser.add_argument('-ts', '--no_subjects', dest="no_subjects", type=int, default='8', help='background value')
    # parser.add_argument('--no_channels', type=int, default=6, help='number of channels')
    # parser.add_argument('-bgval', '--background_value', dest="background_value", type=float, default='0',
    #                     help='background value')
    # parser.add_argument('-us', '--upsampling_rate', dest="upsampling_rate", type=int, default=2, help='upsampling rate')
    # parser.add_argument('-ir', '--input_radius', dest="input_radius", type=int, default=5, help='input radius')
    # parser.add_argument('--pad_size', type=int, default=-1,
    #                     help='size of padding applied before patch extraction. Set -1 to apply maximal padding.')
    # parser.add_argument('--is_clip', action='store_true',
    #                     help='want to clip the images (0.1% - 99.9% percentile) before patch extraction? ')
    # parser.add_argument('--patch_sampling_opt', type=str, default='default',
    #                     help='sampling scheme for patch extraction')
    # parser.add_argument('--transform_opt', type=str, default='standard', help='normalisation transform')
    # parser.add_argument('-pp', '--postprocess', dest='postprocess', action='store_true',
    #                     help='post-process the estimated highres output?')

    arg = parser.parse_args()
    return vars(arg)  ## return a dictionary type of arguments and the values.

def set_conf_info(gen_conf, train_conf):
    opt = argument_parse() # read parser from the command line

    if opt['dataset_path'] is not None: gen_conf['dataset_path'] = opt['dataset_path']
    if opt['base_path'] is not None: gen_conf['base_path'] = opt['base_path']
    if opt['job_name'] is not None: gen_conf['job_name'] = opt['job_name']

    if opt['dataset'] is not None: train_conf['dataset'] = opt['dataset']
    if opt['no_subject'] is not None: gen_conf['dataset_info'][train_conf['dataset']]['num_volumes'] = opt['no_subject']

    if opt['extraction_step'] is not None: train_conf['extraction_step'] = tuple(opt['extraction_step'])
    if opt['extraction_step_test'] is not None: train_conf['extraction_step_test'] = tuple(opt['extraction_step_test'])
    if opt['input_patch'] is not None: train_conf['patch_shape'] = tuple(opt['input_patch'])
    if opt['output_patch'] is not None: train_conf['output_patch'] = tuple(opt['output_patch'])

    if opt['approach'] is not None: train_conf['approach'] = opt['approach']
    if opt['no_epochs'] is not None: train_conf['num_epochs'] = opt['no_epochs']
    if opt['batch_size'] is not None: train_conf['batch_size'] = opt['batch_size']
    if opt['patience'] is not None: train_conf['patience'] = opt['patience']

    if opt['learning_rate'] is not None: train_conf['learning_rate'] = opt['learning_rate']
    if opt['decay'] is not None: train_conf['decay'] = opt['decay']
    if opt['downsize_factor'] is not None: train_conf['downsize_factor'] = opt['downsize_factor']
    if opt['num_kernels'] is not None: train_conf['num_kernels'] = opt['num_kernels']
    if opt['num_filters'] is not None: train_conf['num_filters'] = opt['num_filters']
    if opt['mapping_times'] is not None: train_conf['mapping_times'] = opt['mapping_times']
    return opt, gen_conf, train_conf

def conf_dataset(gen_conf, train_conf):
    # configure log/model/result/evaluation paths.
    gen_conf['log_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['log_path'])
    gen_conf['model_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['model_path'])
    gen_conf['results_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['results_path'])
    gen_conf['evaluation_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['evaluation_path'])

    dataset = train_conf['dataset']

    if dataset == 'HBN':
        return conf_HBN_dataset(gen_conf, train_conf)
    if dataset == 'HCP-Wu-Minn-Contrast' :
        return conf_HCPWuMinnContrast_dataset(gen_conf, train_conf)
    if dataset == 'IBSR' :
        return conf_IBSR_dataset(gen_conf, train_conf)

def conf_IBSR_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = os.listdir(hcp_dataset_path)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf


def conf_HBN_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = os.listdir(hcp_dataset_path)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def conf_HCPWuMinnContrast_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = os.listdir(hcp_dataset_path)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def save_conf_info(gen_conf, train_conf):
    dataset_name = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset_name]

    # check and create parent folder
    csv_filename_gen = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        'gen_conf',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    csv_filename_train = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        'train_conf',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    csv_filename_dataset = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        'dataset',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    ## check and make folders
    csv_foldername = os.path.dirname(csv_filename_gen)
    if not os.path.isdir(csv_foldername) :
        os.makedirs(csv_foldername)

    # save gen_conf
    with open(csv_filename_gen, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, gen_conf.keys())
        w.writeheader()
        w.writerow(gen_conf)

    with open(csv_filename_train, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, train_conf.keys())
        w.writeheader()
        w.writerow(train_conf)

    with open(csv_filename_dataset, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dataset_info.keys())
        w.writeheader()
        w.writerow(dataset_info)

def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)
