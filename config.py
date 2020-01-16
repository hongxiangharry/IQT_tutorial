general_configuration = {
    'num_classes' : 3, # label classes
    'dataset_path' : '/home/moucheng/harry/data/to_georgia', # PC
    'base_path' : '/home/moucheng/harry/data/tutorial_result/', # PC
    'job_name' : 'srunet16_16_2_nf4', # 'srunet16_16_2_nf4' or 'anisounet16_16_2_nf4'
    'log_path' : 'log',
    'model_path' : 'models',
    'results_path' : 'result',
    'evaluation_path': 'evaluation',
    'dataset_info' : {
        'HCP-Wu-Minn-Contrast': {
            'format' : 'nii',
            'dimensions': (260, 311, 260), # output shape
            'num_volumes': [0, 5], # train and test
            'modalities': 1,
            'general_pattern': '{}/{}_acpc_dc_restore_brain{}.nii',
            'path': 'HCP',
            'postfix_category': {'cnn_input': 0, 'cnn_gt': 1, 'preproc_out': 2, 'preproc_in': 3},
            'postfix': ['_procin', '', '_procin', ''],
            'modality_categories': ['T1w', 'T2w', 'FLAIR', 'T2starw'],
            'downsample_scale' : 8,
            'sparse_scale' : [1, 1, 8],
            'shrink_dim' : 3,
            'is_preproc': False, # if pre-processing input?
            'upsample_scale': [1, 1, 8],
            'interp_order' : 3 # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'SRUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'HCP-Wu-Minn-Contrast',
    'dimension' : 3,
    'extraction_step' : (16, 16, 2),
    'extraction_step_test' :(16, 16, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 200,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 4),
    'bg_discard_percentage' : 0.2,
    'patience' : 5,
    'validation_split' : 0.20,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'ishomo': False
}
