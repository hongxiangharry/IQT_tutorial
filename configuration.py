general_configuration = {
    'num_classes' : 4,
    # 'dataset_path' : '/Users/hongxianglin/document/ucl_coding/mri_image/',
    'dataset_path' : '/scratch0/harrylin',
    'log_path' : '/home/harrylin/segmentation_data/log',
    'model_path' : '/home/harrylin/segmentation_data/models',
    # 'results_path' : '/Users/hongxianglin/document/ucl_coding/pfcnn_results/',
    'results_path' : '/home/harrylin/segmentation_data/result/',
    'dataset_info' : {
        'iSeg2017' : {
            'format' : 'analyze',
            'dimensions' : (144, 192, 256),
            'num_volumes' : 10,
            'modalities' : 2,
            'general_pattern' : 'subject-{}-{}.hdr',
            'path' : 'iSeg2017/iSeg-2017-Training/',
            'inputs' : ['T1', 'T2', 'label']
        },
        'IBSR18' : {
            'format' : 'nii',
            'dimensions' : (256, 128, 256),
            'num_volumes' : 10, # at least 3
            'modalities' : 1,
            'general_pattern' : 'IBSR_{0:02}/IBSR_{0:02}_{1}.nii.gz',
            'path' : 'IBSR18/',
            'inputs' : ['ana_strip', 'segTRI_ana']
        },
        'MICCAI2012' : {
            'format' : 'nii',
            'dimensions' : (256, 287, 256),
            'num_volumes' : [15, 20],
            'modalities' : 1,
            'general_pattern' : ['{}/{}_tmp.nii.gz', '{}/{}_3C_tmp.nii.gz', '{}/{}_{}.nii.gz'],
            'path' : 'MICCAI2012/',
            'folder_names' : ['training-images', 'training-labels', 'testing-images', 'testing-labels']
        },
        'HCP-Wu-Minn-Contrast': {
            'format' : 'nii',
            'dimensions': (256, 287, 256), # need to revise
            'num_volumes': [50, 10], # train and test
            'modalities': 1,
            'general_pattern': '{}/T1w/{}_acpc_dc_restore_brain{}.nii',
            'path': 't1t2image',
            'postfix': ['', ''],
            'modality_categories': ['T1w', 'T2w', 'FLAIR', 'T2starw']
        }
    }
}

training_configuration = {
    'activation' : 'softmax',
    'approach' : 'DolzMulti',
    'dataset' : 'IBSR18',
    'dimension' : 3,
    'extraction_step' : (3, 9, 3),
    'extraction_step_test' : (3, 3, 3),
    'loss' : 'categorical_crossentropy',
    'metrics' : ['acc'],
    'num_epochs' : 2,
    'optimizer' : 'Adam',
    'output_shape' : (9, 9, 9),
    'patch_shape' : (27, 27, 27),
    'bg_discard_percentage' : 0.2,
    'patience' : 1,
    'validation_split' : 0.20,
    'verbose' : 1,
}

## test_configuration
testing_configuration = {
    'activation' : 'softmax',
    'approach' : 'DolzMulti',
    'dataset' : 'IBSR18',
    'dimension' : 3,
    'extraction_step' : (3, 9, 3),
    'extraction_step_test' : (3, 3, 3),
    'loss' : 'categorical_crossentropy',
    'metrics' : ['acc'],
    'num_epochs' : 2,
    'optimizer' : 'Adam',
    'output_shape' : (9, 9, 9),
    'patch_shape' : (27, 27, 27),
    'bg_discard_percentage' : 0.2,
    'patience' : 1,
    'validation_split' : 0.20,
    'verbose' : 1,
}