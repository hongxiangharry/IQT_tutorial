from .Kamnitsas import generate_kamnitsas_model
from .Dolz import generate_dolz_multi_model
from .Cicek import generate_unet_model as generate_unet_cicek_model
from .Guerrero import generate_uresnet_model
from .UnetContrast import generate_unet_model as generate_unet_contrast_model
from .UnetContrastHeteroTypeA import generate_hetero_unet_model as generate_hetero_unet_contrast_modelA
from .UnetContrastHeteroTypeB import generate_hetero_unet_model as generate_hetero_unet_contrast_modelB
from .UnetContrastHeteroTypeC import generate_hetero_unet_model as generate_hetero_unet_contrast_modelC

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'Kamnitsas' :
        return generate_kamnitsas_model(gen_conf, train_conf)
    if approach == 'DolzMulti' :
        return generate_dolz_multi_model(gen_conf, train_conf)
    if approach == 'Cicek' :
        return generate_unet_cicek_model(gen_conf, train_conf)
    if approach == 'Guerrero' :
        return generate_uresnet_model(gen_conf, train_conf)
    if approach == 'UnetContrast' :
        return generate_unet_contrast_model(gen_conf, train_conf)
    if approach == 'UnetContrastHeteroTypeA' :
        return generate_hetero_unet_contrast_modelA(gen_conf, train_conf)
    if approach == 'UnetContrastHeteroTypeB' :
        return generate_hetero_unet_contrast_modelB(gen_conf, train_conf)
    if approach == 'UnetContrastHeteroTypeC' :
        return generate_hetero_unet_contrast_modelC(gen_conf, train_conf)
    return None
