from .IsoUnet import generate_iso_unet_model
from .AnisoUnetOld import generate_aniso_unet_old_model
from .SRUnet import generate_srunet_model
from .AnisoUnet import generate_aniso_unet_model

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'IsoUnet' :
        return generate_iso_unet_model(gen_conf, train_conf)
    if approach == 'AnisoUnetOld' :
        return generate_hetero_unet_contrast_modelA(gen_conf, train_conf)
    if approach == 'SRUnet' :
        return generate_srunet_model(gen_conf, train_conf)
    if approach == 'AnisoUnet' :
        return generate_aniso_unet_model(gen_conf, train_conf)
    return None
