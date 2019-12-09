"""
SPECIFY MODEL TYPE AND BACKBONE
INCLUDE ONLY, DO NOT EXECUTE
"""
from helpers import *
import segmentation_models as sm
import segmentation_models.backbones as smb
import keras


model_type = 'my_res_unet'
#model_type = 'unet'
#model_type = 'fpn'
#model_type = 'linknet'
#model_type = 'pspnet'


backbone = 'resnet34'


if model_type == 'my_res_unet':
    preprocessing = adjust_input
else:
    preprocessing = smb.get_preprocessing(backbone)


def create_model(double_size=True, slide_augmentation=True, trainable_encoder=True, n=32, dropout=0.2):
    if model_type == 'my_res_unet':
        model = my_res_unet(n=n, batch_norm=True, dropout=dropout, slide_augmentation=slide_augmentation)
    else:
        image_size = 256 if double_size else 128
        if model_type == 'unet':
            model = sm.Unet(backbone_name=backbone,
                            input_shape=(image_size, image_size, 3),
                            classes=1,
                            activation='sigmoid',
                            encoder_weights='imagenet',
                            encoder_freeze=not trainable_encoder,
                            encoder_features='default',
                            decoder_block_type='upsampling',
                            decoder_filters=(16*n, 8*n, 4*n, 2*n, n),
                            decoder_use_batchnorm=True)
        elif model_type == 'fpn':
            model = sm.FPN(backbone_name=backbone,
                           input_shape=(image_size, image_size, 3),
                           classes=1,
                           activation='sigmoid',
                           encoder_weights='imagenet',
                           encoder_freeze=not trainable_encoder,
                           encoder_features='default',
                           pyramid_block_filters=256,
                           pyramid_use_batchnorm=True,
                           pyramid_dropout=None,
                           final_interpolation='bilinear')
        elif model_type == 'linknet':
            model = sm.Linknet(backbone_name=backbone,
                               input_shape=(image_size, image_size, 3),
                               classes=1,
                               activation='sigmoid',
                               encoder_weights='imagenet',
                               encoder_freeze=not trainable_encoder,
                               encoder_features='default',
                               decoder_block_type='upsampling',
                               decoder_filters=(None, None, None, None, 16),
                               decoder_use_batchnorm=True)
        elif model_type == 'pspnet':
            image_size = 240 if double_size else 120
            model = sm.PSPNet(backbone_name=backbone,
                              input_shape=(image_size, image_size, 3),
                              classes=1,
                              activation='sigmoid',
                              encoder_weights='imagenet',
                              encoder_freeze=not trainable_encoder,
                              downsample_factor=8,
                              psp_conv_filters=512,
                              psp_pooling_type='avg',
                              psp_use_batchnorm=True,
                              psp_dropout=None,
                              final_interpolation='bilinear')
        else:
            print('Invalid segmentation model type')
            exit(0)

        if not slide_augmentation:
            x = keras.layers.Input(shape=(101, 101, 1), name='input')
            if model_type == 'pspnet':
                y = keras.layers.ZeroPadding2D(((9, 10), (9, 10)), name='zero_pad_input')(x)
            else:
                y = keras.layers.ZeroPadding2D(((13, 14), (13, 14)), name='zero_pad_input')(x)
            y = keras.layers.Cropping2D()(y)
        else:
            if model_type == 'pspnet':
                x = keras.layers.Input(shape=(120, 120, 1), name='input')
            else:
                x = keras.layers.Input(shape=(128, 128, 1), name='input')
            y = x
        if double_size:
            y = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(y)
        y = keras.layers.concatenate([y, y, y], name='channel_x3')
        y = model(y)
        if double_size:
            y = keras.layers.AvgPool2D(pool_size=(2, 2))(y)
        model = keras.models.Model(x, y)

    return model


########################################################################################################################
# MyResUNet
########################################################################################################################

def conv_block(n, no, x, batch_norm=True, debug_print=True):
    if debug_print:
        print('----------------------------------------------')
        print('Input shape:  ' + str(x.shape[1:]))
        print('Block type:   ' + str(n) + ' -> ' + str(no))

    c1 = keras.layers.Conv2D(n, (3, 3), padding='same')(x)
    if batch_norm:
        c1 = keras.layers.BatchNormalization()(c1)
    if debug_print:
        print("   c1")
    c1 = keras.layers.Activation(activation='relu')(c1)

    c2 = keras.layers.Conv2D(n, (3, 3), padding='same')(c1)
    if batch_norm:
        c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.add([c1, c2])
    if debug_print:
        print("   c1+c2")
    c2 = keras.layers.Activation(activation='relu')(c2)

    c3 = keras.layers.Conv2D(n, (3, 3), padding='same')(c2)
    if batch_norm:
        c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.add([c1, c2, c3])
    if debug_print:
        print("   c1+c2+c3")
    c3 = keras.layers.Activation(activation='relu')(c3)

    c4 = keras.layers.Conv2D(n, (3, 3), padding='same')(c3)
    if batch_norm:
        c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.add([c1, c2, c3, c4])
    if debug_print:
        print("   c1+c2+c3+c4")
    c4 = keras.layers.Activation(activation='relu')(c4)

    y = keras.layers.Conv2D(no, (3, 3), padding='same')(c4)
    if batch_norm:
        y = keras.layers.BatchNormalization()(y)
    if debug_print:
        print("   y")
    y = keras.layers.Activation(activation='relu')(y)

    if debug_print:
        print('Output shape: ' + str(y.shape[1:]))

    return y


def down_block(x, dropout=0.2, pool=True):

    y = x

    if pool:
        y = keras.layers.MaxPool2D((2, 2))(y)

    if dropout > 0.0:
        y = keras.layers.Dropout(dropout)(y)

    return y


def up_block(n, x, skip, layer='upsampling'):

    if layer == 'upsampling':
        y = keras.layers.UpSampling2D(size=(2, 2))(x)
    else:
        y = keras.layers.Conv2DTranspose(n, (2, 2), strides=(2, 2), padding='same')(x)

    y = keras.layers.concatenate([y, skip])

    return y


def my_res_unet(n=16, batch_norm=True, dropout=0.2, slide_augmentation=False):

    if not slide_augmentation:
        x = keras.layers.Input(shape=(101, 101, 1), name='input')
        p = keras.layers.ZeroPadding2D(((13, 14), (13, 14)), name='zero_pad_input')(x)
        p = keras.layers.Cropping2D()(p)
    else:
        x = keras.layers.Input(shape=(128, 128, 1), name='input')
        p = x

    c1 = conv_block(n, 2*n, x=p, batch_norm=batch_norm)
    p1 = down_block(c1, pool=True, dropout=dropout)

    c2 = conv_block(2*n, 4*n, x=p1, batch_norm=batch_norm)
    p2 = down_block(c2, pool=True, dropout=dropout)

    c3 = conv_block(4*n, 8*n, x=p2, batch_norm=batch_norm)
    p3 = down_block(c3, pool=True, dropout=dropout)

    c4 = conv_block(8*n, 16*n, x=p3, batch_norm=batch_norm)
    p4 = down_block(c4, pool=True, dropout=dropout)

    c5 = conv_block(16*n, 16*n, x=p4, batch_norm=batch_norm)
    p5 = down_block(c5, pool=False, dropout=dropout)

    u6 = up_block(16*n, x=p5, skip=c4)
    c6 = conv_block(16*n, 8*n, x=u6, batch_norm=batch_norm)

    u7 = up_block(8*n, x=c6, skip=c3)
    c7 = conv_block(8*n, 4*n, x=u7, batch_norm=batch_norm)

    u8 = up_block(4*n, x=c7, skip=c2)
    c8 = conv_block(4*n, 2*n, x=u8, batch_norm=batch_norm)

    u9 = up_block(2*n, x=c8, skip=c1)
    c9 = conv_block(2*n, n, x=u9, batch_norm=batch_norm)

    y = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    if not slide_augmentation:
        y = keras.layers.Cropping2D(((13, 14), (13, 14)))(y)

    return keras.models.Model(inputs=x, outputs=y)
