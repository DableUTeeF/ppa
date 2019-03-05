import tensorflow as tf
from keras import layers, models, applications


def SMobileNet(input_tensor):
    alpha = 1.0
    depth_multiplier = 1

    img_input = input_tensor
    x = _conv_block(img_input, 16, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=6)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=7)
    return tf.keras.models.Model(inputs=img_input, outputs=x)


def KSMobileNet(input_tensor):
    alpha = 1.0
    depth_multiplier = 1

    img_input = input_tensor
    x = k_conv_block(img_input, 16, alpha, strides=(2, 2))
    x = k_depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=1)

    x = k_depthwise_conv_block(x, 64, alpha, depth_multiplier,
                               strides=(2, 2), block_id=2)

    x = k_depthwise_conv_block(x, 128, alpha, depth_multiplier,
                               strides=(2, 2), block_id=3)

    x = k_depthwise_conv_block(x, 256, alpha, depth_multiplier,
                               strides=(2, 2), block_id=4)
    x = k_depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = k_depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=6)

    x = k_depthwise_conv_block(x, 512, alpha, depth_multiplier,
                               strides=(2, 2), block_id=7)
    return models.Model(inputs=img_input, outputs=x)


def MobileNet(input_tensor):
    alpha = 1.0
    depth_multiplier = 1

    img_input = input_tensor
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input

    # Create model.
    model = tf.keras.models.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, 416))

    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')
    if alpha == 1.0:
        alpha_text = '1_0'
    elif alpha == 0.75:
        alpha_text = '7_5'
    elif alpha == 0.50:
        alpha_text = '5_0'
    else:
        alpha_text = '2_5'

    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, 224)
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path = tf.keras.utils.get_file(model_name,
                                           weight_path,
                                           cache_subdir='models')
    model.load_weights(weights_path)
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1
    filters = int(filters * alpha)
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = tf.layers.Conv2D(filters, kernel,
                         padding='valid',
                         use_bias=False,
                         strides=strides,
                         name='conv1')(x)
    x = tf.layers.BatchNormalization(fused=True, axis=channel_axis, name='conv1_bn')(x)
    return tf.keras.layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)),
                                          name='conv_pad_%d' % block_id)(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False,
                                        name='conv_dw_%d' % block_id)(x)
    x = tf.layers.BatchNormalization(fused=True,
                                     axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = tf.layers.Conv2D(pointwise_conv_filters, (1, 1),
                         padding='same',
                         use_bias=False,
                         strides=(1, 1),
                         name='conv_pw_%d' % block_id)(x)
    x = tf.layers.BatchNormalization(axis=channel_axis,
                                     name='conv_pw_%d_bn' % block_id)(x)
    return tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


def k_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.Activation('relu', name='conv1_relu')(x)


def k_depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                           depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = applications.mobilenet.DepthwiseConv2D((3, 3),
                                               padding='same' if strides == (1, 1) else 'valid',
                                               depth_multiplier=depth_multiplier,
                                               strides=strides,
                                               use_bias=False,
                                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.Activation('relu', name='conv_pw_%d_relu' % block_id)(x)
