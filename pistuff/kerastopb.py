import tensorflow as tf
from keras import backend as K, layers, models
import os
from backend import _conv_block, _depthwise_conv_block

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from frontend import NolambdaYOLO
import json


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def stupidmodel():
    alpha = 1.0
    depth_multiplier = 1

    inp = tf.keras.layers.Input((416, 416, 3))
    img_input = inp
    x = _conv_block(img_input, 16, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    return tf.keras.models.Model(inputs=inp, outputs=x)


def tiny():
    input_image = tf.keras.layers.Input(shape=(416, 416, 3))

    # Layer 1
    x = tf.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = tf.layers.BatchNormalization(fused=True, name='norm_1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Layer 2 - 5
    for i in range(0, 4):
        x = tf.layers.Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2),
                             use_bias=False)(x)
        x = tf.layers.BatchNormalization(fused=True, name='norm_' + str(i + 2))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Layer 6
    x = tf.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = tf.layers.BatchNormalization(fused=True, name='norm_6')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Layer 7 - 8
    for i in range(0, 2):
        x = tf.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 7), use_bias=False)(x)
        x = tf.layers.BatchNormalization(fused=True, name='norm_' + str(i + 7))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return tf.keras.models.Model(inputs=input_image, outputs=x)


def add_yololayer(model):
    inp = layers.Input((416, 416, 3))
    x = model(inp)
    x = layers.Conv2D(5 * (4 + 1 + 20),
                      (1, 1), strides=(1, 1),
                      padding='same',
                      name='DetectionLayer',
                      kernel_initializer='lecun_normal')(x)
    # x = layers.Reshape((13, 13, 5, 4 + 1 + 20))(x)
    return models.Model(inp, x)


if __name__ == '__main__':
    # Create, compile and train model...
    config_path = 'jsconfig.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo = NolambdaYOLO(backend=config['model']['backend'],
                        input_size=config['model']['input_size'],
                        labels=config['model']['labels'],
                        max_box_per_image=config['model']['max_box_per_image'],
                        anchors=config['model']['anchors'])
    yolo.model.load_weights(config['train']['pretrained_weights'])
    model = yolo.model
    model = stupidmodel()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

    # model.load_weights(config['train']['pretrained_weights'])
    # yolo.model.load_weights('tiny_yolo_cloth.h5')
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, "t", "sTFMNet_model.pbtxt", as_text=True)
        tf.train.write_graph(frozen_graph, "t", "sTFMNet_model.pb", as_text=False)

    # ms = [vgg16.VGG16, resnet50.ResNet50, mobilenet.MobileNet, densenet.DenseNet121]
    # mname = ['vgg16', 'resnet50', 'mobilenet', 'densenet121']
    # ms = [stupidmodel()]
    # mname = ['bn']
    # for i in range(len(ms)):
    #     # model = ms[i](include_top=False, weights=None)
    #     # model = add_yololayer(model)
    #     model = stupidmodel()
    #     frozen_graph = freeze_session(K.get_session(),
    #                                   output_names=[out.op.name for out in model.outputs])
    #     tf.train.write_graph(frozen_graph, "t", mname[i]+"_model.pbtxt", as_text=True)
    #     tf.train.write_graph(frozen_graph, "t", mname[i]+"_model.pb", as_text=False)
