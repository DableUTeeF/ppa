import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import numpy as np
import json
from frontend import create_yolov3_model, dummy_loss, mnet_yolov3_model, rnet50_yolov3_model
from preprocessing import Y3BatchGenerator, parse_annotation
from utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import AdamW, SGDW
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras


def create_training_instances(train_annot_folder,
                              train_image_folder,
                              valid_annot_folder,
                              valid_image_folder,
                              labels,
                              ):
    # parse annotations of the training set
    train_ints, train_labels = parse_annotation(train_annot_folder, train_image_folder, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8 * len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    logdir = tensorboard_logs + os.path.split(saved_weights_name)[-1][:-3] + ' ' + str(datetime.now())

    tensorboard = CustomTensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=True,
    )
    return [checkpoint, reduce_on_plateau, tensorboard]


def create_model(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid, batch_size,
        warmup_batches,
        ignore_thresh,
        multi_gpu,
        saved_weights_name,
        lr,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
):
    template_model, infer_model = create_yolov3_model(
            nb_class=nb_class,
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=max_grid,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            ignore_thresh=ignore_thresh,
            grid_scales=grid_scales,
            obj_scale=obj_scale,
            noobj_scale=noobj_scale,
            xywh_scale=xywh_scale,
            class_scale=class_scale
        )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights('backend.h5', by_name=True)

    train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    # optimizer = SGDW(lr=lr, momentum=0.9)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


if __name__ == '__main__':

    config_path = 'config/config8.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators
    ###############################
    train_generator = Y3BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = Y3BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Create the model
    ###############################
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
    )
    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=2,
        max_queue_size=8,
        validation_data=valid_generator,
        use_multiprocessing=False,
    )

    # make a GPU version of infer_model for evaluation
    infer_model.save_weights(config['train']['saved_weights_name']+'temp')
    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
