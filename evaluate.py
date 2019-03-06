import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from preprocessing import parse_annotation, BatchGenerator
from frontend import YOLO, normalize
import json


if __name__ == '__main__':

    config_path = 'config/config3.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8 * len(train_imgs))

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]
    np.random.shuffle(train_imgs)

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    yolo.load_weights(config['train']['pretrained_weights'])

    generator_config = {
        'IMAGE_H': config['model']['input_size'],
        'IMAGE_W': config['model']['input_size'],
        'GRID_H': config['model']['input_size']//32,
        'GRID_W': config['model']['input_size']//32,
        'BOX': len(config['model']['anchors']),
        'LABELS': config['model']['labels'],
        'CLASS': len(config['model']['labels']),
        'ANCHORS': config['model']['anchors'],
        'BATCH_SIZE': 1,
        'TRUE_BOX_BUFFER': config['model']['max_box_per_image'],
    }
    val_generator = BatchGenerator(valid_imgs,
                                   generator_config,
                                   norm=normalize,
                                   flipflop=False,
                                   shoechanger=False,
                                   zeropad=False,
                                   )
    yolo.evaluate(val_generator)
