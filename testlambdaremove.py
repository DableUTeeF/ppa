import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from frontend import YOLO
import json

if __name__ == '__main__':
    config_path = 'config.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    yolo.model.save_weights('test.h5')


