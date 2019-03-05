import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from frontend import YOLO
import json
from utils import draw_boxes
from PIL import Image
from datetime import datetime


def _main_():
    config_path = 'config/config3.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    yolo.model.load_weights(config['train']['pretrained_weights'])
    #########################

    ###############################
    #   Start the training process
    ###############################
    path = '/media/palm/data/Cloth/dataset/dataset/'
    # path = ''
    counter = 0
    filename = '001dxxyile2uxkblr99uqo6fuhgprpccznlze0z0djhs9gkek2tsm8u5hsfzx62o.jpg'
    # filename = 'download.jpeg'
    image = cv2.imread(path + filename)

    image = cv2.resize(image, (config['model']['input_size'], config['model']['input_size']))
    hour = datetime.now().hour
    minute = datetime.now().minute
    second = datetime.now().second
    ms = datetime.now().microsecond
    j = 0
    sums = 0

    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])
    im = Image.fromarray(image)
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    im.show()


if __name__ == '__main__':
    _main_()
