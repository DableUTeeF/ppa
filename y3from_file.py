import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from frontend import mnet_yolov3_model, rnet50_yolov3_model, create_yolov3_model
import json
from utils import draw_boxesv3, preprocess_input, decode_netoutv3, do_nms, correct_yolo_boxes
from PIL import Image
import numpy as np
from preprocessing import minmaxresize

if __name__ == '__main__':

    config_path = 'config/config8.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    _, infer_model = rnet50_yolov3_model(
        nb_class=len(config['model']['labels']),
        anchors=config['model']['anchors'],
        max_box_per_image=30,
        max_grid=(config['model']['max_input_size'], config['model']['max_input_size']),
        batch_size=1,
        warmup_batches=0,
        ignore_thresh=config['train']['ignore_thresh'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale']
    )
    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    infer_model.load_weights(config['train']['saved_weights_name'])
    #########################

    ###############################
    #   Start the training process
    ###############################
    path = '/media/palm/data/ppa/v3/images/val/'
    pad = 0
    for _ in range(5):
    # if 1:

        # path = ''
        counter = 0
        filename = '001dxxyile2uxkblr99uqo6fuhgprpccznlze0z0djhs9gkek2tsm8u5hsfzx62o.jpg'
        # filename = 'download.jpeg'
        p = os.path.join(path, np.random.choice(os.listdir(path)))
        image = cv2.imread(p)
        image, w, h = minmaxresize(image, 416, 800)
        if pad:
            imsize = image.shape
            if imsize[0] > imsize[1]:
                tempim = np.zeros((imsize[0], imsize[0], 3), dtype='uint8')
                distant = (imsize[0] - imsize[1]) // 2
                tempim[:, distant:distant + imsize[1], :] = image
                image = tempim
                h = imsize[0]
                w = imsize[0]

            elif imsize[1] > imsize[0]:
                tempim = np.zeros((imsize[1], imsize[1], 3), dtype='uint8')
                distant = (imsize[1] - imsize[0]) // 2
                tempim[distant:distant + imsize[0], :, :] = image
                image = tempim
                h = imsize[1]
                w = imsize[1]

        new_image = preprocess_input(image, 416, 416)

        yolos = infer_model.predict(new_image)
        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += decode_netoutv3(yolos[i][0],
                                     config['model']['anchors'],
                                     0.5,
                                     416,
                                     416)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, 416, 416, 416, 416)

        # suppress non-maximal boxes
        do_nms(boxes, 0.45)

        # draw bounding boxes on the image using labels
        draw_boxesv3(image, boxes, config['model']['labels'][::-1], 0.5)
        im = Image.fromarray(image)
        b, g, r = im.split()
        im = Image.merge("RGB", (r, g, b))
        im.show()
