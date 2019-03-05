import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from frontend import NolambdaYOLO
import json
import onnxmltools
from keras import layers, models


def stupidmodel():
    inp = layers.Input((224, 224, 3))
    x = layers.Conv2D(32, 3, padding='same')(inp)
    return models.Model(inp, x)


if __name__ == '__main__':
    config_path = 'jsconfig.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo = NolambdaYOLO(backend=config['model']['backend'],
                        input_size=config['model']['input_size'],
                        labels=config['model']['labels'],
                        max_box_per_image=config['model']['max_box_per_image'],
                        anchors=config['model']['anchors'])
    # yolo.model.load_weights(config['train']['pretrained_weights'])

    # onnx_model = onnxmltools.convert_keras(ResNet50(weights=None), target_opset=8)
    onnx_model = onnxmltools.convert_keras(stupidmodel())

    onnxmltools.utils.save_text(onnx_model, 't/example.json')

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, 't/example.onnx')
