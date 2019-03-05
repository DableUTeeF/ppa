import cv2
import numpy as np
import time


net = cv2.dnn.readNetFromTensorflow('t/TFMNet_model.pb')
img = np.zeros((416, 416, 3), dtype='float32')
blob = cv2.dnn.blobFromImage(img)
net.setInput(blob)

while True:
    t = time.time()
    out = net.forward()
    print(time.time()-t)
