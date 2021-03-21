import numpy as np
import cv2
from flask import Flask, request, jsonify
import json
import matplotlib.pyplot as plt

app = Flask(__name__)
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
f = open("coco.names" , "r")
objectClasses = [line.strip() for line in f.readlines()]

@app.route("/predict", methods=["POST"])

def predict():
    meta = json.load(request.files['meta'])
    img = request.files['img'].read()
    layers = yolo.getLayerNames();
    outputLayerIndices = yolo.getUnconnectedOutLayers();
    outputLayers = [layers[i[0] - 1] for i in outputLayerIndices]
    img = np.frombuffer(img, dtype=np.uint8)
    img = np.reshape(img, meta['shape'])
    
    img = cv2.resize(img, None, fx = 0.4, fy = 0.3)
    height, width, channels = img.shape
    blobs = cv2.dnn.blobFromImage(img, 0.00392 , (416, 416), (0, 0, 0), True , crop = False)
    yolo.setInput(blobs)
    outputs = yolo.forward(outputLayers)
    ot = np.array(outputs)
    
    object_ids = []
    possibilities = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            object_id = np.argmax(scores)
            possibility = scores[object_id]
            
            if(possibility > 0.5):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2);
                y = int(center_y - h / 2);
                
                object_ids.append(object_id)
                possibilities.append(float(possibility))
                boxes.append([x,y,w,h])
                
    #removing the multiple similar objects
    
    uniqueIndices = cv2.dnn.NMSBoxes(boxes, possibilities , 0.4 , 0.6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in uniqueIndices:
            x , y , w , h = boxes[i]
            label = str(objectClasses[object_ids[i]])
            cv2.rectangle(img, (x , y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, label , (x , y - 10), font , 1 , (255, 255, 255), 2)
    
    return jsonify(img.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

