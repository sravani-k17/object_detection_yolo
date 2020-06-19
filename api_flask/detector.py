import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

class Detector:
	def __init__(self,weights_path,config_path):
	    global net
	    net = cv2.dnn.readNet(weights_path,config_path)
	def detect_object(self,image_path):
		classes = ["hairband"]
		layer_names = net.getLayerNames()
		output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		colors = np.random.uniform(0, 255, size=(len(classes), 1))
		img = cv2.imread(image_path)
		img = cv2.resize(img, None, fx=0.4, fy=0.4)
		height, width, channels = img.shape
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)
		
		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
		print(indexes)
		font = cv2.FONT_HERSHEY_PLAIN
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				#print(class_ids[i])
				if(class_ids[i]<len(classes)):
					label = str(classes[class_ids[i]])
					color = colors[class_ids[i]]
					cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
					cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
			
		
		return img,class_ids
		
		
		
