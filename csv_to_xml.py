import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
def save_xml(image_name, bbox, save_dir='./SunHat/Sun hat/Images/', class_name = 'Sun hat',width=1609, height=500, channel=3):
 
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = './SunHat/Sun hat/Images/train.txt'
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
 
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
 
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
 
    for x, y, x1, y1 in bbox:
        left, top, right, bottom = x, y, x1, y1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = class_name
        node_truncated = SubElement(node_object,'truncated')
        node_truncated.text = '1'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
 
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
 
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)
 
    return
 
 
def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox = label_dict.get(image, [])
        save_xml(image_name, bbox)
    return
 
import pandas as pd
import numpy as np
data = pd.read_table("./SunHat/Sun hat/Images/Sun hat_Train_Ready.csv",sep=",")
name_file=open('./SunHat/Sun hat/Images/train.txt','r')
name_file=name_file.readlines()
import re

image_path='./SunHat/Sun hat/Images/'
for name in os.listdir(image_path):
	if name.endswith('jpg'):
    # name = re.sub('/','\\',name)
    # print(name)
    # print(name)
	    img=cv2.imread(os.path.join(image_path,name))
	    height,width  = img.shape[:2]
	    class_name = data[data['filename'] == name]['class'].values
	    if len(class_name) == 0:
	        continue
	    else:
	        class_name = class_name[0]
	    #print(data.columns)
	    name=name.split('/')[-1]
	    xx = np.array(data[data['filename'] == name][['XMin','YMin','XMax','YMax']])
	    print(xx,class_name)
	    bbox=[]
	    for i in range(xx.shape[0]):
	        bbox.append(xx[i])
	    save_xml(image_name=name, bbox=bbox, save_dir='./SunHat/Sun hat/Images/',class_name = class_name, width=width, height=height, channel=3)