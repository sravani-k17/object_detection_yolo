# Run the code with below params
# python test_video.py -c gun_classes.txt -m gun_weights.h5 -a anchors.txt

# python test_video.py -c model_data/obj.names -m yolov4_custom_weights_45000.h5 -a model_data/yolo4_anchors.txt
import os
import colorsys
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import argparse

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(416, 416, 3)), num_anchors//3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
    def close_session(self):
        self.sess.close()

    def detect_image(self, image,cv2_img, model_image_size=(608, 608)):
        start = timer()

        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #             size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 300
        # COLORS = np.random.uniform(0, 255, size=(len(self.class_names), 3)).astype('int32')
        inf = timer()
        print('INference Time:-',inf-start)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)
            label_size = cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
            print(label_size)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            color = self.colors[c]

            print(label, (left, top), (right, bottom),image.size[0],image.size[1])
            cv2.rectangle(cv2_img, (left,top), (right,bottom),color, 2)
            cv2.putText(cv2_img, label, tuple(text_origin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # cv2.imshow('Detections',draw)
            # del draw

        end = timer()
        print(end - start)
        return cv2_img


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('-c','--classes_path',default = 'model_data/obj.names',required =True,help='Path to class names file')
    ap.add_argument('-m','--model_path',default = 'yolov4_custom_weights_22000.h5',required =True,help='Path to model weights file')
    ap.add_argument('-a','--anchors_path',default = 'model_data/yolo4_anchors.txt',required=True,help='anchors text path file')
    # ap.add_argument('-w','--weights_path' ,default = 'yolov4-custom_22000.weights',required=True,help='weights path file')
    args = vars(ap.parse_args())
    model_path = args['model_path']
    anchors_path = args['anchors_path']
    # classes_path = 'model_data/coco_classes.txt'
    classes_path = args['classes_path']

    # model_path = 'yolo4_customweight.h5'
    # anchors_path = 'model_data/yolo4_anchors.txt'
    # classes_path = 'model_data/obj.names'

    score = 0.3
    iou = 0.4

    model_image_size = (416, 416)

    yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)
    while  True:
        video_path = input('Input Video filename if camera type 0:')
        if video_path == '0':
            cap = cv2.VideoCapture(0)
            break
        else:
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                break
            else:
                print("Path doesn't exits")    
    plt.ion()
    # fps = FPS().start()
    count =0 
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    fource = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('./output/Output_knife.avi', fource, 4,
            (width,height), True)
    while True:
        _,image = cap.read()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2_img = image
        image = Image.fromarray(image)

        count+=1
        if count%10 == 0:
            print(count)
            result = yolo4_model.detect_image(image,cv2_img, model_image_size=model_image_size)
            result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            #cv2.imshow("object detection", cv2.resize(result,(700,600),interpolation=cv2.INTER_LANCZOS4))
            writer.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # plt.imshow(result)
            # plt.pause(0.2)
            # plt.show()
    # plt.ioff()
    # plt.show()
    cap.release()
    cv2.destroyAllWindows() 
    yolo4_model.close_session()

