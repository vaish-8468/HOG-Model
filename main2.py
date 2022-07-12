# import cv2
# import numpy as np
# from object_detection import ObjectDetection
# from deep_sort.deep_sort import Deep

# print(cv2.__version__)

# # load object detection
# od=ObjectDetection("dnn_model/yolov4x-mish.weights","dnn_model/yolov4x-mish.cfg")


# od.load_class_names("dnn_model/classes.txt")
# od.load_detection_model(image_size=648,
# nmsThreshold=8.4, confThreshold=8.3)


import os
from pathlib import Path
import cv2
import matplotlib as plt


p=Path("C:/Users/HP/Desktop/iot-picamera/Human-Body-Detection-OpenCV-Python-Source-Code/count/human detection dataset/")
dirs=p.glob("*")


image_data=[]
labels=[]
labels_dict={"0":0,"1":1}

for folder_dir in dirs:
    label=str(folder_dir).split("\\")[-1]

    count=0
    print(folder_dir)

    #iterate over folder directory and pick all the images
    for img_path in folder_dir.glob("*.jpg"):
        print(img_path)
    #     img=cv2.load_img(img_path,target_size=(100,100))
    #     img_array=cv2.img_to_array(img)
    #     image_data.append(img_array)
    #     labels.append(labels_dict[label])
    #     count+=1
    # print(count)  
