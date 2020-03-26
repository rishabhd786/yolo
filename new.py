import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'svg'

options = {"model": "cfg/yolo.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 2,
           "epoch": 5,
           "train": True,
           "annotation": "new_data/annotations/",
           "dataset": "new_data/images/"}
tfnet = TFNet(options)
tfnet.train()