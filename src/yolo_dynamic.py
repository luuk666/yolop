#coding=utf-8
from re import X
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import save
from scipy.stats.stats import trimboth
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from yolop.msg import twoimgs
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import number, random
from std_msgs.msg import Float64MultiArray
########中间这部分是给高斯分布加的#########
import numpy as np
from scipy import stats
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
########################################
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from plots import plot_one_box
from torch_utils import select_device, load_classifier, time_synchronized
torch.cuda.is_available()
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
#np.set_printoptions(threshold=np.inf)
class YOLOP(object):
        def __init__(self,
            weights='yolov5s.pt',  # model.pt path(s)
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False, # use FP16 half-precision inference
            ): 
                self.image = None
                self.yoloresult = twoimgs()
                self.cvb = CvBridge()
                self.source = source
                self.weights = weights
                self.view_img = view_img
                self.save_txt = save_txt
                self.save_img = not nosave and not source.endswith('.txt')
                self.device = select_device(device)
                self.half = False
                ##############################################################
                webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
                self.half  &= self.device.type != 'cpu' 
                model = attempt_load(weights,map_location=self.device) 
                 # load FP32 model
                self.stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
                names = model.module.names if hasattr(model, 'module') else model.names
                if half:
                    model.half()  # to FP16
                # Second-stage classifier
                self.classify = False
                self.modelc = None
                if self.classify:
                    modelc = load_classifier(name='resnet101', n=2)  # initialize
                    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
                    self.modelc = modelc
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                # Run inference
                if self.device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
                self.model = model
                self.imgsz = imgsz
                self.names = names
                self.augment = augment
                self.conf_thres = conf_thres
                self.iou_thres = iou_thres
                self.classes = classes
                self.agnostic_nms = agnostic_nms
                self.probability = None #!!!!!!!!!!!!
                self.pub = rospy.Publisher('/camera/image_raw', Image,queue_size=1)
                rospy.Subscriber("/kitti/camera_color_left/image_raw",Image,self.callback,queue_size=1,buff_size=52428800)
                self.pub2 = rospy.Publisher('robot_pose', Float64MultiArray, queue_size=10)
        
        def loadimage(self, image, imgsz):
                cap = None
                path=None
                img0 = image
                img = letterbox(img0, imgsz, stride=self.stride)[0]
                img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                return path, img, img0, cap
                print('加载图片')

        def callback(self, msg):
                rospy.loginfo('Image received...')
                self.image = self.cvb.imgmsg_to_cv2(msg)
                yolo_idmask = self.image #this should be your first result
                yolo_scoremask = self.image #this should be your second result
                yolo_probability = self.image
                ##############################################################################################
                source   = self.source
                weights  = self.weights
                view_img = self.view_img
                save_txt = self.save_txt
                save_img = self.save_img 
                # Initialize
                #print("初始化")
                set_logging()
                t0 = time.time()
                path, img, img0, cap = self.loadimage(self.image, self.imgsz)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0    # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:                           
                    img = img.unsqueeze(0)
                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.augment)[0]
                
                # Apply NMS
                #print("进行nms")
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
                t2 = time_synchronized()
                
                # Apply Classifier
                #print("应用类")
                if self.classify:
                    pred = apply_classifier(pred, self.modelc, img, img0)
                
                # Process detections
                for i, det in enumerate(pred):
                    p, s, im0 = path, '', img0
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                        
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        
                        p=0
                        wide=0
                        high=0
                        rank_2 = np.ones((im0.shape[0],im0.shape[1]))*0.1
                        for *xyxy, conf, cls in reversed(det):
                            if self.names[int(cls)] == 'person':
                                continue
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            if save_img  or view_img:    #Add bbox to image
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                c = int(cls)
                                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3) 
                                x1=int(xyxy[0])
                                y1=int(xyxy[1])
                                x2=int(xyxy[2])
                                y2=int(xyxy[3])
                                a=np.linspace(int(xyxy[0]),int(xyxy[2]),int(xyxy[2])-int(xyxy[0]))
                                b=np.linspace(int(xyxy[1]),int(xyxy[3]),int(xyxy[3])-int(xyxy[1]))
                                meany=sum(a)/len(a)
                                meanx=sum(b)/len(b)
                                standard_deviationx=5.0
                                standard_deviationy=5.0*(y2-y1)/(x2-x1)                            
                                for i in np.linspace(x1,x2,x2-x1):
                                    for j in np.linspace(y1,y2,y2-y1):
                                        z = np.exp(-((i-meany)**2/(2*(standard_deviationx**2)) + (j - meanx)**2/(2*(standard_deviationy**2))))
                                        #print(np.exp(-((i-meany)**2/(2*(standard_deviation**2)) + (j - meanx)**2/(2*(standard_deviation**2))))
                                        z = z/(np.sqrt(2*np.pi)*standard_deviationy*standard_deviationx)
                                        if (rank_2[int(j-1)][int(i-1)]==0.1):
                                            rank_2[int(j-1)][int(i-1)]=int(z*1000*200*50)
                                            if(rank_2[int(j-1)][int(i-1)]<0.11):
                                                rank_2[int(j-1)][int(i-1)]=0.11
                                        else:
                                            rank_2[int(j-1)][int(i-1)]=rank_2[int(j-1)][int(i-1)]+int(z*1000*200*50)*0.5
                            #rank_2=rank_2*50
                            p=p+1
                            print("SUM of i=",p)#在每张图片处理结束后显示
                        print(f'Done. ({time.time() - t0:.3f}s)')
                        #print("max:",np.max(rank_2))
                        #print("min",np.min(rank_2))
                        max=np.max(rank_2)
                        min=np.min(rank_2)
                        print(shape(rank_2))
                        m=0
                        while m<=479:
                            n=0
                            while  n<=639:
                                rank_2[m,n] = (rank_2[m,n]-min)/(max-min)*0.89+0.1
                                n=n+1
                            m=m+1
                        print("max:",np.max(rank_2))
                        print("min",np.min(rank_2))
                        rate = rospy.Rate(2)
                        sizelist=[im0.shape[0],im0.shape[1]]
                        r1=Float64MultiArray()
                        final=np.array(rank_2.reshape(rank_2.size))
                        r1.data=np.append(sizelist,final)
                        self.pub2.publish(r1)
                        rate.sleep()
                        self.image = im0[:, :, [2, 1, 0]]
                        yolo_idmask= self.image
                        yolo_scoremask=self.image
                #yolo_probability=rank_2
               

		        ##############################################################
                self.yoloresult.rawimg = self.cvb.cv2_to_imgmsg(self.image)
                self.yoloresult.idmask = self.cvb.cv2_to_imgmsg(yolo_idmask)
                self.yoloresult.scoremask = self.cvb.cv2_to_imgmsg(yolo_scoremask)
                #self.yoloresult.probability = self.cvb.cv2_to_imgmsg(yolo_probability)
                self.pub.publish(self.cvb.cv2_to_imgmsg(self.image))
                ###################添加矩阵位置######################



if __name__ == '__main__':
        rospy.init_node("yolop", anonymous=True)
        #*********************************************************
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='.model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        check_requirements(exclude=('pycocotools', 'thop'))
        yolop = YOLOP(**vars(opt))
        rospy.spin()
