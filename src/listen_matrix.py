#coding:utf-8
from importlib.resources import path
import rosbag
from cv_bridge import CvBridge
import rospy
import numpy as np 
import cv2
from std_msgs.msg import Float64MultiArray 

path='/matrix'

bag_file = 'freiburg2_desk_with_person_MIX.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
bag_data = bag.read_messages()    # 利用迭代器返回三个值：{topic标签, msg数据, t时间戳}

bridge = CvBridge()
for topic, msg, t in bag_data:
    if topic == "/robot_pose  ":
        cv_matrix = np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1]))  
        print(cv_matrix)

        timestr = "%.6f" %  msg.header.stamp.to_sec()  # %.6f表示小数点后带有6位，可根据精确度需要修改；
        matrix_name = timestr+ ".txt"                   # txt命名：时间戳.txt
        cv2.imwrite(path+ matrix_name, cv_matrix)        # 保存；