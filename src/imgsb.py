import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from yolop.msg import twoimgs

def callback(msg):
	cvb = CvBridge()
	img = cvb.imgmsg_to_cv2(msg)
	cv2.imshow("img",img)
	cv2.waitKey(1)


if __name__ == '__main__':
	rospy.init_node("imagesb", anonymous=True)
	rospy.Subscriber('/masks', Image, callback)
	rospy.spin()
	
