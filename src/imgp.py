import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np


if __name__ == '__main__':
	rospy.init_node("imagepub", anonymous=True)
	img1 = cv2.imread('dog.jpg')
	img2 = cv2.imread('eagle.jpg')
	cvb = CvBridge()
	pub = rospy.Publisher("/camera/rgb/image_color",Image,queue_size=10)
	rate = rospy.Rate(2)
	i = 0
	while not rospy.is_shutdown():
		if i == 0:
			pub.publish(cvb.cv2_to_imgmsg(img1))
			i=1
		else:
			pub.publish(cvb.cv2_to_imgmsg(img2))
			i=0
		rate.sleep()
	rospy.spin()



