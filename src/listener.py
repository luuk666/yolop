import rospy
import numpy as np 
import cv2
from std_msgs.msg import Float64MultiArray 
def callback(msg):
    rospy.loginfo("I heared")
    #print(np.array(msg.data[0:1]))
    print(np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1])))
    cv2.imwrite('3.jpg',np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1])))
    s = cv2.imread('3.jpg')     
    cv2.imshow('img3',s)
    cv2.waitKey(1)
    #print("n=%.2f",np.array(msg.data).reshape(486,646))
def listener(): 
    rospy.init_node('listener', anonymous= True)
    rospy.Subscriber("robot_pose", Float64MultiArray, callback) 
    rospy.spin() 
if __name__ == '__main__':
    m=0 
    listener() 

