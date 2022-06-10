#coding:utf-8
#from importlib.resources import path
#import rosbag
#from cv_bridge import CvBridge
import rospy
import numpy as np 
import cv2
from std_msgs.msg import Float64MultiArray 
import time
time_start=time.time()
#num=1
def callback(msg):
    #global  num
    global time_start
    time_end=time.time()
    #path='/matrix'
    rospy.loginfo("I heared")
    #print(np.array(msg.data[0:1]))
    np.set_printoptions(threshold=np.inf)  
    #print(np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1])))
    print('totally cost',time_end-time_start)
    cv_matrix = np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1]))  
    cv2.imwrite('3.jpg',np.array(list(msg.data[2:])).reshape(int(msg.data[0]),int(msg.data[1])))
    s = cv2.imread('3.jpg')     
    cv2.imshow('img3',s)
    cv2.waitKey(1)
    #timestr = "%.6f" %  msg.header.stamp.to_sec()  # %.6f表示小数点后带有6位，可根据精确度需要修改；
     #命名格式
    '''matrix_name = str(num) + '.txt'                   # txt命名：时间戳.txt
    #cv2.imwrite(path+ matrix_name, cv_matrix)        # 保存；
    file = open(matrix_name, 'w')
    #写入的语句
    file.write(cv_matrix)'''

    #np.savetxt('./'+ str(time_end-time_start)+'.txt', cv_matrix,fmt ='%d')
    #np.savetxt('./'+ str(cv_matrix.max())+'.txt', cv_matrix,fmt ='%d')
    #np.savetxt('./'+ str(np.unique(cv_matrix)[0])+'.txt', cv_matrix,fmt ='%d')
    #print(str(cv_matrix.max()))
    print(str(np.unique(cv_matrix)[0]))
    #num=num+1

    #print("n=%.2f",np.array(msg.data).reshape(486,646))
def listener(): 
    rospy.init_node('listener', anonymous= True)
    rospy.Subscriber("robot_pose", Float64MultiArray, callback) 
    rospy.spin() 
if __name__ == '__main__':
    m=0 
    listener() 

