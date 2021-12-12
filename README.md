# yolo processor
This is a ros package. To use this package, open your ros workspace, then
```
cd src
```
```
git clone https://github.com/qaz1678249/yolop.git
```
```
cd ..
```
```
catkin_make
```
Don't forget to source after building:
```
source ./devel/setup.bash
```

## Usage of code
__imgp.py__  
This is for test. This node publishes two images in turn for some rate into /imgs topic. In /imgs there is one image.  
__yoloprocess.py__  
This is what you need to complete!!! Put your yolo process code where the comments are as what comments say. This node subscribes to /imgs topic, once it get the image from /imgs, it will process the image and publish the yolo result into /masks topic. There are three images in /masks topic: rawimg, idmask and scoremask.  
__imgsb.py__  
This is for test. This node subscribes to /masks topic. Once it gets the message, it will plot the idmask in the message.
# yolop
# yolop
