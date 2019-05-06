#coding=utf-8
import cv2
import threading
from action_recognition import GesterRecognition
import argparse
import time
from PIL import Image
from images import images
import numpy as np
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_queue',
        action='store_true',
        help='If true, random choice is not performed.')
    parser.set_defaults(use_queue=False)
    args = parser.parse_args()

    return args

class VideoCamera(object):
    def __init__(self,work):
        # 打开摄像头， 0代表笔记本内置摄像头
        self.cap = cv2.VideoCapture('./test.mp4')
        print('*'*50,'Init','*'*50)
        self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.currentFrames =0
        
        #self.opts=parse_opts()
        # cap = cv2.VideoCapture(video_path)
        self.frame_count=0
        self.new_frame_queue=[]
        # print('opt',opts.use_queue)
        self.work=work#GesterRecognition(self.opts.use_queue)
        self.images=images()
        #self.fps=AverageMeter()
        # self.start_time=time.time()
    # 退出程序释放摄像头
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (864, 705), interpolation=cv2.INTER_CUBIC)
        #image=np.concatenate((img.NoneFalse, img.NoneFalse))
        if ret:
            # print(frame.shape)
            self.frame_count+=1
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) 
            self.new_frame_queue.extend([image])
            result = None
            if self.frame_count==self.work.sample_duration:
                self.frame_count=0
                result = self.work.process_frame(self.new_frame_queue)
                self.new_frame_queue.clear()
            if result==None:
                frame = np.concatenate((frame, self.images.NoneFalse))
                #frame = cv2.putText(frame,"Action Recognition: None",(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            else :
                #text = "Action Recognition: {}".format(self.work.labels[result])
                #frame = cv2.putText(frame,text,(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                frame = np.concatenate((frame, self.images.GarbageFalse))
            #frame = cv2.putText(frame,"Abnormal Detection: False",(30,55),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            self.currentFrames += 1
            if self.currentFrames == self.totalFrames - 1:
                self.currentFrames = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # print(jpeg.shape) shape变成了(313124, 1)
            # jpeg = cv2.putText(jpeg,"abnormal action: None",(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)

            return jpeg.tobytes()

        else:
            return None
