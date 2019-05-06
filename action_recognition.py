#coding=utf-8
import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from models import TSN
from transforms import *
import sys
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GesterRecognition(object):
    """docstring for GesterRecognition"""
    def __init__(self,use_queue,model_name='ECO'):
        super(GesterRecognition, self).__init__()

        self.use_queue = use_queue
        print(self.use_queue)
        if self.use_queue==True:
            self.old_frame_queue=[]
        else:
            self.old_frame_queue=None

        self.labels=self.get_labels()
        print(self.labels)
        self.get_model()
    def get_labels(self):  
        labels_dic={}
        labels_dic['drip']=0
        labels_dic['garbage']=1
        labels_dic['smash']=2
        labels_dic['stall']=3
        labels_dic[0]='drip'
        labels_dic[1]='garbage'
        labels_dic[2]='smash'
        labels_dic[3]='stall'
        
        return labels_dic
    def get_model(self):
        self.model=TSN(4, 16, 'finetune', 'RGB',
                base_model='ECO',
                consensus_type='identity', dropout=0.3, partial_bn=False)
        self.model = torch.nn.DataParallel(self.model, device_ids=None).cuda()
        print(("=> loading checkpoint '{}'".format('demo_rgb_epoch_40_checkpoint.pth.tar')))
        checkpoint = torch.load('demo_rgb_epoch_40_checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Done')
        #val mode
        torch.set_grad_enabled(False)

        self.model.eval()
        self.input_size=224
        self.input_mean = [104, 117, 128]
        self.input_std = [1]
        self.sample_duration=16
        # self.normalize=GroupNormalize(self.input_mean,self.input_std)
        def transform(frame):
            normalize = GroupNormalize(self.input_mean, self.input_std)
            transforms=torchvision.transforms.Compose([
                   GroupScale(256),
                   GroupCenterCrop(self.input_size),
                   Stack(roll=True),
                   ToTorchFormatTensor(div=False),
                   normalize,
               ])
            return transforms(frame)
        self.transform_func=transform
    def process_frame(self,new_frame_queue):
        if self.use_queue:
            if len(self.old_frame_queue)==self.sample_duration:
                half_sample_duration=int(self.sample_duration/2)
                frame_from_old_idx=sorted(random.sample(list(range(self.sample_duration)),
                                            half_sample_duration))
                frame_from_old=[self.old_frame_queue[i] for i in frame_from_old_idx]

                frame_from_new_idx=sorted(random.sample(list(range(self.sample_duration)),
                                            half_sample_duration))
                frame_from_new=[new_frame_queue[i] for i in frame_from_new_idx]

                self.old_frame_queue=frame_from_old+frame_from_new

                input_tensor=self.transform_func(self.old_frame_queue).unsqueeze_(0)
                input_var=torch.autograd.Variable(input_tensor,volatile=True).cuda()
                _,predict=self.model(input_var).topk(1)
                result='reslut:{}'.format(self.labels[int(predict.data[0][0])])
                print(result)
                return int(predict.data[0][0])
            else:
                self.old_frame_queue=new_frame_queue
                
                # input_tensor=torch.from_numpy(np.array(frame_queue)).permute(1,0,2,3).unsqueeze_(0)
                input_tensor=self.transform_func(self.old_frame_queue).unsqueeze_(0)
                input_var=torch.autograd.Variable(input_tensor).cuda()
                _,predict=self.model(input_var).topk(1)
                result='result:{}'.format(self.labels[int(predict.data[0][0])])
                print(result)
                return int(predict.data[0][0])
        else:
            
            frame_queue=new_frame_queue

            input_tensor=self.transform_func(frame_queue).unsqueeze_(0)

            input_var=torch.autograd.Variable(input_tensor).cuda()
            _,predict=self.model(input_var).topk(1)
            result='result:{}'.format(self.labels[int(predict.data[0][0])])
            print(result)
            return int(predict.data[0][0])

# def parse_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--use_queue',
#         action='store_true',
#         help='If true, random choice is not performed.')
#     parser.set_defaults(use_queue=False)
#     args = parser.parse_args()

#     return args
# def cv_video(video_path):
#     # opts=parse_opts()
#     # cap = cv2.VideoCapture(video_path)
#     # frame_count=0
#     # new_frame_queue=[]
#     # print('opt',opts.use_queue)
#     # work=GesterRecognition(opts.use_queue)
#     # fps=AverageMeter()
#     # start_time=time.time()
#     while(1):
#         # get a frame
#         ret, frame = cap.read()
#         if ret==0:
#             print('FPS:',1/fps.avg)
#             break
#         frame_count+=1
#         image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) 
#         new_frame_queue.extend([image])
        
#         if frame_count==work.sample_duration:
#             frame_count=0
#             # a=time.time()
#             print(work.process_frame(new_frame_queue))
#             # print('Forward Time:',time.time()-a)
#             new_frame_queue.clear()
#         # show a frame
#         cv2.imshow("capture", frame)
#         end_time=time.time()-start_time
#         fps.update(end_time)
#         start_time=time.time()
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             print('FPS:',1/fps.avg)
#             break
#     cap.release()
#     cv2.destroyAllWindows() 
# cv_video(0)
