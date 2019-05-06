import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import csv

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

#/nfs/project/UCF101/ root_path
#root_path+UCF-101+label+video_name+*.jpg==path
#root_path+list_file ==list_file
class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = '/nfs/project/surveillance/data/abnormal_action_video_data/abnormal_action_data_list.csv'
        self.label_file = os.path.join(root_path,"classInd.txt")
        self.num_segments = num_segments#temporal sample
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = '{:05d}.jpg'
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self.label2index=self.get_labels()
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]
    def  get_labels(self):
        index=0
        label2index={}
        label2index['drip']=0
        label2index['garbage']=1
        label2index['smash']=2
        label2index['stall']=3
        #with open(self.label_file,'r') as myFile:    
        #    for line in myFile:
        #        line=line.strip().split(' ')[1]  
        #        label2index[line]=index
        #        index+=1
        return label2index
    def _parse_list(self):
        #self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        video_list=[]
        with open(self.list_file,'r') as myFile:  
            lines=csv.reader(myFile)  
            for line in lines:  
                # line=line[0]
                #split=line[0]
                label=line[0]#.split(';')
                video_name=line[1]
                video_name=video_name.split('.')[0]
                num_frames=int(line[2])
                
                path=os.path.join('/nfs/project/surveillance/data/abnormal_action_images_data',label,video_name)
                # num_frames=int(line[2])
                # print(self.label2index.keys())
                if self.modality=='Flow':
                    num_frames-=1
                # if label not in self.label2index.keys():
                #    print('shit')
                #    continue
                if num_frames<=0:
                    continue
                video_list.append(VideoRecord([path,num_frames,self.label2index[label]]))
                # print([video_name,num_frames,self.label2index[label]])
        self.video_list=video_list

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
