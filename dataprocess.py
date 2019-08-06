import glob
import numpy as np

import torch
import torch.utils.data as Dataset
import torchvision.transforms as transforms
import cv2

from PIL import Image
dataset = "/home/yunchu/python_workspace/test200/" # this is the path of dataset
#dataset = "/home/yunchu/Workspace/Deep_CNN_with_VAE_for_graspe/test10cornell/"
#dataset = "/Users/zhili/Documents/test_dataset/"

NUM_LABELS = 4
mean = [0.485, 0.456, 0.406],
std = [0.229, 0.224, 0.225]
ratio = 640/1024.0

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start, end, transform = None):
        labels_txt = glob.glob(dataset + "*" + ".txt")
        labels_txt.sort()
        l_l = len(labels_txt)
        image_list = glob.glob(dataset + "*" + ".png")
        image_list.sort()
        l_img = len(image_list)
        self.label_files = labels_txt[int(l_l*start): int(l_l*end)]
        self.image_files = image_list[int(l_img*start): int(l_img*end)]
        self.transform = transform


    def __getitem__(self, index):
        # open image according to given index
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')

        #return label 
        label = self.process_label_files(self.label_files[index])
        label = torch.tensor(label)
        if self.transform: image = self.transform(image)
        return image, label

    
    def __len__(self):
        return len(self.image_files)
    
    def process_label_files(self, label_path):
        boxes = []
        box = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                a = line.split(';')
                '''
                if float(a[2]) < 0:
                    # x y w h theta
                    box = [ratio * float(a[0]), ratio * float(a[1]), ratio * float(a[3]), ratio * float(a[4]), 360.0 + float(a[2])]
                    
                else:
                    # x y h w theta
                    box = [ratio * float(a[0]), ratio * float(a[1]), ratio * float(a[3]), ratio * float(a[4]), float(a[2])]
                '''
                box = [ratio * float(a[0]), ratio * float(a[1]), ratio * float(a[3]), ratio * float(a[4]), float(a[2])]
                '''
                box = [ratio * float(a[0]), ratio * float(a[1]), ratio * float(a[3]), ratio * float(a[4]), float(a[2])]
                box = ((box[0]*ratio, box[1]*ratio), (box[2]*ratio, box[3]*ratio), box[4])
                bbox_contour = cv2.boxPoints(box)
                bbox_contour.reshape((1,8))
                '''
                #boxes.append(bbox_contour)
                boxes.append(box)
        return boxes[:NUM_LABELS]

class MyDataset_Cornell(torch.utils.data.Dataset):
    def __init__(self, dataset, transform = None, start = 0, end = 1.0):
        labels_txt = glob.glob(dataset + "*" + ".txt")
        labels_txt.sort()
        l_l = len(labels_txt)
        image_list = glob.glob(dataset + "*" + ".png")
        image_list.sort()
        l_img = len(image_list)
        self.label_files = labels_txt[int(l_l*start): int(l_l*end)]
        self.image_files = image_list[int(l_img*start): int(l_img*end)]
        self.transform = transform


    def __getitem__(self, index):
        # open image according to given index
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')

        #return label 
        label = self.box2label(self.label_files[index])
        label = torch.tensor(label)
        if self.transform: image = self.transform(image)
        return image, label

    
    def __len__(self):
        return len(self.image_files)   

    def box2label(self,labels_txt):
        a = np.loadtxt(labels_txt)
        boxes = []
        box = []
        for i in range(a.shape[0]):
            if i % 4 == 0:
                box = [a[i], a[i + 1], a[i + 2], a[i + 3]]
                box = np.int0(box)
                boxes.append(box)
        return boxes
    

def test():
    traindata = MyDataset(dataset = dataset, start = 0, end = 0.8, transform = transforms.Compose([transforms.Resize(640), transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(dataset = traindata, batch_size=2, shuffle=False)
    for i, data in enumerate(trainloader):
        imags, label = data
        print(label.size())
        print(imags.size())
#test()
