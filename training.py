import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import time
import copy
from tensorboardX import SummaryWriter

from dataprocess import *
from grasping_evaluation import *
from myModel import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

#writer
writer = SummaryWriter(log_dir = 'log')

#parameters
DATA_SPLIT = 0.8 # the split of training and validation data
EPOCH = 400
BATCH_SIZE = 10
lr = 0.0005
GPU = True

DATA_SET = "Jacquard"

train_data = MyDataset(dataset = dataset, start = 0, end = DATA_SPLIT, transform = transforms.Compose([transforms.Resize(640), transforms.ToTensor()]))
#train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_data = MyDataset(dataset = dataset, start = DATA_SPLIT, end = 1.0, transform = transforms.Compose([transforms.Resize(640), transforms.ToTensor()]))
#val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)


#train_data = MyDataset_Cornell(dataset = dataset, start = 0, end = DATA_SPLIT, transform = transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

#val_data = MyDataset_Cornell(dataset = dataset, start = DATA_SPLIT, end = 1.0, transform = transforms.Compose([transforms.ToTensor()]))
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

def Loss_calculation(pred, label):
    # the dimension of pred is tensor [batch_size, dim_output]
    # the dimension of label is tensor [batch, num_labels_per_image, 5]
    #Jacquard
    if DATA_SET == "Jacquard":
        label = torch.reshape(label, (pred.size(0), NUM_LABELS * 8))
    else:
        label = torch.reshape(label, (pred.size(0), 24))
    label = label.to(torch.float)

    pred = pred.to(torch.float)

    return mse(pred, label)

def Loss_cal(pred, label):
    if DATA_SET == "Jacquard":
        label = torch.reshape(label, (pred.size(0), NUM_LABELS * 5)) # x y theta h w
    else:
        label = torch.reshape(label, (pred.size(0), 24))
    label = label.to(torch.float)

    pred = pred.to(torch.float)

    loss = 0
    for i in range(0, NUM_LABELS * 5, 5):
        loss += mse(pred, label[:,i:i+5])
    return loss/NUM_LABELS

    



def training():
    #temporarily use ResNet18 as our model
    #model = models.resnet50(pretrained = True)
    #num_ftrs = model.fc.in_features # the input dimension of fc of resnet18
    #model.fc = nn.Linear(num_ftrs, 5 * NUM_LABELS) # the output dim should be 5 corresponding to x, y, w, h, theta
    #model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.Linear(512, 200), nn.Linear(200, 5*NUM_LABELS)) #Jacquard
    #model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.Linear(512, 200), nn.Linear(200, )) #Cornell

    model = myModel()

    if GPU: model = model.to(device)
    #model = sq.cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr = lr)

    #Decay LR by a factor of 0.9 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 20, gamma = 0.9)

    since = time.time()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-'*20)

        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == "train":
                exp_lr_scheduler.step()
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            running_acc = 0

            #Iterate over data
            for i, data in enumerate(train_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                #zero the parameter gradients
                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
        
                    pred = model(images) # pred should have the same dim with labels
                    
                    #loss = Loss_calculation(pred, labels)
                    loss = Loss_cal(pred, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()
                    
                #statistics
                running_loss += loss.item() * images.size(0)
                if DATA_SET == "Jacquard":
                    running_acc += acc_cornell(pred, labels, images.size(0))
                    batch_loss = loss.item()
                    batch_acc = acc_cornell(pred, labels, images.size(0))
                else:
                    running_acc += acc_cornell(pred, labels, images.size(0))
                    batch_loss = loss.item()
                    batch_acc = acc_cornell(pred, labels, images.size(0))

                #print('{} Batch_Loss: {:.4f} Batch_Acc: {:.4f}'.format(phase, batch_loss, batch_acc))
            
            epoch_loss = running_loss / train_data.__len__()
            epoch_acc = running_acc / train_data.__len__()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalars('log/train', {'train_loss':epoch_loss, 'train_acc':epoch_acc}, epoch)
            else:
                writer.add_scalars('log/validation', {'validation_loss': epoch_loss, 'validation_acc': epoch_acc}, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    writer.close()

if __name__ == '__main__':
    training()
