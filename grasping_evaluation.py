from shapely.geometry import Polygon
import torch

from dataprocess import *

def grasp_to_bbox(grasp_stack):
    x, y, h, w, theta = grasp_stack
    edge1 = np.array([x -w/2*np.cos(theta) +h/2*np.sin(theta), y -w/2*np.sin(theta) -h/2*np.cos(theta)])
    edge2 = np.array([x +w/2*np.cos(theta) +h/2*np.sin(theta), y +w/2*np.sin(theta) -h/2*np.cos(theta)])
    edge3 = np.array([x +w/2*np.cos(theta) -h/2*np.sin(theta), y +w/2*np.sin(theta) +h/2*np.cos(theta)])
    edge4 = np.array([x -w/2*np.cos(theta) -h/2*np.sin(theta), y -w/2*np.sin(theta) +h/2*np.cos(theta)])

    return np.array([edge1, edge2, edge3, edge4])

def calculate_IOU(box_model, box_label, theta_model, theta_label):
    p1 = Polygon(box_model).convex_hull
    p2 = Polygon(box_label).convex_hull
    iou = p1.intersection(p2).area / (p1.area + p2.area - p1.intersection(p2).area)

    angle_diff = np.abs(theta_model * 180 / np.pi - theta_label * 180 / np.pi)

    return iou, angle_diff
    #if angle_diff <= 30.0 and iou >= 0.25:
    #    count += 1

def acc(pred, label, num_imgs):
    # num_imgs is the number of images in a batch let's say 20
    # dimension of label: [batch_size(num_imgs), num_of_label(10), 5(x y theta w h)]
    iou_per_image = []
    accuracy = 0.0
    count = 0

    pred = torch.reshape(pred, (num_imgs, NUM_LABELS, 5))

    for i in range(num_imgs):
        for j in range(label.size(1)):
            pred_bbox = grasp_to_bbox(pred[i,j,:].detach().cpu().numpy())
            label_bbox = grasp_to_bbox(label[i,j,:].detach().cpu().numpy())
            iou, angle_diff = calculate_IOU(pred_bbox, label_bbox, pred[i,j,-1].detach().cpu().numpy(), label[i,j,-1].detach().cpu().numpy())
            iou_per_image.append([iou, angle_diff])
        
        for k in range(len(iou_per_image)):
            if iou_per_image[k][0] >= 0.25 and iou_per_image[k][1] <= 30.0:
                count += 1
        
        accuracy += count/num_imgs

        #renew lists and var
        iou_per_image = []
        count = 0
    
    return accuracy




