import cv2
from training import *



def process_label_files(label_path):
    boxes = []
    box = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            a = line.split(';')
            #if float(a[2]) < 0:
                # x y w h theta
                #box = [float(a[0]), float(a[1]), float(a[4]), float(a[3]), float(a[2])]
            #else:
                # x y h w theta
            box = [float(a[0]), float(a[1]), float(a[3]), float(a[4]), float(a[2])]
            
            boxes.append(box)
    return boxes[:NUM_LABELS]

def visulization():
    imgs = glob.glob(dataset + "*" + ".png")
    imgs.sort()
    labels = glob.glob(dataset + "*" + ".txt")
    labels.sort()

    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]

        img = cv2.imread(img)
        cv2.namedWindow("Image" + str(i))

        boxes = process_label_files(label)

        for box in boxes:
            #bbox_contour = grasp_to_bbox(box)
            #box = ((box[0], box[1]), (box[2], box[3]), box[4])
            bbox_contour = cv2.boxPoints(box)
            #print(bbox_contour.shape)
            cv2.drawContours(img, [bbox_contour.astype(int)], -1, (0, 255, 0), 1)

        cv2.imwrite("test_bbox" + "{}.png".format(i), img)
        break

visulization()