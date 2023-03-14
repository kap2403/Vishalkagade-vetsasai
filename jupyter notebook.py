#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt


# In[ ]:





# # conversion and Iou calculation functions
# 

# In[2]:


import numpy as np
from typing import Tuple, Union


def convert_xyxy_to_cxcywh(
    x1: Union[int, float],
    y1: Union[int, float],
    x2: Union[int, float],
    y2: Union[int, float]
) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return cx, cy, w, h


def convert_cxcywh_to_xyxy(
    cx: Union[int, float],
    cy: Union[int, float],
    w: Union[int, float],
    h: Union[int, float]
) -> Tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def convert_xywh_to_xyxy(
    x: Union[int, float],
    y: Union[int, float],
    w: Union[int, float],
    h: Union[int, float]
) -> Tuple[float, float, float, float]:
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2


def convert_poly_to_yolobbox(
    polygon: np.ndarray
) -> Tuple[float, float, float, float]:
    """polygon must be a Nx2 matrix
    """
    x1 = polygon[:, 0].min()
    x2 = polygon[:, 0].max()
    y1 = polygon[:, 1].min()
    y2 = polygon[:, 1].max()
    return convert_xyxy_to_cxcywh(x1, y1, x2, y2)


def convert_yolo_to_xyxy(
    cx: Union[int, float],
    cy: Union[int, float],
    w: Union[int, float],
    h: Union[int, float],
    img_w: Union[int, float],
    img_h: Union[int, float]
) -> Tuple[float, float, float, float]:
    x1 = cx * img_w - (img_w * w) / 2.0
    x2 = cx * img_w + (img_w * w) / 2.0
    y1 = cy * img_h - (img_h * h) / 2.0
    y2 = cy * img_h + (img_h * h) / 2.0
    return x1, y1, x2, y2


def convert_normalized_to_xyxy(
    x1: Union[int, float],
    y1: Union[int, float],
    x2: Union[int, float],
    y2: Union[int, float],
    img_w: Union[int, float],
    img_h: Union[int, float]
) -> Tuple[float, float, float, float]:
    x1 = x1 * img_w
    x2 = x2 * img_w
    y1 = y1 * img_h
    y2 = y2 * img_h
    return x1, y1, x2, y2


def calc_iou(x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max):
    i_min_x = max(x1_min, x2_min)
    i_min_y = max(y1_min, y2_min)
    i_max_x = min(x1_max, x2_max)
    i_max_y = min(y1_max, y2_max)

    inter_width = max((i_max_x - i_min_x), 0)
    inter_height = max((i_max_y - i_min_y), 0)

    width_box1 = abs(x1_max - x1_min)
    height_box1 = abs(y1_max - y1_min)

    width_box2 = abs(x2_max - x2_min)
    height_box2 = abs(y2_max - y2_min)

    area1 = width_box1 * height_box1
    area2 = width_box2 * height_box2
    intersection = (inter_width * inter_height)
    union = area1 + area2 - intersection
    iou = intersection / union

    if iou < 0.0 or iou > 1.0:
        iou = 0.0

    return iou, i_min_x, i_min_y, i_max_x, i_max_y


def np_calc_iou(
    box_xyxy_gt: Tuple[float, float, float, float],
    anchor_boxes_xyxy: np.ndarray,  # Nx4 boxes in XYXY
) -> np.ndarray:
    x2_min = anchor_boxes_xyxy[:, 0:1]
    y2_min = anchor_boxes_xyxy[:, 1:2]
    x2_max = anchor_boxes_xyxy[:, 2:3]
    y2_max = anchor_boxes_xyxy[:, 3:4]

    x1_min, y1_min, x1_max, y1_max = box_xyxy_gt
    x1_min = np.ones_like(x2_min) * x1_min
    y1_min = np.ones_like(x2_min) * y1_min
    x1_max = np.ones_like(x2_min) * x1_max
    y1_max = np.ones_like(x2_min) * y1_max

    i_min_x = np.maximum(x1_min, x2_min)
    i_min_y = np.maximum(y1_min, y2_min)
    i_max_x = np.minimum(x1_max, x2_max)
    i_max_y = np.minimum(y1_max, y2_max)

    inter_width = np.clip((i_max_x - i_min_x), 0, None)
    inter_height = np.clip((i_max_y - i_min_y), 0, None)

    width_box1 = (x1_max - x1_min)
    height_box1 = (y1_max - y1_min)

    width_box2 = (x2_max - x2_min)
    height_box2 = (y2_max - y2_min)

    area1 = np.abs(width_box1 * height_box1)
    area2 = np.abs(width_box2 * height_box2)
    intersection = (inter_width * inter_height)
    union = area1 + area2 - intersection
    iou = intersection / union

    iou[(iou < 0.0) | (iou > 1.0)] = 0.0

    return iou[:, 0]  # , i_min_x, i_min_y, i_max_x, i_max_y


# # creating default boxes

# In[3]:


from typing import List, Tuple

def create_default_boxes(
    image_size: Tuple[int, int],                 # target image dimension [height, width]
    grid_sizes: List[Tuple[int, int]],           # [(grid1.dim0, grid1.dim0), (grid2.dim0, grid2.dim1)]
    anchors_sizes: List[List[Tuple[float, float]]],     # [(box1.width, box1.height), (box2.width, box2.height)]
    normalize: bool = True
) -> Tuple[List[Tuple[float, float, float, float]],  # boxes_xyxy
           List[Tuple[float, float, float, float]],  # boxes_xywh
           List[Tuple[float, float, float, float]]]:  # boxes_cxcywh

    image_height, image_width = image_size[:2]

    boxes_xyxy = list()
    boxes_xywh = list()
    boxes_cxcywh = list()
    
    grid_idx = 0
    for grid_dim0, grid_dim1 in grid_sizes:
        grid_size_dim0 = image_height / float(grid_dim0)
        grid_size_dim1 = image_width / float(grid_dim1)

        for i_grid in range(grid_dim0):
            for j_grid in range(grid_dim1):
                for sx, sy in anchors_sizes[grid_idx]:
                    cy = grid_size_dim0 / 2.0 + grid_size_dim0 * i_grid
                    cx = grid_size_dim1 / 2.0 + grid_size_dim1 * j_grid
                    box_w = grid_size_dim1 * sx
                    box_h = grid_size_dim0 * sy

                    x_min = cx - box_w / 2.0
                    x_max = cx + box_w / 2.0
                    y_min = cy - box_h / 2.0
                    y_max = cy + box_h / 2.0

                    if normalize:
                        cx = cx / image_width
                        cy = cy / image_height
                        box_w = box_w / image_width
                        box_h = box_h / image_height
                        x_min = x_min / image_width
                        x_max = x_max / image_width
                        y_min = y_min / image_height
                        y_max = y_max / image_height

                    boxes_xyxy.append((x_min, y_min, x_max, y_max))
                    boxes_xywh.append((x_min, y_min, box_w, box_h))
                    boxes_cxcywh.append((cx, cy, box_w, box_h))
        
        grid_idx += 1

    return boxes_xyxy, boxes_xywh, boxes_cxcywh


# # creating dataset for training 

# In[4]:


import cv2
import os

import torch

import numpy as np
from tqdm import tqdm


class DatasetYOLOFormat(torch.utils.data.Dataset):

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        self.classes = list()
        self.bboxes = list()
        self.images_path = list()

        file_names = [file_name.replace(".txt", "") for file_name in sorted(
            os.listdir(folder_path)) if file_name.endswith(".txt")]
        for file_name in file_names:
            bbox = list()
            class_ids = list()
            file_path = os.path.join(folder_path, file_name + ".txt")
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    data = [float(x) for x in line.split(" ") if len(line.replace(" ", "")) > 0]

                    bbox.append(data[1:])
                    class_ids.append([int(data[0])])

            file_path = os.path.join(folder_path, file_name + ".jpg").replace(os.sep, "/")

            self.bboxes.append(bbox)
            self.classes.append(class_ids)
            self.images_path.append(file_path)

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.images_path[idx])
        bbox = self.bboxes[idx]
        class_id = self.classes[idx]

        return img, bbox, class_id


# SSD Dataset

def gen_training_data(default_boxes_xywh, default_boxes_xyxy, labels_id, boxes_gt_cxcywh):
    output_offsets = np.zeros_like(default_boxes_xywh)
    output_class_ids = np.zeros((output_offsets.shape[0], 1), dtype=np.uint8)

    for class_id, box_gt_cxcywh in zip(labels_id, boxes_gt_cxcywh):
        class_id = int(class_id[0]) + 1

        cx, cy, bw, bh = box_gt_cxcywh[:]
        bx, by, x1_max, y1_max = convert_cxcywh_to_xyxy(cx, cy, bw, bh)

        iou = np_calc_iou((bx, by, x1_max, y1_max), default_boxes_xyxy)
        mask_iou_high = iou > 0.9

        if np.sum(mask_iou_high) > 0:
            output_class_ids[mask_iou_high, 0] = class_id

            Dx = default_boxes_xywh[mask_iou_high, 0]
            Dy = default_boxes_xywh[mask_iou_high, 1]
            Dw = default_boxes_xywh[mask_iou_high, 2]
            Dh = default_boxes_xywh[mask_iou_high, 3]

            # Bx, By, Bw, Bh
            output_offsets[mask_iou_high, 0] = (bx - Dx) / Dw
            output_offsets[mask_iou_high, 1] = (by - Dy) / Dh
            output_offsets[mask_iou_high, 2] = np.log(bw / Dw)
            output_offsets[mask_iou_high, 3] = np.log(bh / Dh)

    return output_offsets, output_class_ids


class DatasetSSDTrain(torch.utils.data.Dataset):

    def __init__(self, folder_path_yolo_dataset, *,
                 default_boxes_xywh: np.ndarray,
                 default_boxes_xyxy: np.ndarray, transform=None):
        super().__init__()

        self.folder_path = folder_path_yolo_dataset
        self.transform = transform

        self.classes = list()
        self.bboxes = list()
        self.images_path = list()

        file_names = [file_name.replace(".txt", "") for file_name in sorted(
            os.listdir(self.folder_path)) if file_name.endswith(".txt")]
        for file_name in file_names:
            bbox = list()
            class_ids = list()
            file_path = os.path.join(self.folder_path, file_name + ".txt")
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    data = [float(x) for x in line.split(" ") if len(line.replace(" ", "")) > 0]

                    bbox.append(data[1:])
                    class_ids.append([int(data[0])])

            file_path = os.path.join(self.folder_path, file_name + ".jpg").replace(os.sep, "/")

            self.bboxes.append(bbox)
            self.classes.append(class_ids)
            self.images_path.append(file_path)

        self.list_offsets = list()
        self.list_classes = list()

        for label_idx in tqdm(range(len(self.images_path))):

            boxes_gt_cxcywh, labels_id = self.bboxes[label_idx], self.classes[label_idx]

            output_offsets, output_class_ids = gen_training_data(
                default_boxes_xywh, default_boxes_xyxy, labels_id, boxes_gt_cxcywh
            )

            self.list_offsets.append(output_offsets)
            self.list_classes.append(output_class_ids)

    def __len__(self) -> int:
        # return 1
        return len(self.images_path)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.images_path[idx])

        boxes = self.list_offsets[idx]
        classes = self.list_classes[idx]

        if self.transform is not None:
            img = self.transform(img)
            boxes = torch.from_numpy(boxes)
            classes = torch.from_numpy(classes.reshape((-1,)))

        return img, boxes, classes


# # generating prior boxes

# In[5]:


boxes_xyxy, boxes_xywh, boxes_cxcywh = create_default_boxes(
    (300, 300), 
    [ 
        (38, 38),
        (19, 19),
        (10, 10),
        (5, 5),
        (3, 3),
        (1, 1),
    ],
    [
        [(1.0, 1.0), (2.0, 1.0), (1.0,2.0), (1.0, 1.0),],
        [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0), (1.0,3.0), (1.0, 1.0)],
        [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0), (1.0,3.0), (1.0, 1.0)],
        [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0), (1.0,3.0), (1.0, 1.0)],
        [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (1.0, 1.0),],
        [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (1.0, 1.0),],
    ]
)

default_boxes_xyxy = np.array(boxes_xyxy)
default_boxes_xywh = np.array(boxes_xywh)


# #  loading dataset

# In[6]:


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300, 300)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = DatasetSSDTrain(
    "F:\project folder\Car_LicensePlate_Dataset",
    default_boxes_xywh=default_boxes_xywh,
    default_boxes_xyxy=default_boxes_xyxy,
    transform=train_transform
)


# In[7]:


#train_set, test_Set = torch.utils.data.random_split(dataset, [5,0])
train_set, test_Set = torch.utils.data.random_split(dataset, [500,206])


# In[8]:


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10, shuffle=True, num_workers=0
)
test_loader=torch.utils.data.DataLoader(
    test_Set, batch_size=1, shuffle=True, num_workers=0
)


# # model

# In[9]:


class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=3,   out_channels=64,  kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)
        self.conv3  = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.conv4  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8  = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9  = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1= nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        self.conv14 = nn.Conv2d(512, 1024,kernel_size=3,padding=6,dilation=6)
        self.conv15 = nn.Conv2d(1024, 1024,kernel_size=1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        head1=x
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        head2=x
        return head1,head2


# In[10]:


class Auxilarylayers(nn.Module):
    def __init__(self):
        super(Auxilarylayers,self).__init__()
        self.conv16 =nn.Conv2d(1024,256,kernel_size=1,padding=0)
        self.conv17 =nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)
        
        self.conv18 =nn.Conv2d(512,128,kernel_size=1,padding=0)
        self.conv19 =nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        
        self.conv20 =nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.conv21 =nn.Conv2d(128,256,kernel_size=3,padding=0)
        
        self.conv22 =nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.conv23 =nn.Conv2d(128,256,kernel_size=3,padding=0)

    def forward(self,head2):
        x =F.relu(self.conv16(head2))
        x =F.relu(self.conv17(x))
        head3=x
        x =F.relu(self.conv18(x))
        x =F.relu(self.conv19(x))
        head4=x
        x =F.relu(self.conv20(x))
        x =F.relu(self.conv21(x))
        head5=x
        x =F.relu(self.conv22(x))
        x =F.relu(self.conv23(x))
        head6=x
        
        return head3,head4,head5,head6
  


# In[11]:


class Predection(nn.Module):
    def __init__(self,classes):
        super(Predection,self).__init__()
        self.classes=classes
        self.Head1=nn.Conv2d(512,4*4,kernel_size=3, padding=1)
        self.Head2=nn.Conv2d(1024,6*4,kernel_size=3, padding=1)
        self.Head3=nn.Conv2d(512,6*4,kernel_size=3, padding=1)
        self.Head4=nn.Conv2d(256,6*4,kernel_size=3, padding=1)
        self.Head5=nn.Conv2d(256,4*4,kernel_size=3, padding=1)
        self.Head6=nn.Conv2d(256,4*4,kernel_size=3, padding=1)
        
        self.class1=  nn.Conv2d(512,  4 * classes, kernel_size=3, padding=1)
        self.class2=  nn.Conv2d(1024, 6 * classes, kernel_size=3, padding=1)
        self.class3 = nn.Conv2d(512,  6 * classes, kernel_size=3, padding=1)
        self.class4 = nn.Conv2d(256,  6 * classes, kernel_size=3, padding=1)
        self.class5 = nn.Conv2d(256,  4 * classes, kernel_size=3, padding=1)
        self.class6 = nn.Conv2d(256,  4 * classes, kernel_size=3, padding=1)

    def forward(self,head1,head2,head3,head4,head5,head6):
        #size=head1.size(0)
        box1=self.Head1(head1)
        box1=box1.permute(0,2,3,1).contiguous()
        box1=box1.view(box1.size(0),-1,4)
        
        
        class1 = self.class1(head1)
        class1 = class1.permute(0, 2, 3, 1).contiguous()
        class1 = class1.view(class1.size(0),-1,self.classes)
        
        
        
        box2=self.Head2(head2)
        box2=box2.permute(0,2,3,1).contiguous()
        box2=box2.view(box2.size(0),-1,4)
        
        class2 = self.class2(head2)
        class2 = class2.permute(0, 2, 3, 1).contiguous()
        class2 = class2.view(class2.size(0),-1,self.classes)
        
        box3=self.Head3(head3)
        box3=box3.permute(0,2,3,1).contiguous()
        box3=box3.reshape(box3.size(0),-1,4)
        
        class3 = self.class3(head3)
        class3 = class3.permute(0, 2, 3, 1).contiguous()
        class3 = class3.view(class3.size(0),-1,self.classes)
        
        box4=self.Head4(head4)
        box4=box4.permute(0,2,3,1).contiguous()
        box4=box4.view(box4.size(0),-1,4)
        
        
        class4 = self.class4(head4)
        class4 = class4.permute(0, 2, 3, 1).contiguous()
        class4 = class4.view(class4.size(0),-1,self.classes)
        
        box5=self.Head5(head5)
        box5=box5.permute(0,2,3,1).contiguous()
        box5=box5.view(box5.size(0),-1,4)
        
        
        class5 = self.class5(head5)
        class5 = class5.permute(0, 2, 3, 1).contiguous()
        class5 = class5.view(class5.size(0),-1,self.classes)
        
        
        box6=self.Head6(head6)
        box6=box6.permute(0,2,3,1).contiguous()
        box6=box6.view(box6.size(0),-1,4)
        
        
        class6 = self.class6(head6)
        class6 = class6.permute(0, 2, 3, 1).contiguous()
        class6 = class6.view(class6.size(0),-1,self.classes)
        
        
        boxes=torch.cat([box1,box2,box3,box4,box5,box6],dim=1)
        classess=torch.cat([class1,class2,class3,class4,class5,class6],dim=1)
        return boxes,classess


# In[12]:


class ssd(nn.Module):
    def __init__(self,classes):
        super(ssd, self).__init__()
        self.classes=classes
        self.vgg=VGG16_NET()
        self.auxilary=Auxilarylayers()
        self.predection=Predection(classes)
    def forward(self,image):
        head1,head2=self.vgg(image)
        head3,head4,head5,head6=self.auxilary(head2)
        boxes,classes=self.predection(head1,head2,head3,head4,head5,head6)
        return boxes,classes


# In[13]:


model=ssd(3)


# In[14]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# # loss function

# In[15]:


def compute_loss(output_classes, target_classes, output_boxes, target_boxes, device):
    
    n_batch = output_classes.shape[0]
    
    alpha=1.0
    gamma=3.0
    
    positive_loss = 0
    negative_loss = 0
    bbox_loss = 0
    for i_batch in range(n_batch):
        
        positive_mask = target_classes[i_batch][:] > 0
        negative_mask = target_classes[i_batch][:] == 0
        print(torch.sum(positive_mask))

        if torch.sum(positive_mask) > 0:
            bbox_loss += F.smooth_l1_loss(output_boxes[i_batch][positive_mask], target_boxes[i_batch][positive_mask], reduction='mean')
            
            positive_loss += F.cross_entropy(output_classes[i_batch][positive_mask], target_classes[i_batch][positive_mask], ignore_index=255, reduction='mean', label_smoothing=0.0) / float(n_batch)
            
            # ce_loss = F.cross_entropy(output_classes[i_batch][positive_mask], target_classes[i_batch][positive_mask], ignore_index=255, reduction='none', label_smoothing=0.0) 
            # pt = torch.exp(-ce_loss)
            # focal_loss = alpha * (1-pt)**gamma * ce_loss
            # positive_loss += focal_loss.mean() / float(n_batch)

        if torch.sum(negative_mask) > 0:
            negative_loss += F.cross_entropy(output_classes[i_batch][negative_mask], target_classes[i_batch][negative_mask], ignore_index=255, reduction='mean', label_smoothing=0.0) / float(n_batch)
            
            # ce_loss = F.cross_entropy(output_classes[i_batch][negative_mask], target_classes[i_batch][negative_mask], ignore_index=255, reduction='none', label_smoothing=0.0) 
            # pt = torch.exp(-ce_loss)
            # focal_loss = alpha * (1-pt)**gamma * ce_loss
            # negative_loss += focal_loss.mean() / float(n_batch)

    #print("\nbbox_loss:", bbox_loss.item() * 2)
    #print("positive_loss:", positive_loss.item() * 2)
    #print("negative_loss:", negative_loss.item() * 0.1)
    return positive_loss * 2 + negative_loss * 0.1 + bbox_loss * 4


def compute_loss2(output_classes, target_classes, output_boxes, target_boxes, device):
    
    output_classes = output_classes.view(-1, 3)
    target_classes = target_classes.view(-1)
    
    output_boxes = output_boxes.view(-1, 4)
    target_boxes = target_boxes.view(-1, 4)
    
    positive_loss = 0
    negative_loss = 0
    bbox_loss = 0
        
    positive_mask = target_classes > 0
    negative_mask = target_classes == 0

    if torch.sum(positive_mask) > 0:
        bbox_loss = F.smooth_l1_loss(output_boxes, target_boxes, reduction='sum')
        positive_loss = F.cross_entropy(output_classes, target_classes, ignore_index=255, reduction='sum', label_smoothing=0.0) 

    if torch.sum(negative_mask) > 0:
        negative_loss = F.cross_entropy(output_classes, target_classes, ignore_index=255, reduction='sum', label_smoothing=0.0) 

    #print("\nbbox_loss:", bbox_loss.item() * 2)
    #print("positive_loss:", positive_loss.item() * 2)
    #print("negative_loss:", negative_loss.item() * 0.1)
    return positive_loss * 2 + negative_loss * 0.1 + bbox_loss * 4



# optimizer = optim.Adam(model.parameters(), lr=0.01)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# In[18]:


device=torch.device('cuda')
model=model.to(device)


# In[19]:


import torch


# In[20]:


from jupyterplot import ProgressPlot


# In[25]:


lowest_loss = None
init_epoch = 0
LOAD_PRETRAINED_MODEL =True
if LOAD_PRETRAINED_MODEL:
    model_path = "E:\model_scripted.pt"
    model = torch.jit.load(model_path)

model.eval()
dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0
)

# idx = 0
running_loss = 0
for input_tensor, target_boxes, target_classes in (dataset_loader):
    # idx += 1
    # if idx < 87:
    #     continue
    
    input_tensor = input_tensor.float().to(device)
    target_boxes = target_boxes.float().to(device)
    target_classes = target_classes.to(device, dtype=torch.long)
    

        
        
    with torch.no_grad():
        output_boxes, output_classes = model(input_tensor)

        loss = compute_loss(output_classes, target_classes, output_boxes, target_boxes, device)
        running_loss += loss.item() / 1.0

    break

print("epoch", init_epoch,
      '; Loss: ', running_loss)


# In[26]:


out_preds, out_class = F.softmax(output_classes, dim=2).squeeze(0).max(1)
# out_class = output_classes.squeeze(0).max(1).detach().cpu().numpy()

out_preds = out_preds.detach().cpu().numpy()
out_class = out_class.detach().cpu().numpy()
out_boxes = output_boxes.squeeze(0).detach().cpu().numpy()

print(out_class.shape)
plt.plot(out_preds[out_class != 0],'.')
plt.show()


# In[27]:


img_draw = np.clip((input_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255),0, 255).astype(np.uint8).copy(order='C')

# img_draw = np.zeros(img_draw.shape, dtype=np.uint8)
# print(img_draw.shape)
#out_preds, out_class

for box_id in range(default_boxes_xywh.shape[0]):

    if int(out_class[box_id]) != 2 or out_preds[box_id] < 0.9:
        continue
    
    Bx, By, Bw, Bh = out_boxes[box_id, :]
    Dx, Dy, Dw, Dh = default_boxes_xywh[box_id, :]    
    bx = Dx + Dw * Bx
    by = Dy + Dh * By
    bw = Dw * np.exp(Bw)
    bh = Dh * np.exp(Bh)

    x1, y1, x2, y2 = convert_xywh_to_xyxy(bx, by, bw, bh)
    x1, y1, x2, y2 = convert_normalized_to_xyxy(x1, y1, x2, y2, img_draw.shape[1], img_draw.shape[0])
    cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    
plt.figure(figsize=(10,10))
plt.imshow(img_draw[:,:,::-1])
plt.show()


# In[28]:


final_bboxes = list()

for class_id in [1, 2]:
    mask = (out_class == class_id) * (out_preds > 0.999)
    if np.sum(mask) > 0:
        sel_boxes, sel_preds, sel_default_boxes = out_boxes[mask, :][:], out_preds[mask][:], default_boxes_xywh[mask, :][:]
        sort_index = np.argsort(sel_preds)[::-1]
        sel_boxes = sel_boxes[sort_index]
        sel_preds = sel_preds[sort_index]
        sel_default_boxes = sel_default_boxes[sort_index]
        
        # convert all offsets to bbox_xyxy
        
        for box_id in range(sel_boxes.shape[0]):
            Bx, By, Bw, Bh = sel_boxes[box_id, :]
            Dx, Dy, Dw, Dh = sel_default_boxes[box_id, :]    
            bx = Dx + Dw * Bx
            by = Dy + Dh * By
            bw = Dw * np.exp(Bw)
            bh = Dh * np.exp(Bh)
            x1_min, y1_min, x1_max, y1_max = convert_xywh_to_xyxy(bx, by, bw, bh)
            sel_boxes[box_id, :] = x1_min, y1_min, x1_max, y1_max
        
        while sel_preds.shape[0] > 0:
            # print(sel_preds.shape[0])
            
            # get current bbox
            x1_min, y1_min, x1_max, y1_max = sel_boxes[0, :]
            final_bboxes.append((x1_min, y1_min, x1_max, y1_max))
            
            iou = np_calc_iou((x1_min, y1_min, x1_max, y1_max), sel_boxes)
            mask_iou_high = iou < 0.1
            
            sel_boxes = sel_boxes[mask_iou_high]
            sel_preds = sel_preds[mask_iou_high]
            

print(len(final_bboxes))
print(sel_preds)


img_draw = np.clip((input_tensor[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255),0, 255).astype(np.uint8).copy(order='C')

for x1, y1, x2, y2 in final_bboxes:

    x1, y1, x2, y2 = convert_normalized_to_xyxy(x1, y1, x2, y2, img_draw.shape[1], img_draw.shape[0])
    cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    #cv2.putText(img_draw,"car'",(int(x1), int(y1)-10),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),1)
plt.figure(figsize=(10,10))
plt.imshow(img_draw[:,:,::-1])
plt.show()


# In[ ]:




