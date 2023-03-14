from utils import DatasetSSDTrain
from model import ssd
from utils import create_default_boxes
from utils import gen_training_data
from utils import device
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
import argparse
parser = argparse.ArgumentParser(description ='Search some files')
parser.add_argument('file_directory' , help = 'images_and_labels_path')
parser.add_argument('number_of_classes' , help = 'no_of_classes
parser.add_argument('save_model' , help = 'path_for_model_to_save')
args = parser.parse_args()

file_path = args.file_directory
classes = args.number_of_classes
path = args.save_model



#conversion and Iou calculation functions

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



# Dataloader for images and bounding box

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300, 300)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = DatasetSSDTrain(
    file_path,
    default_boxes_xywh=default_boxes_xywh,
    default_boxes_xyxy=default_boxes_xyxy,
    transform=train_transform
)

print(len(dataset))

train_set, test_Set = torch.utils.data.random_split(dataset, [len(dataset)*0.7,len(dataset)*0.3])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10, shuffle=True, num_workers=0
)
test_loader=torch.utils.data.DataLoader(
    test_Set, batch_size=1, shuffle=True, num_workers=0
)


# model 


model = ssd(classes)
model=model.to(device)



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
    return positive_loss * 2 + negative_loss * 0.1 + bbox_loss * 4



# optimizer = optim.Adam(model.parameters(), lr=0.01)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)





for epoch in range(1):

    running_loss = 0.0
    for input_tensor, target_boxes, target_classes in tqdm(train_loader):
        input_tensor = input_tensor.float().to(device)
        target_boxes = target_boxes.float().to(device)
        target_classes = target_classes.view(-1).to(device, dtype=torch.long)
        optimizer.zero_grad()
        output_boxes, output_classes = model(input_tensor)
        loss = compute_loss2(output_classes, target_classes, output_boxes, target_boxes, device)
        running_loss += loss.item() / 100.0
        loss.backward()
        optimizer.step()
    print("epoch", epoch, running_loss)

checkpoint_path = path
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'train_loss': running_loss,
  }, checkpoint_path)