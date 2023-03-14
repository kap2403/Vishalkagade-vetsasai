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


from utils import device
from model import ssd
import argparse
parser = argparse.ArgumentParser(description ='Search some files')
parser.add_argument('file_directory' , help = 'images_and_labels_path')
parser.add_argument('number_of_classes' , help = 'no_of_classes')
parser.add_argument('model_path' , help = 'saved_model_path')

args = parser.parse_args()

file_path = args.file_directory
classes = args.number_of_classes
model_path = args.model_path


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

train_set, test_Set = torch.utils.data.random_split(dataset, [500,206])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10, shuffle=True, num_workers=0
)
test_loader=torch.utils.data.DataLoader(
    test_Set, batch_size=1, shuffle=True, num_workers=0
)


model = ssd(classes)
model=model.to(device)

def compute_loss(output_classes, target_classes, output_boxes, target_boxes, device):
    
    n_batch = output_classes.shape[0]
    
    positive_loss = 0
    negative_loss = 0
    bbox_loss = 0
    for i_batch in range(n_batch):
        
        positive_mask = target_classes[i_batch][:] > 0
        negative_mask = target_classes[i_batch][:] == 0


        if torch.sum(positive_mask) > 0:
            bbox_loss += F.smooth_l1_loss(output_boxes[i_batch][positive_mask], target_boxes[i_batch][positive_mask], reduction='mean')
            
            positive_loss += F.cross_entropy(output_classes[i_batch][positive_mask], target_classes[i_batch][positive_mask], ignore_index=255, reduction='mean', label_smoothing=0.0) / float(n_batch)


        if torch.sum(negative_mask) > 0:
            negative_loss += F.cross_entropy(output_classes[i_batch][negative_mask], target_classes[i_batch][negative_mask], ignore_index=255, reduction='mean', label_smoothing=0.0) / float(n_batch)
            
    return positive_loss * 2 + negative_loss * 0.1 + bbox_loss * 4

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


PATH = model_path
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])


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



out_preds, out_class = F.softmax(output_classes, dim=2).squeeze(0).max(1)
# out_class = output_classes.squeeze(0).max(1).detach().cpu().numpy()

out_preds = out_preds.detach().cpu().numpy()
out_class = out_class.detach().cpu().numpy()
out_boxes = output_boxes.squeeze(0).detach().cpu().numpy()

print(out_class.shape)
plt.plot(out_preds[out_class != 0],'.')
plt.show()


img_draw = np.clip((input_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255),0, 255).astype(np.uint8).copy(order='C')


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

final_bboxes = list()

for class_id in [1, 2]:
    mask = (out_class == class_id) * (out_preds > 0.1)
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
plt.figure(figsize=(10,10))
plt.imshow(img_draw[:,:,::-1])
plt.show()