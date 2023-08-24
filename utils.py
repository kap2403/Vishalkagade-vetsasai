import numpy as np
from typing import Tuple, Union
import torch
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# creating dataset for training 

import cv2
import os

import torch

import numpy as np
from tqdm import tqdm


# SSD Dataset
def gen_training_data(default_boxes_xywh, default_boxes_xyxy, labels_id, boxes_gt_cxcywh):
    output_offsets = np.zeros_like(default_boxes_xywh)
    output_class_ids = np.zeros((output_offsets.shape[0], 1), dtype=np.uint8)

    for class_id, box_gt_cxcywh in zip(labels_id, boxes_gt_cxcywh):
        class_id = int(class_id[0]) + 1

        cx, cy, bw, bh = box_gt_cxcywh[:]
        bx, by, x1_max, y1_max = convert_cxcywh_to_xyxy(cx, cy, bw, bh)

        iou = np_calc_iou((bx, by, x1_max, y1_max), default_boxes_xyxy)
        if class_id ==0:
            mask_iou_high = iou > 0.1

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
        elif class_id ==1:
            mask_iou_high = iou > 0.1

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

