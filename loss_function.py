from utils import optim
from utils import model
import cv2

def compute_loss(output_classes, target_classes, output_boxes, target_boxes, device):
    
    n_batch = output_classes.shape[0]
    
    positive_loss = 0.0
    negative_loss = 0.0
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

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)