import numpy as np

from .image_process import cmap
import torch


def cal_type_iou(img1, img2, type):
    color = cmap[type]*255
    img1=img1.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img2=img2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    diff1 = np.abs(img1 - color)
    i1=(diff1<3).all(axis=-1)
    # Image.fromarray(i1).save(f"i1_{type}.png")
    diff2 = np.abs(img2 - color)
    i2=(diff2<3).all(axis=-1)
    # Image.fromarray(i2).save(f"i2_{type}.png")
    intersection = np.logical_and(i1, i2)
    union = np.logical_or(i1, i2)
    intersection = np.sum(intersection)
    union = np.sum(union)
    return intersection, union

def cal_iou(img1, img2):
    intersections=[]
    unions=[]
    ious=[]
    for i in range(8):
        intersection, union = cal_type_iou(img1, img2, i)
        if union!=0:
            iou = intersection / union
            intersections.append(intersection)
            unions.append(union)
            ious.append(iou)
    micro_iou = np.sum(intersections) / np.sum(unions)
    macro_iou = np.mean(ious)
    return micro_iou, macro_iou
    