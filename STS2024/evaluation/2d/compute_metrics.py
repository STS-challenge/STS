
import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import multiprocessing as mp
from collections import OrderedDict
from SurfaceDice import (compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient,
                         compute_iou_score)


def labelme_to_mask(points, image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [points], (1))
    return mask


def compute_multi_class_iou(gt, seg):
    iou = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        iou.append(compute_iou_score(gt_i, seg_i))
    return np.mean(iou)


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=5)
args = parser.parse_args()


seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
num_workers = args.num_workers


def compute_metrics(json_name):
    with open(os.path.join(gt_dir, json_name), mode='r', encoding='utf-8') as f:
        gt_data = json.load(f)

    with open(os.path.join(seg_dir, json_name), mode='r', encoding='utf-8') as f1:
        seg_data = json.load(f1)

    # define dict
    gt_dict = {}
    seg_dict = {}

    # step1： gt_data transfer to mask dict, class label is the dict key
    for i in range(len(gt_data['shapes'])):
        instance_label = gt_data['shapes'][i]['label']
        gt_points = np.array(gt_data['shapes'][i]['points'], dtype=np.int32)
        image_Height = gt_data['imageHeight']
        image_Width = gt_data['imageWidth']
        instance_mask = labelme_to_mask(gt_points, image_Height, image_Width)
        # avoid multi instances having same label
        same_instance_mask = gt_dict.get(instance_label, instance_mask)
        same_instance_mask[instance_mask == 1] = 1
        gt_dict[instance_label] = same_instance_mask

    # step1： seg_data transfer to mask dict, class label is the dict key
    for i in range(len(seg_data['shapes'])):
        instance_label = seg_data['shapes'][i]['label']
        seg_points = np.array(seg_data['shapes'][i]['points'], dtype=np.int32)
        image_Height = seg_data['imageHeight']
        image_Width = seg_data['imageWidth']
        instance_mask = labelme_to_mask(seg_points, image_Height, image_Width)
        # avoid multi instances having same label
        same_instance_mask = seg_dict.get(instance_label, instance_mask)
        same_instance_mask[instance_mask == 1] = 1
        seg_dict[instance_label] = same_instance_mask

    # step3： image-level metrics
    gt_stack_mask = np.zeros((gt_data['imageHeight'], gt_data['imageWidth']), dtype=np.uint8)
    for instance_class, instance_mask in gt_dict.items():
        gt_stack_mask[instance_mask != 0] = 1

    seg_stack_mask = np.zeros((seg_data['imageHeight'], seg_data['imageWidth']), dtype=np.uint8)
    for instance_class, instance_mask in seg_dict.items():
        seg_stack_mask[instance_mask != 0] = 1

    # image-level metrics
    image_dsc = compute_dice_coefficient(gt_stack_mask != 0, seg_stack_mask != 0)
    image_iou = compute_iou_score(gt_stack_mask != 0, seg_stack_mask != 0)
    # newaxis： 2d -> 3d for compute_surface_distances function
    surface_distance = compute_surface_distances(gt_stack_mask[np.newaxis, ...] != 0,
                                                 seg_stack_mask[np.newaxis, ...] != 0,  spacing_mm=[1, 1, 1])
    image_nsd = compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2)

    # step4: instance-level metrics
    TP = 0
    instance_dsc_list, instance_iou_list, instance_nsd_list = [], [], []
    for gt_instance_class, gt_instance_mask in gt_dict.items():
        if gt_instance_class not in seg_dict.keys():
            seg_instance_mask = np.zeros_like(gt_instance_mask, dtype=np.uint8)
        else:
            seg_instance_mask = seg_dict[gt_instance_class]
        # metrics computation
        instance_dsc_list.append(compute_dice_coefficient(gt_instance_mask != 0, seg_instance_mask != 0))
        # newaxis： 2d -> 3d for compute_surface_distances function
        surface_distance = compute_surface_distances(gt_instance_mask[np.newaxis, ...] != 0,
                                                     seg_instance_mask[np.newaxis, ...] != 0, spacing_mm=[1, 1, 1])
        instance_nsd_list.append(compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2))
        iou = compute_iou_score(gt_instance_mask != 0, seg_instance_mask != 0)
        instance_iou_list.append(iou)
        if iou > 0.5:
            TP += 1
    instance_dsc = np.mean(instance_dsc_list)
    instance_iou = np.mean(instance_iou_list)
    instance_nsd = np.mean(instance_nsd_list)
    ia = TP / len(list(set(gt_dict.keys()).union(set(seg_dict.keys()))))

    return json_name, image_dsc, image_iou, image_nsd, instance_dsc, instance_iou, instance_nsd, ia


if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['image_DSC'] = []
    seg_metrics['image_IoU'] = []
    seg_metrics['image_NSD'] = []
    seg_metrics['instance_DSC'] = []
    seg_metrics['instance_IoU'] = []
    seg_metrics['instance_NSD'] = []
    seg_metrics['IA'] = []
    
    json_names = [name for name in os.listdir(gt_dir) if name.endswith('json')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(json_names)) as pbar:
            for i, (json_name,
                    image_dsc,
                    image_iou,
                    image_nsd,
                    instance_dsc,
                    instance_iou,
                    instance_nsd,
                    ia) in enumerate(pool.imap_unordered(compute_metrics, json_names)):
                seg_metrics['case'].append(json_name)
                seg_metrics['image_DSC'].append(np.round(image_dsc, 4))
                seg_metrics['image_IoU'].append(np.round(image_iou, 4))
                seg_metrics['image_NSD'].append(np.round(image_nsd, 4))
                seg_metrics['instance_DSC'].append(np.round(instance_dsc, 4))
                seg_metrics['instance_IoU'].append(np.round(instance_iou, 4))
                seg_metrics['instance_NSD'].append(np.round(instance_nsd, 4))
                seg_metrics['IA'].append(np.round(ia, 4))
                pbar.update()
    df = pd.DataFrame(seg_metrics)
    # rank based on case column
    df = df.sort_values(by=['case'])
    df.to_csv(csv_dir, index=False)
