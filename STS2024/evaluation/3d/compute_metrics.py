
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import multiprocessing as mp
from collections import OrderedDict
from SurfaceDice import (compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient,
                         compute_iou_score)


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


def compute_metrics(niigz_name):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(seg_dir, niigz_name)))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir, niigz_name)))

    # image-level metrics
    image_dsc = compute_dice_coefficient(gt != 0, seg != 0)
    image_iou = compute_iou_score(gt != 0, seg != 0)
    surface_distance = compute_surface_distances(gt != 0, seg != 0, spacing_mm=[1, 1, 1])
    image_nsd = compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2)

    # instance-level metrics
    instance_dsc = compute_multi_class_dsc(gt, seg)
    instance_nsd = compute_multi_class_nsd(gt, seg, spacing=[1, 1, 1])
    # TP means iou > 0.5 and class is equal
    TP = 0
    iou_list = []
    for i in np.unique(gt):
        if i == 0:
            continue
        iou = compute_iou_score(gt == i, seg == i)
        iou_list.append(iou)
        if iou > 0.5:
            TP += 1
    # instance-level iou
    instance_iou = np.mean(iou_list)
    # instance-level ia, -1 means the background 0. IA is a bit like class-level IoU
    ia = TP / (len(list(set(np.unique(gt)).union(set(np.unique(seg))))) - 1)
    return niigz_name, image_dsc, image_iou, image_nsd, instance_dsc, instance_iou, instance_nsd, ia


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
    
    niigz_names = [name for name in os.listdir(gt_dir) if name.endswith('nii.gz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(niigz_names)) as pbar:
            for i, (niigz_name,
                    image_dsc,
                    image_iou,
                    image_nsd,
                    instance_dsc,
                    instance_iou,
                    instance_nsd,
                    ia) in enumerate(pool.imap_unordered(compute_metrics, niigz_names)):
                seg_metrics['case'].append(niigz_name)
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
