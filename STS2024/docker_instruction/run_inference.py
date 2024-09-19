import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

# TODO: import your model
from model.UNet import UNet


INPUT_DIR = '/inputs'
OUTPUT_DIR = '/outputs'


def main():
    # config device
    device = torch.device('cuda:0')
    # load model and checkpoint file
    unet_model = UNet(in_channels=1, n_class=32).to(device).float()
    # unet_model.load_state_dict(torch.load('unet_weights.pth', map_location=device))
    with torch.no_grad():
        unet_model.eval()
        # load the current case since there would only be one case in the INPUT_DIR
        case_name = os.listdir(INPUT_DIR)[0]
        print(case_name)
        # load image as numpy array
        case_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_DIR, case_name)))
        # image pre-processing
        original_shape = case_image.shape
        target_shape = (128, 128, 128)
        zoom_factors = (target_shape[0] / original_shape[0],
                        target_shape[1] / original_shape[1],
                        target_shape[2] / original_shape[2])
        zoom_image = zoom(case_image, zoom_factors, order=1)
        zoom_image = (zoom_image - zoom_image.min()) / (zoom_image.max() - zoom_image.min() + 1e-8)
        # model inference
        batch_image = zoom_image[np.newaxis, np.newaxis, :, :, :]
        batch_tensor = torch.from_numpy(batch_image).to(device).float()
        batch_prediction = unet_model(batch_tensor)
        # prediction pro-processing
        zoom_prediction = batch_prediction.cpu().numpy()[0, 0, :, :, :]
        upsample_factors = (1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2])
        case_prediction = zoom(zoom_prediction, upsample_factors, order=1)
        case_prediction[case_prediction >= 0.5] = 1
        case_prediction[case_prediction <= 0.5] = 0
        # prediction save
        sitk_prediction = sitk.GetImageFromArray(case_prediction)
        case_tag = case_name.split('.')[0]
        sitk.WriteImage(sitk_prediction, os.path.join(OUTPUT_DIR, '%s_Mask.nii.gz' % case_tag))


if __name__ == "__main__":
    main()

