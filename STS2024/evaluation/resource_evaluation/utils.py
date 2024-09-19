
import os


def dir_is_empty(dir_path):
    files = os.listdir(dir_path)
    if len(files) == 0:
        return True
    else:
        return False


def split_base_and_extension(filename):
    if filename.endswith('.nii.gz'):
        return filename[:-7], filename[-7:]
    elif filename.endswith('.jpg') or filename.endswith('.png'):
        return filename[:-4], filename[-4:]
    else:
        raise ValueError('Filename must end with .nii.gz or .jpg or .png')

