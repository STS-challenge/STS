
import os
import shutil
import time
import torch
from logger import add_file_handler_to_logger, logger
from utils import dir_is_empty, split_base_and_extension


docker_path = '/root/sts24/team_docker/'        # put docker in this folder
test_image_path = '/root/sts24/images'          # all testing set cases
save_path = '/root/sts24/test_results/'         # evaluation results will be saved in this folder
os.makedirs(save_path, exist_ok=True)

# docker input and output
docker_input_folder = '/root/sts24/inputs/'
docker_output_folder = '/root/sts24/outputs'
os.makedirs(docker_input_folder, exist_ok=True)
os.makedirs(docker_output_folder, exist_ok=True)
# folder authorization to ensure that the prediction results can be saved
os.system('chmod -R 777 {}'.format(docker_output_folder))


# Team docker
team_docker = [x for x in os.listdir(docker_path) if x.endswith('.tar.gz')][0]
team_docker_name = team_docker.split('.')[0].lower()

try:
    # load docker
    print('loading docker:', team_docker_name)
    cmd = 'docker load -i {}'.format(os.path.join(docker_path, team_docker))
    os.system(cmd)
    time.sleep(5)

    # create result save dir
    team_result_folder = os.path.join(save_path, team_docker_name)
    if os.path.exists(team_result_folder):
        shutil.rmtree(team_result_folder)
    os.mkdir(team_result_folder)

    # logger
    add_file_handler_to_logger(name='main', dir_path=os.path.join(team_result_folder, 'logs'), level='DEBUG')

    # test for each case
    test_cases = sorted(os.listdir(test_image_path))
    for case in test_cases:
        if not dir_is_empty(docker_input_folder):
            logger.error('please check inputs folder', docker_input_folder)
            raise ValueError('please check inputs folder')
        # copy one case to input folder
        shutil.copy(os.path.join(test_image_path, case), docker_input_folder)
        # test one case
        start_time = time.time()
        os.system('python run_docker.py -docker_name {} -docker_input_folder {} -docker_output_folder {} -docker_result_folder {}'.format(
                 team_docker_name,
                 docker_input_folder,
                 docker_output_folder,
                 team_result_folder
        ))
        logger.info(f'{case} finished!')
        logger.info(f'{case} cost time: {time.time() - start_time}')
        # remove the tested case
        os.remove(os.path.join(docker_input_folder, case))

        # parse segmentation case file name
        case_base, case_extension = split_base_and_extension(case)
        if case_extension == '.nii.gz':
            segmentation_case_extension = '.nii.gz'
        elif case_extension == '.jpg' or case_extension == '.png':
            segmentation_case_extension = '.json'
        else:
            raise ValueError('Please check input extension')
        segmentation_case_name = case_base + '_Mask' + segmentation_case_extension

        # move segmentation file
        if os.path.exists(os.path.join(docker_output_folder, segmentation_case_name)):
            os.rename(os.path.join(docker_output_folder, segmentation_case_name), os.path.join(team_result_folder, segmentation_case_name))

    # calculate efficiency-related metrics
    os.system("python load_json.py -docker_name {} -docker_result_folder {}".format(team_docker_name, team_result_folder))

    # keep docker output folder
    shutil.move(docker_output_folder, team_result_folder)
    os.mkdir(docker_output_folder)

    #
    torch.cuda.empty_cache()
    shutil.rmtree(docker_input_folder)
    os.mkdir(docker_input_folder)

    # remove docker image
    os.system("docker rmi {}:latest".format(team_docker_name))
except Exception as e:
    logger.exception(e)
