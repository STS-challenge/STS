
import argparse
import json
import os
import time
import psutil as p
from multiprocessing import Manager, Process
from pynvml.smi import nvidia_smi
import numpy as np
from utils import split_base_and_extension
from logger import add_file_handler_to_logger, logger

manager = Manager()
gpu_list = manager.list()
cpu_list = manager.list()
RAM_list = manager.list()


def cpu_usage():
    t = p.cpu_times()
    return [t.user, t.system, t.idle]


before = cpu_usage()


def get_cpu_usage():
    global before
    now = cpu_usage()
    delta = [now[i] - before[i] for i in range(len(now))]
    total = sum(delta)
    before = now
    return [(100.0*dt)/(total+0.1) for dt in delta]


def daemon_process(time_interval, gpu_id=0):
    while True:
        nvsmi = nvidia_smi.getInstance()
        dictm = nvsmi.DeviceQuery('memory.free, memory.total')
        gpu_memory = (
                dictm['gpu'][gpu_id]['fb_memory_usage']['total'] - dictm['gpu'][gpu_id]['fb_memory_usage']['free']
        )
        cpu_usage = get_cpu_usage()
        RAM = p.virtual_memory().used/1048576  # 1048576 = 1024 * 1024
        gpu_list.append(gpu_memory)
        RAM_list.append(RAM)
        cpu_list.append(cpu_usage)
        time.sleep(time_interval)


def save_result(start_time, json_path, gpu_list_, cpu_list_copy, RAM_list_copy):
    js = dict()
    with open(json_path, mode='w', encoding='utf-8') as f:
        js['gpu_memory'] = gpu_list_
        js['cpu_list'] = cpu_list_copy
        js['RAM_list'] = RAM_list_copy
        json.dump(js, f, indent=4)

    infer_time = time.time() - start_time
    with open(json_path, mode='r', encoding='utf-8') as f:
        js = json.load(f)
    with open(json_path, mode='w', encoding='utf-8') as f:
        js['time'] = infer_time
        json.dump(js, f, indent=4)
    time.sleep(2)
    logger.info('save result end')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-time_interval', default=0.1, help='GPU usage sampling interval')
    parser.add_argument('-sleep_time', default=5, help='sleep time')
    parser.add_argument('-shell_path', default='/opt/algorithm/predict.sh', help='predict shell path')
    # XXX: in case of someone use lower docker, please use specified GPU !!!
    parser.add_argument('-gpu_id', default=0, help='CUDA_VISIBLE_DEVICE')
    parser.add_argument('-docker_input_folder', default='/sts24/inputs/', help='docker input folder')
    parser.add_argument('-docker_output_folder', default='/sts24/outputs/', help='docker output folder')
    parser.add_argument('-docker_result_folder', default='/sts24/test_results/algorithm', help='docker result folder')
    parser.add_argument('-docker_name', default='algorithm', help='docker name')
    args = parser.parse_args()
    logger.info(f'We are evaluating {args.docker_name}')

    # GPU area json save path
    file_tag, _ = split_base_and_extension(os.listdir(args.docker_input_folder)[0])
    json_path = os.path.join(args.docker_result_folder, '{}.json'.format(file_tag))

    # logger
    add_file_handler_to_logger(name='main', dir_path=os.path.join(args.docker_result_folder, 'logs'), level='DEBUG')

    try:
        # Start record
        p1 = Process(target=daemon_process, args=(args.time_interval, args.gpu_id,))
        p1.daemon = True
        p1.start()
        start_time = time.time()
        # Run docker container
        cmd = 'docker run --network="none" --security-opt="no-new-privileges" --shm-size=28gb -m 28G --gpus="device={}" --name {} --rm -v {}:/inputs/ -v {}:/outputs/ {}:latest /bin/bash -c "sh {}" '.format(
            args.gpu_id,
            args.docker_name,
            args.docker_input_folder,
            args.docker_output_folder,
            args.docker_name,
            args.shell_path)
        logger.info(f'cmd is : {cmd}')
        logger.info('start predict...')
        os.system(cmd)
        # Save result
        RAM_list = list(RAM_list)
        RAM_list_copy = RAM_list.copy()
        cpu_list = list(cpu_list)
        cpu_list_copy = cpu_list.copy()
        gpu_list = list(gpu_list)
        gpu_list_copy = gpu_list.copy()
        logger.info(json_path)
        save_result(start_time, json_path, gpu_list_copy, cpu_list_copy, RAM_list_copy)
        # Avoid premature exit
        time.sleep(5)
    except Exception as error:
        logger.exception(error)

