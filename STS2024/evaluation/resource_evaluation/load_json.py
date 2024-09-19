import json
import csv
import argparse
import matplotlib
import os
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from logger import add_file_handler_to_logger, logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-docker_name', default='algorithm', help='team docker name')
    parser.add_argument('-docker_result_folder', default='/sts24/test_results/algorithm', help='docker result folder')
    time_interval = 0.1
    args = parser.parse_args()

    # logger
    add_file_handler_to_logger(name='main', dir_path=os.path.join(args.docker_result_folder, 'logs'), level='DEBUG')
    logger.info('we are counting: {}'.format(args.docker_name))

    # result csv file path
    csv_path = os.path.join(args.docker_result_folder, args.docker_name + '_Efficiency.csv')

    # all case json file list
    json_list = sorted([x for x in os.listdir(args.docker_result_folder) if x.endswith('.json') and 'Mask' not in x])
    alldata = []
    for item in json_list:
        logger.info('calculating {}'.format(item))
        csv_l = []
        name = item.split('.')[0]
        csv_l.append(name)
        zitem = os.path.join(args.docker_result_folder, item)
        with open(zitem) as f:
            try:
                js = json.load(f)
            except Exception as error:
                logger.error(f'{item} have error')
                logger.exception(error)
            if 'time' not in js:
                logger.error(f"{item} don't have time!!!!")
                logger.info(f'Manually compute {item}')
                time = time_interval * len(js['gpu_memory'])
            else:
                time = js['time']
            csv_l.append(np.round(time, 2))
            # CPU
            user, system, all_cpu_used = [item[0] for item in js['cpu_list']], [item[1] for item in js['cpu_list']], [
                100 - item[2] for item in js['cpu_list']]
            plt.cla()
            x = [item * time_interval for item in range(len(user))]
            plt.xlabel('Time (s)', fontsize='large')
            plt.ylabel('CPU Utilization (%)', fontsize='large')
            plt.plot(x, all_cpu_used, 'b', ms=10, label='Used %')
            plt.legend()
            plt.savefig(zitem.replace('.json', '_CPU-Time.png'))
            # RAM
            RAM = js['RAM_list']
            plt.cla()
            x = [item * time_interval for item in range(len(RAM))]
            plt.xlabel('Time (s)', fontsize='large')
            plt.ylabel('RAM (MB)', fontsize='large')
            plt.plot(x, RAM, 'b', ms=10, label='RAM')
            plt.legend()
            plt.savefig(zitem.replace('.json', '_RAM-Time.png'))
            # GPU
            mem = js['gpu_memory']
            x = [item * time_interval for item in range(len(mem))]
            plt.cla()
            plt.xlabel('Time (s)', fontsize='large')
            plt.ylabel('GPU Memory (MB)', fontsize='large')
            plt.plot(x, mem, 'b', ms=10, label='a')
            plt.savefig(zitem.replace('.json', '_GPU-Time.png'), dpi=300)
            count_set = set(mem)
            max_mem = max(count_set)
            csv_l.append(np.round(max_mem))
            csv_l.append(np.round(sum(mem) * time_interval))
            csv_l.append(np.round(max(all_cpu_used), 2))
            csv_l.append(np.round(sum(all_cpu_used) * time_interval, 2))
            csv_l.append(np.round(max(RAM), 2))
            csv_l.append(np.round(sum(RAM) * time_interval))
        alldata.append(csv_l)

    # write result
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Name', 'Time', 'MaxGPU_Mem', 'AUC_GPU_Time', 'MaxCPU_Utilization', 'AUC_CPU_Time', 'MaxRAM',
                     'AUC_RAM_Time'])
    for i in alldata:
        writer.writerow(i)
    f.close()
