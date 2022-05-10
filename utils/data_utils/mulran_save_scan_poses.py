# Based on submap-extraction tools from: https://sites.google.com/view/mulran-pr/tool

import glob
import os
import numpy as np
import csv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.eval_config import get_config_eval
cfg = get_config_eval()
basedir = cfg.mulran_dir


def findNnPoseUsingTime(target_time, all_times, data_poses):
    time_diff = np.abs(all_times - target_time)
    nn_idx = np.argmin(time_diff)
    return data_poses[nn_idx]


sequences = ['KAIST/KAIST_01', 'KAIST/KAIST_02', 'KAIST/KAIST_03',
             'DCC/DCC_01', 'DCC/DCC_02', 'DCC/DCC_03',
             'Riverside/Riverside_01', 'Riverside/Riverside_02', 'Riverside/Riverside_03']

for sequence in sequences:
    sequence_path = basedir + sequence + '/Ouster/'
    scan_names = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))

    with open(basedir + sequence + '/global_pose.csv', newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray([int(t) for t in np.asarray(data_poses)[:, 0]])

    for scan_name in scan_names:
        scan_time = int(scan_name.split('/')[-1].split('.')[0])
        scan_pose = findNnPoseUsingTime(scan_time, data_poses_ts, data_poses)
        with open(basedir + sequence + '/scan_poses.csv', 'a', newline='') as csvfile:
            posewriter = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            posewriter.writerow(scan_pose)
