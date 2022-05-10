import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--kitti_eval_seq', type=int, default=8)
eval_arg.add_argument('--mulran_eval_seq', type=str,
                      default='Riverside/Riverside_02')
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth')
eval_arg.add_argument('--eval_batch_size', type=int, default=1)
eval_arg.add_argument('--test_num_workers', type=int, default=1)
eval_arg.add_argument("--eval_random_rotation", type=str2bool,
                      default=False, help="If random rotation. ")
eval_arg.add_argument("--eval_random_occlusion", type=str2bool,
                      default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=3,
                      type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")


# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset
data_arg.add_argument('--eval_dataset', type=str, default='KittiDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=False)
data_arg.add_argument("--eval_save_counts", type=str2bool, default=False)
data_arg.add_argument("--eval_plot_pr_curve", type=str2bool, default=False)
data_arg.add_argument('--num_points', type=int, default=80000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/mnt/088A6CBB8A6CA742/Datasets/Kitti/dataset/',
                      help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'val': [],
    'test': [0]
})

data_arg.add_argument('--mulran_dir', type=str,
                      default='/mnt/088A6CBB8A6CA742/Datasets/MulRan/', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


def get_config_eval():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)
