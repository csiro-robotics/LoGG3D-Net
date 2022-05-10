from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.mulran.mulran_dataset import load_poses_from_csv, load_timestamps_csv
from utils.data_loaders.kitti.kitti_dataset import load_poses_from_txt, load_timestamps

__all__ = ['evaluate_sequence_reg']


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.info(f'Finished saving: {file_name}')


def evaluate_sequence_reg(model, cfg):
    save_descriptors = cfg.eval_save_descriptors
    save_counts = cfg.eval_save_counts
    plot_pr_curve = cfg.eval_plot_pr_curve
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(cfg.revisit_criteria), int(cfg.skip_time))
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')
        timestamps = load_timestamps(sequence_path + 'times.txt')
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/kitti_tuples/')
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/mulran_tuples/')
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))

    test_loader = make_data_loader(cfg,
                                   cfg.test_phase,
                                   cfg.eval_batch_size,
                                   num_workers=cfg.test_num_workers,
                                   shuffle=False)

    iterator = test_loader.__iter__()
    logging.info(f'len_dataloader {len(test_loader.dataset)}')

    num_queries = len(positions_database)
    num_thresholds = len(thresholds)

    # Databases of previously visited/'seen' places.
    seen_poses, seen_descriptors, seen_feats = [], [], []

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    start_time = timestamps[0]

    for query_idx in range(num_queries):

        input_data = iterator.next()
        prep_timer.tic()
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
        if not len(lidar_pc) > 0:
            logging.info(f'Corrupt cloud id: {query_idx}')
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        prep_timer.toc()
        desc_timer.tic()
        output_desc, output_feats = model(input)  # .squeeze()
        desc_timer.toc()
        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()

        global_descriptor = np.reshape(global_descriptor, (1, -1))
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]

        if len(global_descriptor) < 1:
            continue

        seen_descriptors.append(global_descriptor)
        seen_poses.append(query_pose)

        if (query_time - start_time - cfg.skip_time) < 0:
            continue

        if save_descriptors:
            feats = output_feats.cpu().detach().numpy()
            seen_feats.append(feats)
            continue

        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - cfg.skip_time))
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_poses = np.copy(seen_poses)
        db_seen_poses = db_seen_poses[:tt+1]
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])

        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        ret_timer.tic()
        feat_dists = cdist(global_descriptor, db_seen_descriptors,
                           metric=cfg.eval_feature_distance).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
        ret_timer.toc()

        place_candidate = seen_poses[nearest_idx]
        p_dist = np.linalg.norm(query_pose - place_candidate)

        # is_revisit = check_if_revisit(query_pose, db_seen_poses, cfg.revisit_criteria)
        is_revisit = is_revisit_list[query_idx]
        is_correct_loc = 0
        if is_revisit:
            num_revisits += 1
            if p_dist <= cfg.revisit_criteria:
                num_correct_loc += 1
                is_correct_loc = 1

        logging.info(
            f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist} p_dist: {p_dist}')

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= cfg.revisit_criteria:
                    num_true_positive[thres_idx] += 1

                elif p_dist > cfg.not_revisit_criteria:
                    num_false_positive[thres_idx] += 1

            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

    F1max = 0.0
    Precisions, Recalls = [], []
    if not save_descriptors:
        for ithThres in range(num_thresholds):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres]

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)

                F1 = 2 * Precision * Recall * (1/(Precision + Recall))

            if F1 > F1max:
                F1max = F1
                F1_TN = nTrueNegative
                F1_FP = nFalsePositive
                F1_TP = nTruePositive
                F1_FN = nFalseNegative
                F1_thresh_id = ithThres
            Precisions.append(Precision)
            Recalls.append(Recall)
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(
            f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
        logging.info(
            f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(
            f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        logging.info(f'F1_thresh_id: {F1_thresh_id}')
        logging.info(f'F1max: {F1max}')

        if plot_pr_curve:
            plt.title('Seq: ' + str(eval_seq) +
                      '    F1Max: ' + "%.4f" % (F1max))
            plt.plot(Recalls, Precisions, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1, 0, 1.1])
            plt.xticks(np.arange(0, 1.01, step=0.1))
            plt.grid(True)
            save_dir = os.path.join(os.path.dirname(__file__), 'pr_curves')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_seq = str(eval_seq).split('/')[-1]
            plt.savefig(save_dir + '/' + eval_seq + '.png')

    if not save_descriptors:
        logging.info('Average times per scan:')
        logging.info(
            f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s Ret: {ret_timer.avg}s ---")
        logging.info('Average total time per scan:')
        logging.info(
            f"--- {prep_timer.avg + desc_timer.avg + ret_timer.avg}s ---")

    if save_descriptors:
        save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        desc_file_name = '/logg3d_descriptor.pickle'
        save_pickle(seen_descriptors, save_dir + desc_file_name)
        feat_file_name = '/logg3d_feats.pickle'
        save_pickle(seen_feats, save_dir + feat_file_name)

    if save_counts:
        save_dir = os.path.join(os.path.dirname(
            __file__), 'pickles/', str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
        save_pickle(num_false_positive, save_dir +
                    '/num_false_positive.pickle')
        save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
        save_pickle(num_false_negative, save_dir +
                    '/num_false_negative.pickle')

    return F1max
