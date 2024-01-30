import argparse
import logging
import os
import torch.utils.data
from models.ggcnn import GGCNN
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from matplotlib import pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import my_plot_detect_grasps

logging.basicConfig(level=logging.INFO)

root_dir = "./result_test"

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate mobilenet')
    # Network
    parser.add_argument('--network', type=str, default='/home/lqh/ggcnn/my_weights_jacquard/epoch_98_iou_0.98',
                        help='Path to saved network to evaluate')
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='jacquard', help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default='/data_jiang/lqh/J_11', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.1, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0, help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=1, help='Dataset workers')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', default=1, help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', default=1, help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    num1 = []
    num2 = []
    s = 0
    folder_path_false = root_dir + '/false_image'
    folder_path_false2 = root_dir + '/false_image2'
    folder_path_true = root_dir + '/true_image'
    if not os.path.exists(folder_path_false):
        os.makedirs(folder_path_false)
    if not os.path.exists(folder_path_false2):
        os.makedirs(folder_path_false2)
    if not os.path.exists(folder_path_true):
        os.makedirs(folder_path_true)

    output_str = '\n'.join([urls for urls in test_data.dataset.grasp_files])
    with open(root_dir + '/my_directory_path.txt', 'a') as f3:
        f3.write(output_str)

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx + 1, len(test_data)))

            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s is None:
                    s = 0
                ans = round(s, 3)

                filename_with_extension = os.path.basename(test_data.dataset.grasp_files[idx])
                filename, extension = os.path.splitext(filename_with_extension)
                filename_without_extension = os.path.splitext(filename)[0]

                if ans >= 0.2:
                    results['correct'] += 1
                    with open(root_dir + '/true_value.txt', 'a') as f:
                        f.write(str(ans))
                        f.write("\n")
                else:
                    results['failed'] += 1
                    print(test_data.dataset.grasp_files[idx])

                    with open(root_dir + '/false_path.txt', 'a') as f:
                        f.write(str(test_data.dataset.grasp_files[idx]))
                        f.write("\n")

                    with open(root_dir + '/false_value.txt', 'a') as f:
                        f.write(str(ans))
                        f.write("\n")

                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

                    with open(root_dir + '/false_data.txt', 'a') as f:
                        if not grasps:
                            f.write("0;0;0;0;0" + '\n')
                            print("0;0;0;0;0")
                        for g in grasps:
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_gtbb(didx, rot, zoom),
                                       test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                       test_data.dataset.get_depth(didx, rot, zoom), q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img
                                       )
                if ans >= 0.2:
                    file_count_true = len(os.listdir(folder_path_true))
                    file_name_true = str(file_count_true) + str("_") + str(filename_without_extension) + '.png'
                    plt.savefig(os.path.join(folder_path_true, file_name_true))
                else:
                    file_count_false = len(os.listdir(folder_path_false))
                    file_name_false = str(file_count_false) + str("_") + str(filename_without_extension) + '.png'
                    file_name_false2 = str(file_count_false) + '.png'
                    plt.savefig(os.path.join(folder_path_false, file_name_false))
                    plt.savefig(os.path.join(folder_path_false2, file_name_false2))

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                  results['correct'] + results['failed'],
                                                  results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))
