import datetime
import os
import sys
import argparse
import logging
import cv2
import torch
import torch.utils.data
from torchsummary import summary
import tensorboardX
from utils.visualisation.gridshow import gridshow
from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
logging.basicConfig(level=logging.INFO)
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
use_cuda = torch.cuda.is_available()
logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()
    print(f"device_num is:{device}")
else:
    device = torch.device("cpu")
    print("CUDA is not available")

def parse_args():
    parser = argparse.ArgumentParser(description='Train mobileV2')
    # Network
    parser.add_argument('--network', type=str, default='mobileV2', help='Network Name in '
    '(xception)(ggcnn2)(ggcnn)(mobileV2)(mobilev2pruning)(squeeze)(shuffle)(resnet50)(resnet101)(resnet152)')
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='jacquard', help='Dataset Name ("cornell" or "jacquard" or "multi_targets")')
    parser.add_argument('--dataset-path', type=str, default='/data_jiang/lqh/Jacquard_th10', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.7, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=101, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1001, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=251, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='training_example', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')
    # parser.add_argument('--vis', default=True, help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)

            yc = [yy.to(device) for yy in y]

            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

def run():
    args = parse_args()

    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb

    mobilenet = get_network(args.network)
    net = mobilenet(input_channels=input_channels) #modify

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0005) #0.00025;0.0005;0.001
    logging.info('Done')


    summary(net, (input_channels, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    Epoch = []
    Loss = []
    IOU = []

    with open(save_folder + '/document.txt', 'a') as f:
        f.write("network:" + str(args.network))
        f.write("\n")
        f.write("dataset:" + str(args.dataset))
        f.write("\n")
        f.write("dataset_path:" + str(args.dataset_path))
        f.write("\n")
        f.write("batch_size:" + str(args.batch_size))
        f.write("\n")
        f.write("epochs:" + str(args.epochs))
        f.write("\n")

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
        Epoch.append(epoch)
        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))
        with open('train_value.txt', 'a') as f:
            f.write(str(test_results['correct']/(test_results['correct']+test_results['failed'])))
            f.write("\n")

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # scheduler.step(iou)
        Loss.append(train_results['loss'])
        IOU.append(iou)
        with open(save_folder + '/IOU_value.txt', 'a') as f:
            f.write(str(IOU))
            f.write("\n")
        with open(save_folder + '/Loss_value.txt', 'a') as f:
            f.write(str(Loss))
            f.write("\n")
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_loss_%0.2f_statedict.pt' % (epoch, iou, train_results['loss'])))
            best_iou = iou

    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(Epoch, Loss, color='b', linestyle='-', label='loss')
    ax2.plot(Epoch, IOU, color='r', linestyle='-', label='IOU')
    plt.title('Loss-IOU')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('IOU', color='r')
    plt.ylim([0, 1])
    plt.xlim([-1, 102])
    plt.savefig('th10_lr0.0005.png')
    plt.show()

if __name__ == '__main__':
    run()
