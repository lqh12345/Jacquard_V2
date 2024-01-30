import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps


def plot_output(ground_truth_bbs, rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax, color="green")
    ax.set_title('predictive RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax, color="green")
    ax.set_title('predictive depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(rgb_img)
    for gt in ground_truth_bbs:
        # gt.plot(ax)
        gt.plot(ax, color="green", linewidth=0.5, alpha=0.5)
    # for g in gs:
    #     g.plot(ax, color="red")
    ax.set_title('annotated RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(depth_img, cmap='gray')
    for gt in ground_truth_bbs:
        # gt.plot(ax)
        gt.plot(ax, color="green", linewidth=0.5, alpha=0.5)
    # for g in gs:
    #     g.plot(ax, color="red")
    ax.set_title('annotated depth')
    ax.axis('off')

    # ax = fig.add_subplot(3, 2, 5)
    # plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    # ax.set_title('Q')
    # ax.axis('off')
    # plt.colorbar(plot)
    #
    # ax = fig.add_subplot(3, 2, 6)
    # plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    # ax.set_title('Angle')
    # ax.axis('off')
    # plt.colorbar(plot)

    # plt.show()

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """
    max_gs = 0

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs #doushizhegechengli
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    # print("gs:", gs)
    # print("gt_bbs:", gt_bbs)
    # print("ground_truth_bbs:", ground_truth_bbs)
    # print("GraspRectangles:", GraspRectangles)
    for g in gs:
        max_gs = g.max_iou(gt_bbs)
        # print("max_gs:", max_gs)
        return max_gs

# def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
#     """
#     Plot the output of a GG-CNN
#     :param rgb_img: RGB Image
#     :param depth_img: Depth Image
#     :param grasp_q_img: Q output of GG-CNN
#     :param grasp_angle_img: Angle output of GG-CNN
#     :param no_grasps: Maximum number of grasps to plot
#     :param grasp_width_img: (optional) Width output of GG-CNN
#     :return:
#     """
#     gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(rgb_img)
# for g in gs:
#     g.plot(ax)
# ax.set_title('RGB')
# ax.axis('off')
#
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(depth_img, cmap='gray')
# for g in gs:
#     g.plot(ax)
# ax.set_title('Depth')
# ax.axis('off')
#
# ax = fig.add_subplot(2, 2, 3)
# plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
# ax.set_title('Q')
# ax.axis('off')
# plt.colorbar(plot)
#
# ax = fig.add_subplot(2, 2, 4)
# plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
# ax.set_title('Angle')
# ax.axis('off')
# plt.colorbar(plot)
# plt.show()

# def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
#     """
#     Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
#     A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
#     :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
#     :param grasp_angle: Angle outputs of GG-CNN
#     :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
#     :param no_grasps: Maximum number of grasps to consider per image.
#     :param grasp_width: (optional) Width output from GG-CNN
#     :return: success
#     """
#     data_ture = []
#     data_false = []
#     num1 = 0
#     num2 = 0
#     if not isinstance(ground_truth_bbs, GraspRectangles):
#         gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
#     else:
#         gt_bbs = ground_truth_bbs
#     gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
#     for g in gs:
#         data = g.max_iou(gt_bbs)
#         if data > 0.25:
#             data_ture.append(data)
#             print('data:', data)
#             print('g.max_iou(gt_bbs):', g.max_iou(gt_bbs))
#             print('num1:', num1)
#             num1 += 1
#             # if num1 > 5:
#             plt.hist(data_ture, bins=6)
#             plt.title('Histogram_true')
#             plt.xlabel('Values')
#             plt.ylabel('IOU')
#             plt.savefig('/home/lqh/ggcnn/ture.png')
#             plt.show()
#             return True
#         else:
#             data_false.append(data)
#             num2 = num2 + 1
#             if num2 > 5:
#                 plt.hist(data_false, bins=6)
#                 plt.title('Histogram_false')
#                 plt.xlabel('Values')
#                 plt.ylabel('IOU')
#                 plt.savefig('/home/lqh/ggcnn/false.png')
#                 plt.show()
#             return False

# def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
#     """
#     Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
#     A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
#     :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
#     :param grasp_angle: Angle outputs of GG-CNN
#     :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
#     :param no_grasps: Maximum number of grasps to consider per image.
#     :param grasp_width: (optional) Width output from GG-CNN
#     :return: success
#     """
#
#     if not isinstance(ground_truth_bbs, GraspRectangles):
#         gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
#     else:
#         gt_bbs = ground_truth_bbs
#     gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
#     for g in gs:
#         return g.max_iou(gt_bbs)
#         # if g.max_iou(gt_bbs) > 0.25:
#             # return True
#     # else:
#
#         # return False
