"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw

box_colormap = [
    [1.0, 1.0, 1.0],
    [0, 1.0, 0],
    [0, 1.0, 1.0],
    [1.0, 1.0, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(
        color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def get_transform_matrix():
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 1],])


def draw_scenes(vis, points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, confidence=None, tracks=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.detach().cpu().numpy()

    # points[:,:3] = np.dot(points[:,:3] , get_transform_matrix().T)
    # vis.get_render_option().line_width = 10
    vis.get_render_option().point_size = 4.0
    vis.get_render_option().background_color = np.zeros((3))

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(
            np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    vis.add_geometry(pts)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1.0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1.0, 0),
                       ref_labels, ref_scores, confidence)

    if tracks is not None:
        vis = draw_tracks(vis, tracks, (0, 1.0, 0), ref_labels)


def draw_tracks(vis, tracks, color=(0, 1.0, 0), ref_labels=None):
    for i in range(tracks.__len__()):

        if ref_labels is None:
            uni_color = color
        else:
            uni_color = box_colormap[ref_labels[i]]

        nodes = [node.cpu().numpy() for node in tracks[i]]
        lines = [(i, i + 1) for i in range(nodes.__len__() - 1)]
        colors = [uni_color for _ in range(nodes.__len__() - 1)]

        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(nodes)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

    return vis


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """

    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    box3d.color = np.ones((3))

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1.0, 0), ref_labels=None, score=None, confidence=None):
    for i in range(gt_boxes.shape[0]):

        if confidence is not None:
            if score[i] < confidence[ref_labels[i]]:
                continue

        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        if score is not None:
            corners = box3d.get_box_points()
            # text = o3d.geometry.create_text_geometry('%.2f' % score[i], 20, 10)
            # text.translate(corners[5])
            # text_3d("test", corners[5], font_size=32)
            # vis.add_geometry(text)
            # vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
