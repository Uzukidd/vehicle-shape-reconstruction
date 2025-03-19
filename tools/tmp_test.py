import open3d as o3d
import numpy as np


def visuzalize_point_cloud(points, box_vertices=None, filter_points=None,
                           vis_vertices=False, vis_filter=False):

    vis = vis = o3d.visualization.Visualizer()
    vis.create_window()

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    point_cloud = o3d.geometry.PointCloud()
    points = points.detach().cpu().numpy()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    vis.add_geometry(point_cloud)



    if vis_vertices:
        for row in range(box_vertices.shape[0]):
            for column in range(box_vertices.shape[1]):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                # print(type(key_points[row][column]))
                sphere.translate(box_vertices[row][column])
                sphere.paint_uniform_color([1, 0, 0])
                vis.add_geometry(sphere)

    if vis_filter:
        filter = o3d.geometry.PointCloud()
        filter_points = filter_points.detach().cpu().numpy()
        filter.points = o3d.utility.Vector3dVector(filter_points)
        colors = np.zeros((filter_points.shape[0], 3))
        colors[:, 0] = 1
        filter.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(filter)

    vis.run()
    vis.destroy_window()
