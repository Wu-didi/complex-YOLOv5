import numpy as np
import mayavi.mlab

#mayavi显示点云
def read_pointcloud_from_bin_file(bin_path):
    pointcloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
    print(pointcloud.shape)
    print(type(pointcloud))
    x = pointcloud[:, 0]  # x position of point
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0)
    y = pointcloud[:, 1]  # y position of point
    ymin = np.amin(y, axis=0)
    ymax = np.amax(y, axis=0)
    z = pointcloud[:, 2]  # z position of point
    zmin = np.amin(z, axis=0)
    zmax = np.amax(z, axis=0)
    print(xmin,xmax,ymin,ymax,zmin,zmax)
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         # 灰度图的伪彩映射
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    # 绘制原点
    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere",scale_factor=1)
    # 绘制坐标
    axes = np.array(
        [[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0]],
        dtype=np.float64,
    )
    #x轴
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    #y轴
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    #z轴
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    mayavi.mlab.show()


def plot3Dbox(corners):
    for i in range(corners.shape[0]):
        corner = corners[i]
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        mayavi.mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)
        # mlab.show(stop=True)

def main():
    corners = np.array([[[0.0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]])
    #plot3Dbox(corners)
    point_path = r"dataset\kitti\training\velodyne\000075.bin"
    read_pointcloud_from_bin_file(point_path)

if __name__ == '__main__':
    main()
