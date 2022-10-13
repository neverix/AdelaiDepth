# To install dependencies: pip install numpy, open3d, opencv-python
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from regex import D
import scipy.spatial
import numba as nb
import cv2
import copy

from sklearn.metrics import hamming_loss


def process_depth(dep):
    depth = dep.copy()
    depth -= depth.min()
    depth /= depth.max()
    depth = 1 / np.clip(depth, 0.2, 1)
    # 9 not available because it requires 8-bit
    blurred = cv2.medianBlur(depth, 5)
    maxd = cv2.dilate(blurred, np.ones((3, 3)))
    mind = cv2.erode(blurred, np.ones((3, 3)))
    edges = maxd - mind
    threshold = .05  # Better to have false positives
    pick_edges = edges > threshold
    return 1 / dep, pick_edges


# @nb.jit
def make_mesh(pic, depth, pick_edges):
    faces = []
    im = np.asarray(pic)
    grid = np.mgrid[0:im.shape[0], 0:im.shape[1]].transpose(1, 2, 0
                                                            ).reshape(-1, 2)[..., ::-1]
    flat_grid = grid[:, 1] * im.shape[1] + grid[:, 0]
    positions = np.concatenate(((grid - np.array(im.shape[:-1])[np.newaxis, :]
                                 / 2) / im.shape[1] * 2,
                                depth.flatten()[flat_grid][..., np.newaxis]),
                               axis=-1)
    positions[:, :-1] *= positions[:, -1:]
    positions[:, 1] *= -1
    colors = im.reshape(-1, 3)[flat_grid]

    def c(x, y): return y * im.shape[1] + x
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if pick_edges[y, x]:
                continue
            if x > 0 and y > 0:
                faces.append([c(x, y), c(x, y - 1), c(x - 1, y)])
            if x < im.shape[1] - 1 and y < im.shape[0] - 1:
                faces.append([c(x, y), c(x, y + 1), c(x + 1, y)])

    faces = np.asarray(faces)
    # face_colors = np.asarray([colors[i[0]] for i in faces])
    # positions = positions[]
    return positions, faces, colors


def args_to_mat(tx, ty, tz, rx, ry, rz):
    mat = np.eye(4)
    mat[:3, :3] = scipy.spatial.transform.Rotation.from_euler(
        "XYZ", (rx, ry, rz)).as_matrix()
    mat[:3, 3] = tx, ty, tz
    return mat  # Create a renderer with the desired image size


class Rendererer():
    def __init__(self, w=1024, h=1024, near=0.01, far=4., fov=90.):
        self.render = o3d.visualization.rendering.OffscreenRenderer(w, h)
        self.render.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self.vertical_field_of_view = fov  # between 5 and 90 degrees

        aspect_ratio = w / h  # azimuth over elevation
        self.near_plane = near
        self.far_plane = far
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        self.render.scene.camera.set_projection(
            self.vertical_field_of_view, aspect_ratio, self.near_plane, self.far_plane, fov_type)

        # center = [0, 0, -1]  # look_at target
        # eye = [0, 0, 0]  # camera position
        # up = [0, 1, 0]  # camera orientation
        # self.render.setup_camera(fov, center, eye, up)

    def __call__(self, img, depth, translation=np.array([0.0, 0., 0.])):
        # img: (h, w, 3)
        # depth: (h, w)
        # translation: (3)

        dep, edges = process_depth(depth)
        vertices, faces, colors = make_mesh(img, dep, edges)
        vertices = vertices.reshape(-1, 3)
        vertices[:, -1] *= -1
        # vertices = vertices + translation
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
            vertices.astype("float64")), o3d.utility.Vector3iVector(faces))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        self.render.scene.add_geometry("geom", mesh, mtl)

        # center = np.asarray([0, 0, 0]) + translation  # look_at target
        # eye = np.asarray([0, 0, -1]) + translation  # camera position
        # up = [0, 1, 0]  # camera orientation
        # self.render.scene.camera.look_at(center, eye, up)

        # Read the image into a variable
        img = np.asarray(self.render.render_to_image())
        dep = np.asarray(self.render.render_to_depth_image())

        # clean up
        self.render.scene.remove_geometry("geom")

        # * (self.far_plane - self.near_plane) + self.near_plane
        return img, dep * (self.far_plane - self.near_plane) + self.near_plane


class DepthFupper():
    def __init__(self, w=1024, h=1024, scale=0.1):
        self.renderer = Rendererer(w, h)
        self.scale = scale

    def __call__(self, dep):
        # img = np.stack((dep * 0,) * 3, axis=-1)
        d = dep - dep.min()
        img = np.stack((d / d.max(),) * 3, axis=-1)
        translation = np.random.normal() * self.scale
        img, dep = self.renderer(img, dep, translation)
        # img, dep = self.renderer(img, dep, -translation)
        return dep


# Create the mesh geometry.
# (We use arrows instead of a sphere, to show the rotation more clearly.)
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
# size=1.0, origin=np.array([0.0, 0.0, 0.0]))
# h = 32
# img_dep = np.zeros((h, h, 3), dtype=np.uint8)
# img = cv2.imread("midas.jpg") / 255.
img_dep = cv2.cvtColor(cv2.imread("midas.jpg"), cv2.COLOR_RGB2GRAY)
img_dep = img_dep.astype("float32") / 255. * 2 + 20
ren = DepthFupper()
plt.imshow(img_dep)
plt.colorbar()
plt.show()
plt.imshow(ren(img_dep))
plt.colorbar()
plt.show()
