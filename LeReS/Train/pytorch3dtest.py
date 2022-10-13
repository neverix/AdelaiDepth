# To install dependencies: pip install numpy, open3d, opencv-python
from torch import nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from matplotlib import pyplot as plt
import numpy as np
from regex import D
import scipy.spatial
import numba as nb
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import torch

import trimesh
import cv2


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
    # plt.imshow(pick_edges)
    # plt.colorbar()
    # plt.show()
    pick_edges = np.logical_or(
        pick_edges, cv2.dilate((dep <= 1e-4).astype("uint8"), np.ones((3, 3))))
    # plt.imshow(pick_edges)
    # plt.colorbar()
    # plt.show()
    return dep, pick_edges


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

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if pick_edges[y, x]:
                continue
            if x > 0 and y > 0:
                faces.append([y * im.shape[1] + x,
                              (y - 1) * im.shape[1] + x,
                              y * im.shape[1] + (x-1)])
            if x < im.shape[1] - 1 and y < im.shape[0] - 1:
                faces.append([y * im.shape[1] + x, (y + 1) *
                             im.shape[1] + x, y * im.shape[1] + (x + 1)])

    faces = np.asarray(faces)
    face_colors = np.asarray([colors[i[0]] for i in faces])
    # positions = positions[]
    return positions, faces, colors


def args_to_mat(tx, ty, tz, rx, ry, rz):
    mat = np.eye(4)
    mat[:3, :3] = scipy.spatial.transform.Rotation.from_euler(
        "XYZ", (rx, ry, rz)).as_matrix()
    mat[:3, 3] = tx, ty, tz
    return mat  # Create a renderer with the desired image size


# https://github.com/facebookresearch/pytorch3d/issues/84
class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        # (N, H, W, 3) RGBA image
        return torch.cat((images, fragments.zbuf), dim=-1)


class Rendererer():
    def __init__(self, w=512, h=512, near=0.01, far=4., fov=90.):
        self.w, self.h = w, h

        raster_settings = RasterizationSettings(
            image_size=self.h,  # 256,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None
        )
        batch = 1
        device = "cpu"
        cameras = FoVPerspectiveCameras(device=device)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader(device=device)
        )

    def __call__(self, img, depth, translation=np.array([0.0, 0., 0.]), rev=False):
        # img: (h, w, 3)
        # depth: (h, w)
        # translation: (3)

        dep, edges = process_depth(depth)
        vertices, faces, vert_colors = make_mesh(img, dep, edges)
        vertices = vertices.reshape(-1, 3)
        # vertices[:, -1] *= -1
        # vertices = vertices + translation

        mat = args_to_mat(*translation)
        if rev:
            mat = np.linalg.inv(mat)
        vertices = vertices  # * np.array([1.0, 1.0, -1.0])
        vertices = np.concatenate(
            (vertices, vertices[..., -1:] * 0 + 1), axis=-1)
        vertices = (vertices @ mat)[..., :3]
        # tri_mesh = trimesh.Trimesh(vertices=vertices,  # + translation,
        #                            faces=faces,
        #                            face_colors=np.concatenate((face_colors,
        #                                                        face_colors[..., -1:]
        #                                                        * 0 + 255),
        #                                                       axis=-1).reshape(-1, 4),
        #                            smooth=False,
        #                            )

        # mesh = pyrender.mesh.Mesh.from_trimesh(tri_mesh, smooth=False)
        # scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0]))
        # camera = pyrender.PerspectiveCamera(
        #     yfov=np.pi / 2, aspectRatio=self.h/self.w)
        # scene.add(camera)  # , pose=mat)  # , pose=mat)
        # scene.add(mesh)
        # rgb, d = self.render.render(scene, pyrender.constants.RenderFlags.FLAT)
        with torch.no_grad():
            textures = TexturesVertex(
                torch.from_numpy(vert_colors).unsqueeze(0).float())
            mesh = Meshes(torch.from_numpy(vertices).unsqueeze(0).float(),
                          torch.from_numpy(faces).unsqueeze(0).float(),
                          textures)
            # mesh = load_objs_as_meshes(["Wolves.obj"] * batch, device=device)
            images = self.renderer(mesh)
        # mask = d == 0
        # rgb = rgb.copy()
        # rgb[mask] = 0
        # res = Image.fromarray(np.concatenate(
        # (rgb, ((mask[..., np.newaxis]) == 0).astype(np.uint8) * 255), axis=-1))
        return images.detach().cpu().numpy()[0, ..., :3], images.detach().cpu().numpy()[0, ..., -1]


class DepthCrusher():
    def __init__(self, w=1024, h=1024, t_scale=4.0, r_scale=0.025):
        self.renderer = Rendererer(w, h)
        self.t_scale = t_scale
        self.r_scale = r_scale

    def __call__(self, dep):
        # img = np.stack((dep * 0,) * 3, axis=-1)
        d = dep - dep.min()
        img = np.stack((d / d.max(),) * 3, axis=-1)
        translation = np.random.normal(size=6)
        translation[:3] *= self.t_scale
        translation[3:] *= self.r_scale
        img, dep = self.renderer(img, dep, translation, rev=False)
        # # process_depth(dep)
        # plt.imshow(dep)
        # plt.colorbar()
        # plt.show()
        img, dep = self.renderer(img, dep, translation, rev=True)
        # dep[dep < 1e-4] = dep.max() * 2
        return dep


if __name__ == "__main__":
    # Create the mesh geometry.
    # (We use arrows instead of a sphere, to show the rotation more clearly.)
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
    # size=1.0, origin=np.array([0.0, 0.0, 0.0]))
    # h = 32
    # img_dep = np.zeros((h, h, 3), dtype=np.uint8)
    # img = cv2.imread("midas.jpg") / 255.
    img_dep = cv2.cvtColor(cv2.imread("midas.jpg"), cv2.COLOR_RGB2GRAY)
    img_dep = img_dep.astype("float32") / 255. * 10 + 10
    img_dep = (img_dep - img_dep.min()) / \
        (img_dep.max() - img_dep.min()) * 20 + 5
    ren = DepthCrusher()
    # plt.imshow(img_dep)
    # plt.colorbar()
    # plt.show()
    plt.imshow(ren(img_dep))
    plt.colorbar()
    plt.show()
