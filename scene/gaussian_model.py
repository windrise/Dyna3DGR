# Copyright (C) 2023, Inria
# under the terms of the LICENSE.md file.

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_cosine_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation_inverse
import torch.nn.functional as F

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


class GaussianModel:
    def __init__(self, fea_dim=0, with_motion_mask=True, **kwargs):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance


        self._xyz = torch.empty(0)
        self._cmesh = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.with_motion_mask = with_motion_mask
        if self.with_motion_mask:
            fea_dim += 1
        self.fea_dim = fea_dim
        self.feature = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def param_names(self):
        return ['_xyz', '_scaling', '_rotation', '_density', 'xyz_gradient_accum']

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)
        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._density = nn.Parameter(gs._density)
        new_gs.feature = nn.Parameter(gs.feature)
        return new_gs

    @property
    def motion_mask(self):
        if self.with_motion_mask:
            return torch.sigmoid(self.feature[..., -1:])
        else:
            return torch.ones_like(self._xyz[..., :1])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def get_covariance_phy(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return strip_symmetric(self.covariance_activation(self.get_scaling, scaling_modifier, rotation))

    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_fbp(self, fbp_image, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1,
                        num_samples=150000):
        self.spatial_lr_scale = spatial_lr_scale
        fbp_image = fbp_image.unsqueeze(-1)

        bs, D, H, W, _ = fbp_image.shape
        fbp_image = fbp_image.permute(0, 4, 1, 2, 3)
        fbp_image = F.interpolate(fbp_image, size=(H, H, W), mode='trilinear', align_corners=False)
        grad_x = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 2:, 1:-1, 1:-1])
        grad_x_padded = F.pad(grad_x, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_y_padded = F.pad(grad_y, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_z_padded = F.pad(grad_z, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_norm = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2 + grad_z_padded ** 2)
        grad_norm = grad_norm.reshape(-1)
        _, indices = torch.topk(grad_norm, num_samples)
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), torch.arange(W)), dim=-1).reshape(-1,
                                                                                                                3).cuda()
        sampled_coords = coords[indices]
        grid = torch.zeros((H, H, W), dtype=torch.int32, device="cuda")
        indices_3d = sampled_coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1
        kernel_size = 5
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device="cuda", dtype=torch.float32)
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        num_neighbours = neighbours_count[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]]
        fbp_image[fbp_image < air_threshold] = 0
        densities = ini_density * fbp_image.reshape(-1)[indices]
        print("density:", densities.max(), densities.mean(), densities.min())
        sampled_coords = sampled_coords.float()
        sampled_coords = sampled_coords / torch.tensor([H, H, W], dtype=torch.float, device="cuda")
        print("Number of points at initialisation: ", num_samples)
        densities = inverse_sigmoid(densities).unsqueeze(1)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(sampled_coords.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((num_samples, 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        self.feature = nn.Parameter(-1e-2 * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])

    def create_from_points_cloud(self, pcd: BasicPointCloud, spatial_lr_scale, print_info = True):
        self.spatial_lr_scale = 5
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        features = torch.zeros((fused_color.shape[0], 3, 1)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = inverse_sigmoid(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        densities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.feature = nn.Parameter(-1e-2 * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])

    def create_from_own_points_cloud(self, points, spatial_lr_scale, print_info = True):
        self.spatial_lr_scale = 5
        if type(points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        else:
            fused_point_cloud = points


        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = inverse_sigmoid(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        densities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._cmesh = fused_point_cloud
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.feature = nn.Parameter(-1e-2 * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])

    def create_from_gaussians(self, pcd, densities, scales, rots, spatial_lr_scale, feature):
        self.spatial_lr_scale = spatial_lr_scale

        self._xyz = nn.Parameter(pcd.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.feature = nn.Parameter(feature, requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])

    def create_from_own_gaussians(self, pcd, densities, scales, rots, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

        self._xyz = nn.Parameter(pcd.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.feature = nn.Parameter(-1e-2 * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._density], 'lr': training_args.opacity_lr, "name": "density"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.fea_dim):
            l.append('fea_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        densities = self._density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.fea_dim > 0:
            feature = self.feature.detach().cpu().numpy()
            attributes = np.concatenate((attributes, feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_density(self):
        densities_new = inverse_sigmoid(torch.min(self.get_density, torch.ones_like(self.get_density) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        feas = np.zeros((xyz.shape[0], self.fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.fea_dim > 0:
            self.feature = nn.Parameter(torch.tensor(feas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_densities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "density": new_densities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def check_points_valid(self, points, surface_points, max_distance):
        """
        向量化检查点是否在表面点云的有效范围内
        Args:
            points: 待判断的点集 [N, 3]
            surface_points: 心肌表面点云 [M, 3]
            max_distance: 到表面的最大允许距离
        Returns:
            valid_mask: bool tensor [N]
        """
        diff = points.unsqueeze(1) - surface_points.unsqueeze(0)
        distances = torch.norm(diff, dim=-1)
        min_distances = torch.min(distances, dim=1)[0]
        return min_distances < max_distance

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2, selected_pts_mask=None, without_prune=False, surface_points=None, max_distance=0.05):
        if selected_pts_mask is None:
            n_init_points = self.get_xyz.shape[0]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if surface_points is not None:
            valid_mask = self.check_points_valid(new_xyz, surface_points, max_distance)

            new_xyz = new_xyz[valid_mask]
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1)[valid_mask] / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)[valid_mask]
            new_density = self._density[selected_pts_mask].repeat(N, 1)[valid_mask]

        else:
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_density = self._density[selected_pts_mask].repeat(N, 1)

        print("=================================================")
        print("Spliting {} points".format(new_density.shape[0]))

        self.densification_postfix(new_xyz, new_density, new_scaling, new_rotation)

        if not without_prune:
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        print("=================================================")
        print("Cloning {} points".format(selected_pts_mask.sum()))
        new_xyz = self._xyz[selected_pts_mask]
        new_densities = self._density[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]


        self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, surface_points=None, max_distance=0.1):
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads = grads, grad_threshold = max_grad, scene_extent=extent)
        self.densify_and_split(grads= grads, grad_threshold = max_grad, scene_extent=extent, without_prune=True, surface_points=surface_points)

        prune_mask = (self.get_density < min_opacity).squeeze()
        print("Pruning {} points".format(prune_mask.sum()))
        if surface_points is not None:
            surface_valid_mask = self.check_points_valid(self._xyz, surface_points, max_distance)
            prune_mask = torch.logical_or(prune_mask, ~surface_valid_mask)
        print("Pruning {} points after surface".format(prune_mask.sum()))
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def state_dict(self):
        return {
            '_xyz': self._xyz,
            '_density': self._density,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
        }

    def load_state_dict(self, state_dict):
        self._xyz = nn.Parameter(state_dict['_xyz'].clone().detach().requires_grad_(True))
        self._density = nn.Parameter(state_dict['_density'].clone().detach().requires_grad_(True))
        self._scaling = nn.Parameter(state_dict['_scaling'].clone().detach().requires_grad_(True))
        self._rotation = nn.Parameter(state_dict['_rotation'].clone().detach().requires_grad_(True))

class StandardGaussianModel(GaussianModel):
    def __init__(self, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(fea_dim, with_motion_mask)
        self.all_the_same = all_the_same
    
    @property
    def get_scaling(self):
        scaling = self._scaling.mean()[None, None].expand_as(self._scaling) if self.all_the_same else self._scaling.mean(dim=1, keepdim=True).expand_as(self._scaling)
        return self.scaling_activation(scaling)
