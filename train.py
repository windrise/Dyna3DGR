import os
import time
import torch
import numpy as np
import sys
import copy
import json
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors, KDTree
from monai.metrics import SSIMMetric, PSNRMetric

from scene.deform_model import DeformModel
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.Compute_intensity import compute_intensity
from utils.general_utils import safe_state, get_linear_noise_func
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation

vol_ssim_metric = SSIMMetric(spatial_dims=3)
vol_psnr_metric = PSNRMetric(max_val=1.0)

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), 
                                             torch.linspace(0, 1, steps=h), 
                                             torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def get_inv_covariance(scaling, rotation, scaling_modifier):
    scaling_inv_squared = 1.0 / (scaling * scaling_modifier) ** 2
    S_inv_squared = torch.diag_embed(scaling_inv_squared)
    R_mat = build_rotation(rotation)
    R_transpose = R_mat.transpose(1, 2)
    covariance_inv = torch.matmul(R_mat, torch.matmul(S_inv_squared, R_transpose))
    return covariance_inv

def rendering2(pc: GaussianModel, grid, d_xyz, d_rotation, d_scaling, d_density):
    grid_point = grid.unsqueeze(-2)
    z, x, y = grid_point.shape[1:4]
    density_grid = torch.zeros(1, z, x, y, 1, device='cuda', requires_grad=True)
    new_xyz = pc.get_xyz + d_xyz
    new_density = pc.get_density if d_density is None else pc.get_density + d_density
    new_scaling = pc.get_scaling * d_scaling
    rotations = pc.get_rotation_bias(d_rotation)
    inv_covariance = get_inv_covariance(new_scaling, rotations, scaling_modifier=1.0)
    density_grid = compute_intensity(
        new_xyz.contiguous(),
        grid_point.contiguous(),
        new_density.contiguous(),
        inv_covariance.contiguous(),
        new_scaling.contiguous(),
        density_grid.contiguous(), )
    return density_grid

def calculate_multi_frame_dice(gt, pred, eps=1e-5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    num_frames = gt.shape[-1]
    dice_sum = 0
    dice_split = []
    for frame in range(num_frames):
        pred_flat = pred[..., frame].reshape(-1)
        gt_flat = gt[..., frame].reshape(-1).float()
        tp = torch.sum(pred_flat * gt_flat)
        fp = torch.sum(pred_flat * (1 - gt_flat))
        fn = torch.sum((1 - pred_flat) * gt_flat)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        dice_sum += dice.item()
        dice_split.append(dice.item())
    return dice_sum / num_frames, dice_split

def val_4ddata(total_frame, frame_list, deform, gaussians, gt_4d, image_size, epoch, best_val4d_loss, test_num, label_id, result_path, save_result=False):
    val_4d = torch.zeros((*image_size, total_frame), dtype=torch.float32)
    for j in range(total_frame):
        index = frame_list[j % total_frame]
        fid = torch.tensor(index / total_frame).cuda()
        time_input = fid.unsqueeze(0).expand(deform.deform.node_num, -1)
        ast_noise = 0
        d_values = deform.step(deform.deform.as_gaussians.get_xyz.detach(), time_input + ast_noise, feature=None,
                               motion_mask=deform.deform.as_gaussians.motion_mask)
        d_xyz, d_rot, d_scale, d_density = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_density']
        d_rot = 0
        d_scaling = torch.mean(d_values['d_scaling'], dim=(1), keepdim=True)
        d_scale = d_scaling.expand_as(d_values['d_scaling'])
        grid = create_grid_3d(*image_size)
        grid = grid.cuda().unsqueeze(0).repeat(1, 1, 1, 1, 1)
        train_output = rendering2(gaussians, grid, d_xyz, d_rot, d_scale, d_density)
        val_4d[:, :, :, index] = train_output.squeeze(0).squeeze(3)
    valloss = l1_loss(val_4d, gt_4d).detach()
    if save_result:
        fbp_recon_g = nib.Nifti1Image(val_4d.detach().cpu().numpy(), np.eye(4))
        nib.save(fbp_recon_g, os.path.join(result_path, f"pred_{label_id}_{test_num}.nii.gz"))
    vol_psnr_metric(y_pred=val_4d.permute(3, 0, 1, 2).unsqueeze(0).detach().cpu(),
                    y=gt_4d.permute(3, 0, 1, 2).unsqueeze(0).detach().cpu())
    val_4d_binary = (val_4d > 0.1).float()
    gt_4d_binary = (gt_4d > 0).float()
    dice, dice_split = calculate_multi_frame_dice(gt_4d_binary, val_4d_binary)
    print(f"Final eval results ***  Dice: {dice}")
    return valloss, dice, dice_split, vol_psnr_metric.aggregate().item()

def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    num_points = len(points)
    selected_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(num_points) * 1e10
    selected_indices[0] = np.random.randint(num_points)
    for i in range(1, num_samples):
        last_idx = selected_indices[i - 1]
        dist_to_last = np.sum((points - points[last_idx]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected_indices[i] = np.argmax(distances)
    return selected_indices

def sample_points_from_cardiac(volume_data: np.ndarray, num_points: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    cardiac_indices = np.where(volume_data > 0)
    total_voxels = len(cardiac_indices[0])
    volume_shape = np.array(volume_data.shape)
    if total_voxels < num_points:
        all_points = np.vstack([cardiac_indices[0], cardiac_indices[1], cardiac_indices[2]]).T
        extra_points_needed = num_points - total_voxels
        selected_indices = np.random.choice(total_voxels, extra_points_needed, replace=True)
        base_points = np.vstack([cardiac_indices[0][selected_indices], cardiac_indices[1][selected_indices], cardiac_indices[2][selected_indices]]).T
        extra_points = base_points.copy()
        cardiac_mask = (volume_data > 0)
        for i, point in enumerate(base_points):
            for _ in range(10):
                jitter = np.random.uniform(-0.5, 0.5, size=3)
                new_point = point + jitter
                new_coords = np.round(new_point).astype(int)
                if (0 <= new_coords[0] < volume_shape[0] and
                    0 <= new_coords[1] < volume_shape[1] and
                    0 <= new_coords[2] < volume_shape[2]):
                    if cardiac_mask[new_coords[0], new_coords[1], new_coords[2]]:
                        extra_points[i] = new_point
                        break
        points = np.vstack([all_points, extra_points])
    else:
        selected_indices = np.random.choice(total_voxels, num_points, replace=False)
        points = np.vstack([cardiac_indices[0][selected_indices], cardiac_indices[1][selected_indices], cardiac_indices[2][selected_indices]]).T
    normalized_points = points / (volume_shape - 1)[:, np.newaxis].T
    return normalized_points

def process_ACDC(patient_id, label_value, data_root, acdc_info_path):
    root_dir = data_root
    target_size = [128, 128, 32]
    patient_id_num = int(patient_id)
    sub_dir = "training" if patient_id_num < 101 else "testing"
    patient_dir = os.path.join(root_dir, sub_dir, f"patient{patient_id}")
    with open(acdc_info_path, 'r') as f:
        acdc_info = json.load(f)
    ed_frame = acdc_info[patient_id]['ED']
    es_frame = acdc_info[patient_id]['ES']
    
    def get_file_paths(frame):
        frame_str = f"{frame:02d}"
        image_path = os.path.join(patient_dir, f"patient{patient_id}_frame{frame_str}.nii.gz")
        label_path = os.path.join(patient_dir, f"patient{patient_id}_frame{frame_str}_gt.nii.gz")
        return image_path, label_path
        
    def read_and_resample(image_path, is_label=False):
        image = sitk.ReadImage(image_path)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        target_spacing = [1.5, 1.5, 3.15]
        target_size_res = [
            int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
            int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
            int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(target_size_res)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampled_image = resampler.Execute(image)
        return sitk.GetArrayFromImage(resampled_image).astype(np.float32).transpose(2, 1, 0)
        
    def process_single_frame(image_array, label_array):
        if label_value == 4:
            label_mask = label_array > 0
        else:
            label_mask = label_array == label_value
        image_array = image_array * label_mask
        label_array = label_array * label_mask if label_value == 4 else label_mask * label_value
        nh, nw, nd = image_array.shape
        sh = int((nh - target_size[0]) / 2) if nh > target_size[0] else 0
        sw = int((nw - target_size[1]) / 2) if nw > target_size[1] else 0
        if nh >= target_size[0] and nw >= target_size[1]:
            image_array = image_array[sh:sh + target_size[0], sw:sw + target_size[1]]
            label_array = label_array[sh:sh + target_size[0], sw:sw + target_size[1]]
        else:
            temp_image, temp_label = np.zeros(target_size), np.zeros(target_size)
            start_h, start_w = max(0, (target_size[0] - nh) // 2), max(0, (target_size[1] - nw) // 2)
            h_range, w_range = min(nh, target_size[0]), min(nw, target_size[1])
            temp_image[start_h:start_h + h_range, start_w:start_w + w_range] = image_array[:h_range, :w_range]
            temp_label[start_h:start_h + h_range, start_w:start_w + w_range] = label_array[:h_range, :w_range]
            image_array, label_array = temp_image, temp_label
        if nd >= target_size[2]:
            sd = int((nd - target_size[2]) / 2)
            image_array = image_array[..., sd:sd + target_size[2]]
            label_array = label_array[..., sd:sd + target_size[2]]
        else:
            sd = int((target_size[2] - nd) / 2)
            temp_image, temp_label = np.zeros(target_size), np.zeros(target_size)
            temp_image[..., sd:sd + nd] = image_array
            temp_label[..., sd:sd + nd] = label_array
            image_array, label_array = temp_image, temp_label
        return image_array, label_array

    ed_paths = get_file_paths(ed_frame)
    ed_image = read_and_resample(ed_paths[0], is_label=False)
    ed_label = read_and_resample(ed_paths[1], is_label=True)
    ed_image, ed_label = process_single_frame(ed_image, ed_label)
    
    es_paths = get_file_paths(es_frame)
    es_image = read_and_resample(es_paths[0], is_label=False)
    es_label = read_and_resample(es_paths[1], is_label=True)
    es_image, es_label = process_single_frame(es_image, es_label)
    
    return np.stack([ed_image, es_image], axis=-1), np.stack([ed_label, es_label], axis=-1)

def main(args, dataset, opt, label_value, test_num='001', label_id="label4", deform_model_path=None, result_path=None, data_root=None, acdc_info_path=None):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    writer = SummaryWriter(f"train_acdc_summary/{script_name}_{now}--{test_num}")
    K, deform_type, skinning, hyper_dim, node_num = 3, 'node', False, 8, 12288
    d_rot_as_res, d_rot_as_rotmat = True, False
    local_frame, no_arap_loss, max_d_scale, node_enable_densify_prune, is_scene_static = True, True, -1, False, False
    
    deform = DeformModel(K=K, deform_type=deform_type, skinning=skinning, hyper_dim=hyper_dim, node_num=node_num, 
                         d_rot_as_res=d_rot_as_res and not d_rot_as_rotmat, local_frame=local_frame, 
                         with_arap_loss=not no_arap_loss, max_d_scale=max_d_scale, 
                         enable_densify_prune=node_enable_densify_prune, is_scene_static=is_scene_static)
                         
    opt.position_lr_init, opt.position_lr_final, opt.position_lr_delay_steps, opt.position_lr_max_steps = 0.0001, 1e-07, 0, 30000
    opt.iterations_node_rendering, opt.opacity_lr, opt.densify_grad_threshold, opt.scaling_lr, opt.rotation_lr = 20000, 0.005, 0.0002, 0.0001, 0.0001
    opt.densification_interval, opt.densify_from_iter, opt.deform_lr_scale, opt.deform_lr_max_steps = 500, 500, 0.01, 40000
    opt.iterations_node_sampling, opt.iterations_node_rendering = 5000, 10000
    
    deform.train_setting(opt)
    deform_loaded = deform.load_weights(deform_model_path, iteration=800, descript='')
    image_size = (128, 128, 32)
    processed_images, processed_labels = process_ACDC(test_num, label_value, data_root, acdc_info_path)
    DyHeart = torch.tensor(processed_images, dtype=torch.float32)
    DyHeart = torch.clip(DyHeart, 0, 250)
    DyHeart = DyHeart.div(torch.max(DyHeart))[:,:,:,[0,-1]].cuda()
    total_frame = DyHeart.shape[-1]
    final_points_np = sample_points_from_cardiac(DyHeart[:, :, :, 0].detach().cpu().numpy(), node_num)
    final_points = torch.tensor(final_points_np, dtype=torch.float32).cuda()
    
    if deform.name == 'node' and not deform_loaded:
        print('Initialize nodes with Random point cloud.')
        deform.deform.init(init_pcl=final_points, force_init=True, opt=opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=skinning)
        
    iteration_node_rendering = 0
    frame_list = list(range(0, total_frame))
    best_node_valloss, best_node_iter, start_time = torch.inf, 0, time.time()
    
    for iter_c in range(20000):
        L1_accum_loss = torch.zeros(1).cuda()
        for j in range(total_frame):
            index = j
            fid = torch.tensor(index / total_frame).cuda()
            time_input = fid.unsqueeze(0).expand(deform.deform.as_gaussians.get_xyz.shape[0], -1)
            d_values = deform.deform.query_network(x=deform.deform.as_gaussians.get_xyz.detach(), t=time_input)
            d_xyz, d_density = d_values['d_xyz'], d_values['d_density']
            d_scaling = torch.mean(d_values['d_scaling'], dim=(1), keepdim=True).expand_as(d_values['d_scaling'])
            d_rotation = 0
            grid = create_grid_3d(*image_size).cuda().unsqueeze(0).repeat(1, 1, 1, 1, 1)
            train_output = rendering2(deform.deform.as_gaussians, grid, d_xyz, d_rotation, d_scaling, d_density)
            L1_p_loss = l1_loss(DyHeart[:, :, :, index], train_output.squeeze(0).squeeze(3))
            L1_accum_loss += L1_p_loss
            writer.add_scalar(f'train_loss/image_f{index}', L1_p_loss.item(), iteration_node_rendering)
        
        L1_accum_loss /= total_frame
        loss = L1_accum_loss
        loss.backward(retain_graph=True)
        
        writer.add_scalar('gradients/gaussian_xyz_grad_norm', torch.norm(deform.deform.as_gaussians.get_xyz.grad.detach(), p=2).item(), iteration_node_rendering)
        writer.add_scalar('learning_rate/deform_lr', deform.optimizer.param_groups[0]['lr'], iteration_node_rendering)
        writer.add_scalar('train_loss/overall', loss.item(), iteration_node_rendering)
        
        with torch.no_grad():
            if iteration_node_rendering < opt.iterations_node_sampling:
                if node_enable_densify_prune and iteration_node_rendering > opt.densify_from_iter and deform.deform.as_gaussians.get_xyz.shape[0] < 5000 and (iteration_node_rendering % 500 == 0):
                    deform.deform.as_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.0005, 1.5, surface_points=None, max_distance=0.5)
            elif iteration_node_rendering == opt.iterations_node_sampling:
                strategy = opt.deform_downsamp_strategy
                if strategy == 'direct':
                    original_gaussians = deform.deform.as_gaussians
                    deform.deform.init(init_pcl=original_gaussians.get_xyz, force_init=True, opt=opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=skinning)
                    gaussians = deform.deform.as_gaussians
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling)
                    gaussians._density = torch.nn.Parameter(original_gaussians._density)
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation)
                    deform.train_setting(opt)
            
            if iteration_node_rendering == opt.iterations_node_rendering - 1 and iteration_node_rendering > opt.iterations_node_sampling:
                deform.deform.nodes.data[..., :3] = deform.deform.as_gaussians._xyz
                
            if not iteration_node_rendering == opt.iterations_node_sampling and not iteration_node_rendering == opt.iterations_node_rendering - 1:
                deform.deform.as_gaussians.optimizer.step()
                deform.deform.as_gaussians.update_learning_rate(iteration_node_rendering)
                deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                deform.update_learning_rate(iteration_node_rendering)
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                
            iteration_node_rendering += 1
            deform.update(max(0, iteration_node_rendering - opt.node_warm_up))
            
        if iter_c % 50 == 0:
            val_loss, dice, dice_split, psnrt = val_4ddata(total_frame, frame_list, deform, deform.deform.as_gaussians, DyHeart.detach().cpu(), image_size, iteration_node_rendering, best_node_valloss, test_num, label_id, result_path, False)
            writer.add_scalar('val_loss/overall', val_loss.item(), iteration_node_rendering)
            writer.add_scalar('metrics/dice_all', dice, iteration_node_rendering)
            if val_loss.item() < best_node_valloss:
                best_node_valloss, best_node_iter = val_loss.item(), iteration_node_rendering
                best_state_dict = {k: v.clone().detach() for k, v in deform.deform.state_dict().items()}
            print(f'iteration: {iteration_node_rendering} val_loss: {val_loss.item()}')
            
    end_time = time.time()
    deform.deform.load_state_dict(best_state_dict)
    val_loss, dice_v, dice_split, _ = val_4ddata(total_frame, frame_list, deform, deform.deform.as_gaussians, DyHeart.detach().cpu(), image_size, iteration_node_rendering, best_node_valloss, test_num, label_id, result_path, True)
    print(f"best iteration: {best_node_iter} best val_loss: {best_node_valloss} dice: {dice_v}")
    deform.save_weights(deform_model_path, iteration=best_node_iter, descript=f"{test_num}_acdc_paper_{label_id}_f2_")
    return end_time - start_time

if __name__ == "__main__":
    cpu_num = 5
    for env_var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env_var] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    
    parser = ArgumentParser(description="Training script parameters")
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 6000, 7000] + list(range(8000, 1000001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of ACDC dataset")
    parser.add_argument('--acdc_info', type=str, required=True, help="Path to ACDC_info.json")
    parser.add_argument('--result_path', type=str, required=True, help="Directory to save results")
    parser.add_argument('--label', type=int, default=3, choices=[1,2,3,4], help="Cardiac structure label")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    
    label_id, labelname, group_num = args.label, f"label{args.label}", 10
    patients = [str(i).zfill(3) for i in range(1 + group_num * args.group, 1 + group_num * (args.group + 1))]
    time_list = []
    for test_num in patients:
        time_con = main(args=args, dataset=lp.extract(args), opt=op.extract(args), label_value=label_id, test_num=test_num, label_id=labelname, deform_model_path=args.model_path, result_path=args.result_path, data_root=args.data_root, acdc_info_path=args.acdc_info)
        time_list.append(time_con)
    print("Average time: ", sum(time_list)/len(time_list))
