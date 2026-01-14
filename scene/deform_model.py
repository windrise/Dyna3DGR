import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, ControlNodeWarp, StaticNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, get_cosine_lr_func


model_dict = {'mlp': DeformNetwork, 'node': ControlNodeWarp, 'static': StaticNetwork}


class DeformModel:
    def __init__(self, deform_type='node', d_rot_as_res=True, opt=None, **kwargs):
        self.deform = model_dict[deform_type](d_rot_as_res=d_rot_as_res,opt=opt, **kwargs).cuda()
        self.name = self.deform.name
        self.optimizer = None
        self.spatial_lr_scale = 1
        self.d_rot_as_res = d_rot_as_res

    @property
    def reg_loss(self):
        return self.deform.reg_loss

    def step(self, xyz, time_emb, iteration=0, **kwargs):
        return self.deform(xyz, time_emb, iteration=iteration, **kwargs)

    def train_setting(self, training_args):
        l = []
        for group in self.deform.trainable_parameters():
            lr_scale = {
                'warp': 1.0,
                'scaling': 0.001,
                'rotation': 1.0,
                'backbone': 1.0,
            }.get(group['name'], 1.0)
            
            l.append({
                'params': group['params'],
                'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale * lr_scale,
                'name': group['name']
            })


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale, lr_final=training_args.position_lr_final * training_args.deform_lr_scale,  lr_delay_steps=training_args.position_lr_delay_steps, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

        if self.name == 'node':
            self.deform.as_gaussians_ca.training_setup(training_args)

    def save_weights(self, model_path, iteration, descript):
        out_weights_path = os.path.join(model_path, "deform/"+descript+"iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1, descript=None):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/"+descript+"iteration_{}/deform.pth".format(loaded_iter))
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            return True
        else:
            return False

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform" or param_group["name"] == "mlp" or 'node' in param_group['name']:
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def densify(self, max_grad, x, x_grad, **kwargs):
        if self.name == 'node':
            self.deform.densify(max_grad=max_grad, optimizer=self.optimizer, x=x, x_grad=x_grad, **kwargs)
        else:
            return
        
    def update(self, iteration):
        self.deform.update(iteration)
