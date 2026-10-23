'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
Single GPU version without Horovod support.
'''

import os, sys, copy, time, logging

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = PROJECT_ROOT
MODEL_ROOT = os.path.join(PROJECT_ROOT, "model")
UTILS_ROOT = os.path.join(PROJECT_ROOT, "utils")

for _path in (MODEL_ROOT, REPO_ROOT, UTILS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import DataLoader
import ModelLoader
import global_utils

#from mbv2_baseline import MobileNetV2, str_to_cfg
from mbv2_imgnt_baseline import MobileNetV2
import timm
from mbv2_timm import mbv2_timm

# from resnet import resnet18, resnet50
from parse_net_id import get_network_architecture


def parse_effnet_cfg_from_str(cfg_str):
    """Parse inline cfg_str like '1_16_1_1_3_0.25|6_24_1_2_3_0.25|...' into list of (t,c,n,s,k,se) tuples."""
    blocks = [b.strip() for b in cfg_str.strip().replace('|', ',').split(',') if b.strip()]
    cfg = []
    for b in blocks:
        parts = b.split('_')
        assert len(parts) == 6, f'Each EfficientNet block must have 6 values (t_c_n_s_k_se), got {parts} from "{b}"'
        t, se = float(parts[0]), float(parts[5])
        c, n, s, k = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        cfg.append((t, c, n, s, k, se))
    return cfg


def parse_mbv2_cfg_from_str(cfg_str):
    """Parse inline cfg_str like '1.0_16_1_1|6.0_24_2_2|...' into list of (t,c,n,s) tuples."""
    blocks = [b.strip() for b in cfg_str.strip().replace('|', ',').split(',') if b.strip()]
    cfg = []
    for b in blocks:
        parts = b.split('_')
        assert len(parts) == 4, f'Each block must have 4 values (t_c_n_s), got {parts} from "{b}"'
        t = float(parts[0])
        c, n, s = int(parts[1]), int(parts[2]), int(parts[3])
        cfg.append((t, c, n, s))
    return cfg


def parse_mbv2_cfg_from_txt(txt_path):
    """
    Parse a MobileNetV2 cfg from a text file.

    Expected format (single line), blocks separated by , or |:
        t_c_n_s,t_c_n_s,...  or  t_c_n_s|t_c_n_s|...
    t = expansion ratio (int or float, e.g. 1, 1.0, 4.1, 4.2)
    c = output channels, n = repeat count, s = stride

    Examples:
        Comma: 1_16_1_1,2_20_2_2,6_28_6_2,6_64_4_2,1_96_6_1,1_176_1_2,1_304_1_1
        Pipe:  1.0_16_1_1|4.1_24_3_2|4.2_32_4_2|4.2_64_5_2|4.2_96_4_1|4.2_160_4_2|4.2_320_2_1
    """
    assert os.path.isfile(txt_path), f'cfg txt file not found: {txt_path}'
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
    assert line, f'cfg txt file {txt_path} is empty'
    # Support both comma and pipe delimiters
    blocks = [b.strip() for b in line.replace('|', ',').split(',') if b.strip()]
    cfg = []
    for b in blocks:
        parts = b.split('_')
        assert len(parts) == 4, f'Each block must have 4 values (t_c_n_s), got {parts} from "{b}"'
        t = float(parts[0])  # expansion ratio can be float (e.g. 4.1, 4.2)
        c, n, s = int(parts[1]), int(parts[2]), int(parts[3])
        cfg.append((t, c, n, s))
    return cfg


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for ii, (image, target) in enumerate(data_loader):
            model(image)
            if ii>1:
                break


def save_checkpoint(checkpoint_filename, state_dict):
    #return
    save_dir = os.path.dirname(checkpoint_filename)
    base_filename = os.path.basename(checkpoint_filename)
    backup_filename = os.path.join(save_dir, base_filename + '.backup')

    global_utils.mkdir(save_dir)

    if os.path.isfile(checkpoint_filename):
        if os.path.isfile(backup_filename):
            os.remove(backup_filename)
        os.rename(checkpoint_filename, backup_filename)

    torch.save(state_dict, checkpoint_filename)
    if os.path.isfile(backup_filename):
        os.remove(backup_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':4g', disp_avg=True):
        self.name = name
        self.fmt = fmt
        self.disp_avg = disp_avg
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}{val' + self.fmt + '}'
        fmtstr = fmtstr.format(name=self.name, val=self.val)
        if self.disp_avg:
            fmtstr += '({avg' + self.fmt + '})'
            fmtstr = fmtstr.format(avg=self.avg)
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def split_weights(net):
    """split network weights into to categories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def network_weight_MSRAPrelu_init(net: nn.Module):
    # the gain of xavier_normal_ is computed from gain=magnitude * sqrt(3) where magnitude is 2/(1+0.25**2). [mxnet implementation]

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=3.26033)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    return net

def network_weight_xavier_init(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    return net

def network_weight_random_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def network_weight_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels) * 1e-3
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels) * 1e-3
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def network_weight_01_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels) * 0.1
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels) * 0.1
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def mixup(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)

def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y

def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def config_dist_env_and_opt(opt):
    opt = copy.copy(opt)

    # set world_size, gpu, global rank
    if opt.dist_mode == 'cpu':
        opt.gpu = None
        opt.world_size = 1
        opt.rank = 0
    elif opt.dist_mode == 'single':
        if opt.gpu is None:
            opt.gpu = 0
        opt.world_size = 1
        opt.rank = 0
        torch.cuda.set_device(opt.gpu)
    elif opt.dist_mode == 'auto':
        opt.AutoGPU = global_utils.AutoGPU()
        opt.gpu = opt.AutoGPU.gpu
        opt.world_size = 1
        opt.rank = 0
        torch.cuda.set_device(opt.gpu)
    elif opt.dist_mode == 'mpi':
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        opt.world_size = mpi_size
        opt.rank = mpi_rank
        opt.AutoGPU = global_utils.AutoGPU()
        opt.gpu = opt.AutoGPU.gpu
        torch.cuda.set_device(opt.gpu)
    else:
        raise ValueError('unknown dist_mode={}. Only cpu, single, auto, and mpi are supported.'.format(opt.dist_mode))

    if not opt.dist_mode == 'cpu':
        torch.backends.cudnn.benchmark = True

    # adjust batch_size and learning rate
    if opt.img_per_class is not None and opt.img_per_class != -1:
        # 5, 10, 20, 40, 80
        pass # not used
    elif opt.batch_size is None:
        opt.batch_size = opt.batch_size_per_gpu * opt.world_size
    opt.lr = opt.lr_per_256 * opt.batch_size * opt.grad_accum_steps / 256.0
    opt.target_lr = opt.target_lr_per_256 * opt.batch_size * opt.grad_accum_steps / 256.0

    return opt

def init_model(model, opt, argv):
    if opt.baseline:
        network_weight_xavier_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'xavier':
        network_weight_xavier_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'MSRAPrelu':
        network_weight_MSRAPrelu_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'random':
        network_weight_random_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'zero':
        network_weight_zero_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == '01':
        network_weight_01_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'custom':
        assert hasattr(model, 'init_parameters')
        model.init_parameters()
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'None':
        logging.info('Warning!!! model loaded without initialization !')
    else:
        raise ValueError('Unknown weight_init')

    if hasattr(opt, 'bn_momentum') and opt.bn_momentum is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = opt.bn_momentum

    if hasattr(opt, 'bn_eps') and opt.bn_eps is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = opt.bn_eps

    return model

def get_optimizer(model, opt):
    params = split_weights(model)
    print('opt lr=', opt.lr)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=opt.nesterov)
    elif opt.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(params,
                                         opt.lr,
                                         opt.adadelta_rho,
                                         opt.adadelta_eps,
                                         weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, opt.lr, weight_decay=opt.weight_decay)

    elif opt.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, opt.lr, alpha=0.9, momentum=opt.momentum, weight_decay=opt.weight_decay)

    else:
        raise ValueError('Unknown optimizer: ' + opt.optimizer)

    return optimizer

def load_model(model, load_parameters_from, strict_load=False, map_location='cpu'):
    logging.info('loading params from ' + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict_load)

    return model

def export_model_to_onnx(model, onnx_output_path, input_image_size, batch_size=1, num_channels=3):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model (should be in eval mode)
        onnx_output_path: Path where the ONNX model will be saved
        input_image_size: Size of input images (assumes square images)
        batch_size: Batch size for the exported model (default: 1)
        num_channels: Number of input channels (default: 3 for RGB)
    """
    model.eval()
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Create dummy input with the correct shape on the same device as the model
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    logging.info(f'Exporting model to ONNX: {onnx_output_path}')
    logging.info(f'Input shape: {dummy_input.shape}, Device: {device}')
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,  # store the trained parameter weights
            opset_version=12,    # ONNX opset version
            do_constant_folding=True,  # execute constant folding optimization
            input_names=['input'],
            output_names=['output'],
            # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # variable batch size
        )
    
    logging.info(f'Successfully exported model to ONNX: {onnx_output_path}')
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        logging.info('ONNX model verification passed')
    except ImportError:
        logging.warning('onnx package not available, skipping model verification')
    except Exception as e:
        logging.warning(f'ONNX model verification failed: {e}')

def export_model_to_torchscript(model, torchscript_output_path, input_image_size, batch_size=1, num_channels=3):
    """
    Export a PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model (should be in eval mode)
        torchscript_output_path: Path where the TorchScript model will be saved
        input_image_size: Size of input images (assumes square images)
        batch_size: Batch size for the exported model (default: 1)
        num_channels: Number of input channels (default: 3 for RGB)
    """
    model.eval()
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Create dummy input with the correct shape on the same device as the model
    dummy_input = torch.randn(batch_size, num_channels, input_image_size, input_image_size, device=device)
    
    logging.info(f'Exporting model to TorchScript: {torchscript_output_path}')
    logging.info(f'Input shape: {dummy_input.shape}, Device: {device}')
    
    # Export to TorchScript using tracing
    with torch.no_grad():
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(torchscript_output_path)
            logging.info(f'Successfully exported model to TorchScript: {torchscript_output_path}')
            
            # Verify the exported model by loading it
            try:
                loaded_model = torch.jit.load(torchscript_output_path)
                test_output = loaded_model(dummy_input)
                logging.info(f'TorchScript model verification passed. Output shape: {test_output.shape}')
            except Exception as e:
                logging.warning(f'TorchScript model verification failed: {e}')
        except Exception as e:
            logging.error(f'Failed to export model to TorchScript: {e}')
            raise

def resume_checkpoint(model, optimizer, checkpoint_filename, opt, map_location='cpu'):
    logging.info('resuming from ' + checkpoint_filename)
    checkpoint = torch.load(checkpoint_filename, map_location=map_location)
    assert 'state_dict' in checkpoint
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)

    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        logging.warning('No optimizer state dict found in checkpoint; skipping optimizer loading.')
        optimizer = None

    try:
        training_status_info = checkpoint['training_status_info']
    except:
        logging.warning('No training status info found in checkpoint; skipping training status info loading.')
        training_status_info = None

    try:
        opt.start_epoch = checkpoint['epoch'] + 1
    except:
        logging.warning('No epoch found in checkpoint; skipping epoch loading.')
        opt.start_epoch = 0

    return model, optimizer, training_status_info, opt

def config_model_optimizer_apex(model, optimizer, opt):
    """Configure model and optimizer for apex (mixed precision training)."""
    if opt.apex:
        from apex import amp
        if opt.apex_loss_scale != 'dynamic':
            apex_loss_scale = float(opt.apex_loss_scale)
        else:
            apex_loss_scale = 'dynamic'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.apex_opt_level, loss_scale=apex_loss_scale)

    return model, optimizer

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt, num_train_samples, no_acc_eval=False):
    info = {}

    losses = AverageMeter('Loss ', ':6.4g')
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')
    grad_norm_meter = AverageMeter('GradNorm', ':6.4g')
    # switch to train mode
    model.train()

    lr_scheduler = global_utils.LearningRateScheduler(mode=opt.lr_mode,
                                                      lr=opt.lr,
                                                      num_training_instances=num_train_samples,
                                                      target_lr=opt.target_lr,
                                                      stop_epoch=opt.epochs,
                                                      warmup_epoch=opt.warmup,
                                                      stage_list=opt.lr_stage_list,
                                                      stage_decay=opt.lr_stage_decay)
    lr_scheduler.update_lr(batch_size=epoch * num_train_samples)

    optimizer.zero_grad()
    if opt.use_tqdm:
        _train_loader = tqdm(train_loader, desc=f"Training epoch {epoch}", leave=False)
    else:
        _train_loader = train_loader
    for i, (input, target) in enumerate(_train_loader):

        if not opt.independent_training:
            lr_scheduler.update_lr(batch_size=input.shape[0] * opt.world_size)
        else:
            lr_scheduler.update_lr(batch_size=input.shape[0])
        pass  # end if
        current_lr = lr_scheduler.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        bool_label_smoothing = False
        bool_mixup = False

        if not opt.dist_mode == 'cpu':
            input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

        transformed_target = target

        with torch.no_grad():
            if hasattr(opt, 'label_smoothing') and opt.label_smoothing:
                bool_label_smoothing = True
            if hasattr(opt, 'mixup') and opt.mixup:
                bool_mixup = True

            if bool_label_smoothing and not bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes, smoothing_eps=0.1)

            if not bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes)
                input, transformed_target = mixup(input, transformed_target)

            if bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes, smoothing_eps=0.1)
                input, transformed_target = mixup(input, transformed_target)

        pass  # end with

        # compute output
        if opt.fp16:
            output = model(input.half())
        else:
            output = model(input)
        loss = criterion(output, transformed_target)

        # measure accuracy and record loss
        input_size = int(input.size(0))
        if not no_acc_eval:
            if opt.num_classes < 5:
                acc1,  = accuracy(output, target, topk=(1, ))
                acc5 = [0]
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(float(acc1[0]), input_size)
            top5.update(float(acc5[0]), input_size)
        else:
            acc1 = [0]
            acc5 = [0]

        losses.update(float(loss), input_size)

        if opt.apex:
            from apex import amp
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                # Compute total gradient L2 norm (before clipping) for logging
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                grad_norm_meter.update(total_norm, input_size)
                if opt.grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.grad_clip)
            optimizer.step()
        else:
            # Gradient accumulation: scale loss, step only every grad_accum_steps
            accum_steps = getattr(opt, 'grad_accum_steps', 1)
            loss = loss / accum_steps
            loss.backward()
            # Compute total gradient L2 norm (before clipping) for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm_meter.update(total_norm, input_size)
            if (i + 1) % accum_steps == 0:
                if opt.grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        if i % opt.print_freq == 0 and opt.rank == 0:
            logging.info('Train epoch={}, i={}, loss={:4g}, acc1={:4g}%, acc5={:4g}%, lr={:4g}'.format(
                epoch, i, float(loss), float(acc1[0]), float(acc5[0]), current_lr
            ))
        pass  # end if
    pass  # end for i

    # Single GPU - no sync needed
    losses_avg = losses.avg
    info['losses'] = losses_avg
    info['grad_norm'] = grad_norm_meter.avg
    info['acc1'] = top1.avg

    return info


def validate(val_loader, model, criterion, opt, epoch='N/A', set_name='val'):
    """Evaluate model on a data loader. set_name is used for logging (e.g. 'val' or 'test')."""
    losses = AverageMeter('Loss ', ':6.4g')
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            transformed_target = target
            if (hasattr(opt, 'label_smoothing') and opt.label_smoothing) or (hasattr(opt, 'mixup') and opt.mixup):
                transformed_target = one_hot(transformed_target, num_classes=opt.num_classes, smoothing_eps=None)
            if not opt.dist_mode == 'cpu':
                input = input.cuda(opt.gpu, non_blocking=True)
                target = target.cuda(opt.gpu, non_blocking=True)
                transformed_target = transformed_target.cuda(opt.gpu, non_blocking=True)
            # compute output
            if opt.fp16:
                output = model(input.half())
            else:
                output = model(input)
            if criterion is not None:
                loss = criterion(output, transformed_target)
            else:
                loss = torch.tensor([0])

            # measure accuracy and record loss

            acc1,  = accuracy(output, target, topk=(1, ))
            acc5 = [0]

            input_size = int(input.size(0))
            losses.update(float(loss), input_size)
            top1.update(float(acc1[0]), input_size)
            top5.update(float(acc5[0]), input_size)

            if i % opt.print_freq == 0 and opt.rank == 0:
                logging.info('Eval [{}] epoch={}, i={}, loss={:4g}, acc1={:4g}%, acc5={:4g}%'.format(
                    set_name, epoch, i, float(loss), float(acc1[0]), float(acc5[0])))
        pass  # end for
    pass  # end with

    top1_acc_avg = top1.avg
    top5_acc_avg = top5.avg

    total_val_count = top1.count

    # Single GPU - no sync needed
    logging.info(' * [{}] Validate Acc@1 {:.3f} Acc@5 {:.3f}, n={}'.format(set_name, top1_acc_avg, top5_acc_avg, total_val_count))

    return {'top1_acc': top1_acc_avg, 'top5_acc': top5_acc_avg}


def train_all_epochs(opt, model, optimizer, train_sampler, train_loader, criterion, val_loader, test_loader=None,
                     num_train_samples=None, no_acc_eval=False, save_all_ranks=False, training_status_info=None, save_params=True):
    """When validation runs, both val and test sets are evaluated. Best model selection is based on val set Acc@1."""
    timer_start = time.time()

    if training_status_info is None:
        training_status_info = {}
    training_status_info.setdefault('best_acc1', 0)
    training_status_info.setdefault('best_acc5', 0)
    training_status_info.setdefault('best_acc1_at_epoch', 0)
    training_status_info.setdefault('best_acc5_at_epoch', 0)
    training_status_info.setdefault('training_elasped_time', 0)
    training_status_info.setdefault('validation_elasped_time', 0)
    training_status_info.setdefault('best_test_acc1', 0)
    training_status_info.setdefault('best_test_acc1_at_epoch', 0)

    if num_train_samples is None:
        num_train_samples = len(train_loader)
    acc1 = 0
    acc5 = 0
    test_acc1 = 0
    test_acc5 = 0
    total_epochs = round(opt.epochs / opt.epfrac)

    for epoch in range(opt.start_epoch, total_epochs+1) :
        logging.info('--- Start training epoch {} / {}'.format(epoch, total_epochs))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        training_timer_start = time.time()
        if epoch < total_epochs: #
            train_one_epoch_info = train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt, num_train_samples,
                                               no_acc_eval=no_acc_eval)

            # Write average train metrics to CSV file
            if opt.rank == 0:
                metrics_file = os.path.join(opt.save_dir, 'train_metrics.csv')
                write_header = not os.path.isfile(metrics_file)
                with open(metrics_file, 'a') as f:
                    if write_header:
                        f.write('epoch,train_loss,grad_norm,accuracy\n')
                    f.write('{},{:.6g},{:.6g},{:.6g}\n'.format(
                        epoch,
                        train_one_epoch_info['losses'],
                        train_one_epoch_info['grad_norm'],
                        train_one_epoch_info['acc1']))

        training_status_info['training_elasped_time'] += time.time() - training_timer_start

        # evaluate on validation set (and test set when available)
        if epoch > (total_epochs-10) and val_loader is not None:
        # if epoch%opt.val_freq==0 and  val_loader is not None:
            validation_timer_start = time.time()

            validate_info = validate(val_loader, model, criterion, opt, epoch=epoch, set_name='val')
            acc1 = validate_info['top1_acc']
            acc5 = validate_info['top5_acc']

            if test_loader is not None:
                test_validate_info = validate(test_loader, model, criterion, opt, epoch=epoch, set_name='test')
                test_acc1 = test_validate_info['top1_acc']
                test_acc5 = test_validate_info['top5_acc']
                training_status_info['best_test_acc1'] = max(test_acc1, training_status_info['best_test_acc1'])
                if test_acc1 >= training_status_info['best_test_acc1']:
                    training_status_info['best_test_acc1_at_epoch'] = epoch

            training_status_info['validation_elasped_time'] += time.time() - validation_timer_start

            # Consolidated val+test summary for easy visibility
            if opt.rank == 0:
                if test_loader is not None:
                    logging.info('=== Epoch {} | val Acc@1={:.3f} Acc@5={:.3f} | test Acc@1={:.3f} Acc@5={:.3f} ==='.format(
                        epoch, acc1, acc5, test_acc1, test_acc5))
                else:
                    logging.info('=== Epoch {} | val Acc@1={:.3f} Acc@5={:.3f} ==='.format(epoch, acc1, acc5))

        # remember best acc@1 and save checkpoint
        is_best_acc1 = acc1 > training_status_info['best_acc1']
        is_best_acc5 = acc5 > training_status_info['best_acc5']
        training_status_info['best_acc1'] = max(acc1, training_status_info['best_acc1'])
        training_status_info['best_acc5'] = max(acc5, training_status_info['best_acc5'])

        if is_best_acc1:
            training_status_info['best_acc1_at_epoch'] = epoch
        if is_best_acc5:
            training_status_info['best_acc5_at_epoch'] = epoch

        elasped_hour = (time.time() - timer_start) / 3600
        remaining_hour = (time.time() - timer_start) / float(epoch - opt.start_epoch + 1) * (opt.epochs - epoch) / 3600
        log_msg = (
            '--- Epoch={}, Elasped hour={:8.4g}, Remaining hour={:8.4g}, Training Speed={:4g},'
            ' best_val_acc1={:4g}, best_val_acc1_at_epoch={}, best_val_acc5={}, best_val_acc5_at_epoch={}'.format(
                epoch, elasped_hour, remaining_hour,
                num_train_samples * (epoch + 1) / float(training_status_info['training_elasped_time'] + 1e-8),
                training_status_info['best_acc1'], training_status_info['best_acc1_at_epoch'],
                training_status_info['best_acc5'], training_status_info['best_acc5_at_epoch']
            ))
        if test_loader is not None and 'best_test_acc1' in training_status_info:
            log_msg += ', best_test_acc1={:4g}, best_test_acc1_at_epoch={}'.format(
                training_status_info['best_test_acc1'], training_status_info['best_test_acc1_at_epoch'])
        logging.info(log_msg)

        # ----- save latest epoch -----#
        if save_params and (opt.rank == 0 or save_all_ranks) and \
                ((epoch + 1) % opt.save_freq == 0 or epoch + 1 == opt.epochs):
            checkpoint_filename = os.path.join(opt.save_dir, 'latest-params_rank{}.pth'.format(opt.rank))
            save_checkpoint(checkpoint_filename, {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'top1_acc': acc1,
                'top5_acc': acc5,
                'top1_test_acc': test_acc1 if test_loader is not None else None,
                'top5_test_acc': test_acc5 if test_loader is not None else None,
                'training_status_info': training_status_info
            })

        # ----- save best parameters -----#
        if save_params and is_best_acc1 and (opt.rank == 0 or save_all_ranks):
            checkpoint_filename = os.path.join(opt.save_dir, 'best-params_rank{}.pth'.format(opt.rank))
            save_checkpoint(checkpoint_filename, {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'top1_acc': acc1,
                'top5_acc': acc5,
                'top1_test_acc': test_acc1 if test_loader is not None else None,
                'top5_test_acc': test_acc5 if test_loader is not None else None,
                'training_status_info': training_status_info
            })

    pass  # end for epoch in range(opt.start_epoch, opt.epochs):

    return training_status_info

def build_model(opt):
    """
    Unified model builder for MobileNetV2, ResNet, and ZEN models.
    """
    if hasattr(opt, 'model_type'):
        # if opt.model_type.lower() == 'resnet18':
        #     model = resnet18(num_classes=opt.num_classes, weights=None)
        # elif opt.model_type.lower() == 'resnet50':
        #     model = resnet50(num_classes=opt.num_classes, weights=None)
        # elif 'resnet' in opt.model_type.lower() and not "timm" in opt.model_type.lower():
        #     assert hasattr(opt, 'arch_txt')
        #     # arch_txt e.g., "64,128,256,512"
        #     layer_widths = [int(ii) for ii in opt.arch_txt.split(',')]
        # model = my_resnet(name=opt.model_type.lower(), num_classes=opt.num_classes, layer_widths=layer_widths)
        if opt.model_type.lower() == 'mobilenet':
            _wm = getattr(opt, 'width_mult', 1.0)
            if getattr(opt, 'cfg_str', None) is not None:
                # Build from inline cfg_str (pipe- or comma-separated stages)
                print(f'[build_model] Using inline cfg_str, width_mult={_wm}')
                cfg = parse_mbv2_cfg_from_str(opt.cfg_str)
                model = mbv2_timm(num_classes=opt.num_classes, cfg=cfg, img_size=opt.input_image_size, wm=_wm)
            elif getattr(opt, 'cfg_txt_path', None) is not None:
                # Build from cfg txt file (t_c_n_s,t_c_n_s,... format)
                print(f'[build_model] Using cfg txt file: {opt.cfg_txt_path}, width_mult={_wm}')
                cfg = parse_mbv2_cfg_from_txt(opt.cfg_txt_path)
                model = mbv2_timm(num_classes=opt.num_classes, cfg=cfg, img_size=opt.input_image_size, wm=_wm)
            else:
                # Build from net_id + CSV
                assert hasattr(opt, 'net_id'), "net_id or cfg_txt_path is required for mobilenet model_type"
                csv_file_path = getattr(opt, 'csv_file_path', 'sampled_1k_15ms_a57.csv')
                print(f'[build_model] Using CSV file: {csv_file_path}')
                ncfg = get_network_architecture(opt.net_id, csv_file_path=csv_file_path)
                if opt.stem_head_size is not None:
                    stem_head_size = int(opt.stem_head_size)
                else:
                    stem_head_size = None
                model = mbv2_timm(num_classes=opt.num_classes, cfg=ncfg, img_size=opt.input_image_size, stem_head_size=stem_head_size)
        elif opt.model_type.lower() == 'efficientnet':
            _wm = getattr(opt, 'width_mult', 1.0)
            assert getattr(opt, 'cfg_str', None) is not None, \
                "cfg_str is required for efficientnet model_type"
            print(f'[build_model] EfficientNet from cfg_str, width_mult={_wm}')
            cfg = parse_effnet_cfg_from_str(opt.cfg_str)
            from efficientnet_timm import efficientnet_timm
            model = efficientnet_timm(cfg=cfg, num_classes=opt.num_classes,
                                      wm=_wm, img_size=opt.input_image_size)
        elif opt.model_type.lower() == 'zen':
            model = ModelLoader.get_model(opt, sys.argv)
            # Note: init_model will be called separately in main()
        elif "timm" in opt.model_type.lower():
            _timm_model_name = opt.model_type.lower().split('timm_')[1]
            model = timm.create_model(_timm_model_name, num_classes=opt.num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown model_type: {opt.model_type}")
    else:
        # Fallback: if model_type is not specified, try to use ModelLoader
        # This maintains backward compatibility
        model = ModelLoader.get_model(opt, sys.argv)
    return model

def main(opt, argv):
    assert opt.save_dir is not None
    # Early-exit check: skip if job already in summary (net_id mode only)
    if getattr(opt, 'rank', 0) == 0 and getattr(opt, 'net_id', None) is not None:
        try:
            with open(opt.summary_file, 'r') as f:
                lins = f.readlines()
                for line in lins:
                    _latency = opt.csv_file_path.split('/')[-1].split('ms')[0].split('_')[-1] if hasattr(opt, 'csv_file_path') and opt.csv_file_path else 'unknown'
                    target_line = f'{opt.net_id},{opt.dataset},{opt.frac},{opt.epfrac},{opt.expid},{_latency}'
                    if target_line in line:
                        print(f'Net ID: {opt.net_id} Dataset: {opt.dataset} Fraction: {opt.frac} Epoch Fraction: {opt.epfrac} Latency: {_latency} Best Acc@1: {line.split(",")[-1]}')
                        return
                    if opt.save_dir in line:
                        print(f'Save directory: {opt.save_dir} is in line: {line}')
                        return
        except FileNotFoundError:
            pass

    opt = config_dist_env_and_opt(opt)

    # create log
    if opt.rank == 0:
        log_filename = os.path.join(opt.save_dir, 'train_image_classification.log')
        print('log_filename:', log_filename)
        global_utils.create_logging(log_filename=log_filename)
    else:
        global_utils.create_logging(log_filename=None, level=logging.ERROR)

    logging.info('argv=\n' + str(argv))
    logging.info('opt=\n' + str(opt))
    logging.info('-----')
    if os.path.exists(opt.completion_file) and not opt.export_onnx:
        print(f'===>>completion_file: {opt.completion_file} exists')
        return
    # load dataset
    data_loader_info = DataLoader.get_data(opt, argv)
    train_loader = data_loader_info['train_loader']
    print(f'===>>train_loader length: {len(train_loader)}')
    val_loader = data_loader_info['val_loader']
    test_loader = data_loader_info['test_loader']
    train_sampler = data_loader_info['train_sampler']
    num_train_samples = DataLoader.params_dict[opt.dataset]['num_train_samples']
    print(f'===>>train_dataset length: {len(train_loader.dataset)}')
    # If num_train_samples is 0 or None, get it from the dataset
    if num_train_samples == 0 or num_train_samples is None:
        # Get dataset length (works for both regular datasets and Subset wrappers)
        num_train_samples = len(train_loader.dataset)
        logging.info(f'[DataLoader] num_train_samples not in params_dict, using dataset length: {num_train_samples}')

    # create model
    model = build_model(opt)
    model = init_model(model, opt, argv)

    logging.info('loading model:')
    logging.info(str(model))

    # Optional weight initialization from a directory containing best checkpoints
    pretrained_path = None
    # if opt.pretrained_dir:
    #     candidate_paths = [
    #         os.path.join(opt.pretrained_dir, 'best-params_rank0.pth'),
    #         os.path.join(opt.pretrained_dir, 'best-params_rank1.pth'),
    #         os.path.join(opt.pretrained_dir, 'best-params.pth'),
    #     ]
    #     pretrained_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    #     if pretrained_path is None:
    #         logging.warning(f'pretrained_dir={opt.pretrained_dir} is set but no best-params_*.pth found.')
    if pretrained_path:
        map_location_pretrain = 'cpu'
        if opt.gpu is not None:
            map_location_pretrain = f'cuda:{opt.gpu}'
        if opt.load_parameters_from is not None:
            logging.warning('Both load_parameters_from and pretrained_dir are set; using pretrained_dir for initialization.')
        logging.info(f'=========>Initializing weights from {pretrained_path}')
        time.sleep(10)
        model = load_model(model, pretrained_path, opt.strict_load, map_location=map_location_pretrain)
    elif opt.load_parameters_from:
        model = load_model(model, opt.load_parameters_from, opt.strict_load, map_location='cpu')

    if opt.fp16:
        model = model.half()

    # set device
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model.cuda(opt.gpu)
        logging.info('rank={}, using GPU {}'.format(opt.rank, opt.gpu))

    # Handle ONNX export mode (after device setup)
    if hasattr(opt, 'export_onnx') and opt.export_onnx:
        # Find and load the best checkpoint
        best_checkpoint_path = os.path.join(opt.save_dir, 'best-params_rank0.pth')
        if not os.path.isfile(best_checkpoint_path):
            # Try alternative checkpoint names
            best_checkpoint_path = os.path.join(opt.save_dir, 'best-params.pth')
            if not os.path.isfile(best_checkpoint_path):
                raise FileNotFoundError(f'Best checkpoint not found in {opt.save_dir}. Expected: best-params_rank0.pth or best-params.pth')
        
        map_location = 'cpu'
        if opt.gpu is not None:
            map_location = 'cuda:{}'.format(opt.gpu)
        
        logging.info(f'Loading best checkpoint for ONNX export: {best_checkpoint_path}')
        model = load_model(model, best_checkpoint_path, strict_load=False, map_location=map_location)
        
        # Ensure model is on the correct device
        if opt.gpu is not None:
            model.cuda(opt.gpu)
        
        # Set model to eval mode
        model.eval()
        
        # Determine output paths
        if opt.onnx_output_path is None:
            onnx_output_path = os.path.join(opt.save_dir, 'model.onnx')
        else:
            onnx_output_path = opt.onnx_output_path
        
        # Derive TorchScript output path from ONNX path
        if onnx_output_path.endswith('.onnx'):
            torchscript_output_path = onnx_output_path.replace('.onnx', '.pt')
        else:
            torchscript_output_path = onnx_output_path + '.pt'
        
        # Create output directory if needed
        onnx_dir = os.path.dirname(onnx_output_path)
        if onnx_dir:
            global_utils.mkdir(onnx_dir)
        torchscript_dir = os.path.dirname(torchscript_output_path)
        if torchscript_dir:
            global_utils.mkdir(torchscript_dir)
        
        # Export to ONNX
        export_model_to_onnx(model, onnx_output_path, opt.input_image_size, batch_size=1, num_channels=3)
        
        # Export to TorchScript
        export_model_to_torchscript(model, torchscript_output_path, opt.input_image_size, batch_size=1, num_channels=3)
        
        logging.info(f'Export completed. ONNX model saved to: {onnx_output_path}')
        logging.info(f'TorchScript model saved to: {torchscript_output_path}')
        return

    # define loss function (criterion)
    if (hasattr(opt, 'label_smoothing') and opt.label_smoothing) or (hasattr(opt, 'mixup') and opt.mixup):
        criterion = cross_entropy
    else:
        criterion = nn.CrossEntropyLoss()
        if not opt.dist_mode == 'cpu':
            criterion = criterion.cuda(opt.gpu)

    # get optimizer
    optimizer = get_optimizer(model, opt)

    logging.info('optimizer is :')
    logging.info(str(optimizer))

    # apex (mixed precision training)
    model, optimizer = config_model_optimizer_apex(model, optimizer, opt)

    training_status_info = {}
    training_status_info['best_acc1'] = 0
    training_status_info['best_acc5'] = 0
    training_status_info['best_acc1_at_epoch'] = 0
    training_status_info['best_acc5_at_epoch'] = 0
    training_status_info['training_elasped_time'] = 0
    training_status_info['validation_elasped_time'] = 0

    map_location = 'cpu'
    if opt.gpu is not None:
        map_location = 'cuda:{}'.format(opt.gpu)

    if opt.auto_resume and opt.resume is None:
        latest_pth_fn = os.path.join(opt.save_dir, 'latest-params_rank0.pth')
        if os.path.isfile(latest_pth_fn):
            logging.info(('auto-resume from ' + latest_pth_fn))
            model, optimizer, training_status_info, opt = resume_checkpoint(model, optimizer, latest_pth_fn, opt,
                                                                            map_location=map_location)
    if opt.resume:
        assert not opt.auto_resume
        logging.info(('resume from ' + opt.resume))

        model, optimizer, training_status_info, opt = resume_checkpoint(model, optimizer, opt.resume, opt,
                                                                        map_location=map_location)

    if not opt.evaluate_only:
        training_status_info = train_all_epochs(opt, model, optimizer, train_sampler, train_loader, criterion,
                         val_loader, test_loader=test_loader,
                         num_train_samples=num_train_samples,
                         no_acc_eval=False, save_all_ranks=False,
                         training_status_info=training_status_info)
    else:
        # evaluate_only: run on BOTH val and test sets
        val_acc1 = val_acc5 = test_acc1 = test_acc5 = None
        if val_loader is not None:
            logging.info('=== Evaluating on VAL set ===')
            val_info = validate(val_loader, model, criterion, opt, set_name='val')
            val_acc1, val_acc5 = val_info['top1_acc'], val_info['top5_acc']
        if test_loader is not None:
            logging.info('=== Evaluating on TEST set ===')
            test_info = validate(test_loader, model, criterion, opt, set_name='test')
            test_acc1, test_acc5 = test_info['top1_acc'], test_info['top5_acc']
        if val_loader is None and test_loader is None:
            logging.warning('Neither val_loader nor test_loader available for evaluation.')
        elif opt.rank == 0:
            parts = []
            if val_acc1 is not None:
                parts.append('val Acc@1={:.3f} Acc@5={:.3f}'.format(val_acc1, val_acc5))
            if test_acc1 is not None:
                parts.append('test Acc@1={:.3f} Acc@5={:.3f}'.format(test_acc1, test_acc5))
            if parts:
                logging.info('=== EVAL SUMMARY | {} ==='.format(' | '.join(parts)))

    # mark job done
    # write to summary file
    if not os.path.exists(opt.summary_file):
        with open(opt.summary_file, 'w') as f:
            f.write('')
    if opt.rank == 0:
        if opt.model_type.lower() == 'zen':
            try:
                _latency = opt.plainnet_struct_txt.split('/')[-1].split('.')[0].split('_')[-1]
                with open(opt.summary_file, 'a') as f:
                    f.write(f'{opt.plainnet_struct_txt},{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
                with open(opt.completion_file, 'a') as f:
                    f.write(f'{opt.plainnet_struct_txt},{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
            except:
                with open(opt.summary_file, 'a') as f:
                    f.write(f'{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
                with open(opt.completion_file, 'a') as f:
                    f.write(f'{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
        elif 'timm' in opt.model_type.lower():
            with open(opt.summary_file, 'a') as f:
                f.write(f'{opt.model_type.lower()},{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
            with open(opt.completion_file, 'a') as f:
                f.write(f'{opt.model_type.lower()},{opt.dataset},{opt.frac},{opt.epfrac},-1,{training_status_info["best_acc1"]}\n')
        elif opt.model_type.lower() == 'efficientnet':
            expid = getattr(opt, 'expid', 0)
            cfg_str = getattr(opt, 'cfg_str', 'unknown')
            with open(opt.summary_file, 'a') as f:
                f.write(f'{cfg_str},{opt.dataset},{opt.frac},{opt.epfrac},{expid},{training_status_info["best_acc1"]}\n')
            with open(opt.completion_file, 'a') as f:
                f.write(f'{cfg_str},{opt.dataset},{opt.frac},{opt.epfrac},{expid},{training_status_info["best_acc1"]}\n')
        elif opt.model_type.lower() == 'mobilenet' and getattr(opt, 'cfg_txt_path', None) is not None:
            # Mobilenet from cfg txt file
            cfg_name = os.path.basename(opt.cfg_txt_path)
            latency = -1
            if hasattr(opt, 'cfg_txt_path') and opt.cfg_txt_path is not None:
                parts = opt.cfg_txt_path.split('/')
                for p in parts:
                    if 'ms' in p:
                        try:
                            latency = float(p.split('ms')[0].split('_')[-1])
                            break
                        except Exception:
                            pass
            with open(opt.summary_file, 'a') as f:
                f.write(f'{cfg_name},{opt.dataset},{opt.frac},{opt.epfrac},{latency},{training_status_info["best_acc1"]}\n')
            with open(opt.completion_file, 'a') as f:
                f.write(f'{cfg_name},{opt.dataset},{opt.frac},{opt.epfrac},{latency},{training_status_info["best_acc1"]}\n')
        # elif opt.model_type.lower() in ['resnet18', 'resnet50']:
        #     # ResNet models: model_name,dataset,frac,epfrac,expid,best_acc1
        #     expid = getattr(opt, 'expid', 0)
        #     with open(opt.summary_file, 'a') as f:
        #         f.write(f'{opt.model_type.lower()},{opt.dataset},{opt.frac},{opt.epfrac},{expid},{training_status_info["best_acc1"]}\n')
        else:
            # 1knets_30ms_top32_by_params.csv
            _latency = opt.csv_file_path.split('/')[-1].split('.csv')[0] if hasattr(opt, 'csv_file_path') and opt.csv_file_path else 'unknown'
            with open(opt.summary_file, 'a') as f:
                #  net_id_str, ds_str, frac_str, epfrac_str, expid_str 
                # f.write(f'{opt.save_dir},{opt.dataset},{opt.frac},{opt.epfrac},{_latency},{training_status_info["best_acc1"]}\n')
                f.write(f'{opt.net_id},{opt.dataset},{opt.frac},{opt.epfrac},{opt.expid},{_latency},{training_status_info["best_acc1"]}\n')
            with open(opt.completion_file, 'a') as f:
                f.write(f'{opt.net_id},{opt.dataset},{opt.frac},{opt.epfrac},{_latency},{training_status_info["best_acc1"]}\n')

if __name__ == "__main__":
    import argparse
    # Add net_id and csv_file_path arguments before parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_id', type=int, default=None, help='Network ID to parse from CSV file')
    parser.add_argument('--csv_file_path', type=str, default='sampled_1k_15ms_a57.csv', help='Path to CSV file containing network architectures')
    parser.add_argument('--cfg_txt_path', type=str, default=None,
                        help='Path to txt file with MobileNetV2 cfg (alternative to net_id). Format: t_c_n_s,t_c_n_s,... (7 blocks)')
    parser.add_argument('--cfg_str', type=str, default=None,
                        help='Inline MBV2 cfg string (pipe- or comma-separated stages, e.g. 1.0_16_1_1|6.0_24_2_2|...)')
    parser.add_argument('--width_mult', type=float, default=1.0,
                        help='Width multiplier for MobileNetV2 (passed as wm to mbv2_timm)')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--onnx_output_path', type=str, default=None, help='Path for the exported ONNX model (default: save_dir/model.onnx)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1 = no accumulation)')
    net_id_args, remaining_argv = parser.parse_known_args(sys.argv)

    # Parse remaining arguments using global_utils
    opt = global_utils.parse_cmd_options(remaining_argv)
    opt.net_id = net_id_args.net_id
    opt.csv_file_path = net_id_args.csv_file_path
    opt.cfg_txt_path = net_id_args.cfg_txt_path
    opt.cfg_str = net_id_args.cfg_str
    opt.width_mult = net_id_args.width_mult
    opt.grad_accum_steps = net_id_args.grad_accum_steps
    opt.export_onnx = net_id_args.export_onnx
    opt.onnx_output_path = net_id_args.onnx_output_path

    main(opt, sys.argv)

