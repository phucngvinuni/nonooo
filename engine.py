import torch
import math
import nltk
import torch.nn as nn
import sys

import numpy as np
from utils import *
from tqdm import tqdm
from PIL import Image
from wp_utils import *
from reg_attack import FastGradientSignUntargeted
from timm.data import Mixup
from einops import rearrange
from typing import Iterable, Optional
from timm.utils import accuracy, AverageMeter
from nltk.translate.bleu_score import sentence_bleu
from channel import channel, channel_Rayleigh, channel_Rician
from model_util import Channels
####################################
beta = 1.0

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

# @torch.no_grad()
# def evaluate(net: torch.nn.Module, dataloader: Iterable, 
#              device: torch.device, criterion: torch.nn.Module, train_type='fim', if_attack=False, print_freq=10):
#     net.eval()
#     acc_meter = AverageMeter()
#     loss_meter = AverageMeter()
#     psnr_meter = AverageMeter()
#     ssim_meter = AverageMeter()
    
    # attack = None
    # if if_attack and hasattr(dataloader.dataset, 'args') and dataloader.dataset.args.channel_type == 'none':
    #     attack = FGSM_REG(net, 12./255., 2./255., min_val=0, max_val=1, max_iters=8)
    
#     with torch.no_grad():
#         for batch_idx, (imgs, targets) in enumerate(dataloader):
#             imgs, bm_pos = imgs
#             original_imgs = imgs.clone().to(device)
#             imgs, targets = imgs.to(device), targets.to(device)
#             bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            
#             if if_attack and hasattr(dataloader.dataset, 'args'):
#                 channel_type = dataloader.dataset.args.channel_type
#                 snr_db = dataloader.dataset.args.snr_db
#                 if channel_type == 'awgn':
#                     imgs = torch.complex(imgs, torch.zeros_like(imgs))
#                     imgs = torch.from_numpy(channel(imgs.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#                 elif channel_type == 'rayleigh':
#                     imgs = torch.complex(imgs, torch.zeros_like(imgs))
#                     imgs = torch.from_numpy(channel_Rayleigh(imgs.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#                 elif channel_type == 'rician':
#                     imgs = torch.complex(imgs, torch.zeros_like(imgs))
#                     imgs = torch.from_numpy(channel_Rician(imgs.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#                 elif channel_type == 'none' and attack is not None:
#                     bum_pos = torch.zeros_like(bm_pos)
#                     bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
#                     imgs = attack.perturb(imgs, targets, bum_pos, 'mean', random_start=False, beta=beta)

#             psnr_values = calc_psnr(imgs.unbind(0), original_imgs.unbind(0)) if if_attack else [float('nan')] * imgs.size(0)
#             ssim_values = calc_ssim(imgs.unbind(0), original_imgs.unbind(0)) if if_attack else [float('nan')] * imgs.size(0)
            
#             outputs = net(img=imgs, bm_pos=bm_pos, targets=targets, _eval=True)
#             outputs_x = outputs['out_x']
#             loss = criterion(outputs_x, targets)
#             batch_size = targets.size(0)

#             idx, predicted = outputs_x.max(1)
#             acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
#             loss_meter.update(loss.item(), 1)
#             if if_attack:
#                 psnr_meter.update(sum(psnr_values) / len(psnr_values), n=batch_size)
#                 ssim_meter.update(sum(ssim_values) / len(ssim_values), n=batch_size)
            
#             if batch_idx % print_freq == 0:
#                 print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100] [psnr: %.2f] [ssim: %.4f] [Channel: %s] [SNR: %.1f dB]' 
#                       % (batch_idx * batch_size, len(dataloader.dataset), 
#                          loss_meter.avg, acc_meter.avg * 100, psnr_meter.avg, ssim_meter.avg,
#                          channel_type if if_attack else 'none', snr_db if if_attack else float('nan')))
    
#     test_stat = {
#         'loss': loss_meter.avg,
#         'acc': acc_meter.avg,
#         'psnr': psnr_meter.avg if if_attack else float('nan'),
#         'ssim': ssim_meter.avg if if_attack else float('nan'),
#         'channel': channel_type if if_attack and hasattr(dataloader.dataset, 'args') else 'none',
#         'snr': snr_db if if_attack and hasattr(dataloader.dataset, 'args') else float('nan')
#     }
#     return test_stat

@torch.no_grad()
def evaluate(net: torch.nn.Module, dataloader: Iterable, 
             device: torch.device, criterion: torch.nn.Module, train_type='fim', if_attack=False, print_freq=10):
    net.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    channel = Channels()
    attack = FastGradientSignUntargeted(net, epsilon=0.016, alpha=0.016/5, min_val=0, max_val=1, max_iters=5)
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs, bm_pos = imgs
            original_imgs = imgs.clone().to(device)
            imgs, targets = imgs.to(device), targets.to(device)
            bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            
            if if_attack:
                # bum_pos = torch.zeros_like(bm_pos).to(device, non_blocking=True).flatten(1).to(torch.bool)
                # imgs = attack.perturb(imgs, targets, bum_pos, 'mean', random_start=False, beta=beta)
                
                snr_db = 30  # 可调整为 -9 到 18 dB
                noise_var = torch.FloatTensor([1]).to(device) * 10**(-snr_db/20)
                imgs = channel.Rayleigh(imgs, noise_var.item())
            
            psnr_values = calc_psnr(imgs.unbind(0), original_imgs.unbind(0)) if if_attack else [float('nan')] * imgs.size(0)
            ssim_values = calc_ssim(imgs.unbind(0), original_imgs.unbind(0)) if if_attack else [float('nan')] * imgs.size(0)
            
            batch_size = imgs.size(0)
            outputs = net(img=imgs, bm_pos=bm_pos, targets=targets, _eval=True)
            outputs_x = outputs['out_x']
            loss = criterion(outputs_x, targets)
            
            idx, predicted = outputs_x.max(1)
            acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
            loss_meter.update(loss.item(), 1)
            if if_attack:
                psnr_meter.update(sum(psnr_values) / len(psnr_values), n=batch_size)
                ssim_meter.update(sum(ssim_values) / len(ssim_values), n=batch_size)
            
            if batch_idx % print_freq == 0:
                print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100] [psnr: %.2f] [ssim: %.4f] [SNR: %.1f dB]' 
                      % (batch_idx * batch_size, len(dataloader.dataset), 
                         loss_meter.avg, acc_meter.avg * 100, psnr_meter.avg, ssim_meter.avg, snr_db))
    
    test_stat = {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
        'psnr': psnr_meter.avg if if_attack else float('nan'),
        'ssim': ssim_meter.avg if if_attack else float('nan'),
        'snr': snr_db if if_attack else float('nan')
    }
    return test_stat


# engine.py (Corrected - Option 1)
def train_class_batch(model: torch.nn.Module, samples: torch.Tensor, targets: torch.Tensor,
                      bm_pos: torch.Tensor, criterion: torch.nn.Module, train_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes loss for a batch."""
    if train_type.startswith('std'):
        outputs = model(img=samples, bm_pos=bm_pos, targets=targets, _eval=False)
        outputs_x = outputs['out_x']
        loss = criterion(outputs_x, targets)
    elif train_type.startswith('fim'):
        outputs = model(img=samples, bm_pos=bm_pos, targets=targets, _eval=False)
        outputs_x = outputs['out_x']
        loss = criterion(outputs_x, targets)
        if 'out_c' in outputs:
            fim_loss = sum(F.cross_entropy(out, targets) for out in outputs['out_c']) / len(outputs['out_c'])
            loss += beta * fim_loss
        if 'vq_loss' in outputs:
            loss += outputs['vq_loss']
    else:  # <--- Add an else block
        raise ValueError(f"Invalid train_type: {train_type}.  Must start with 'std' or 'fim'.")

    return loss, outputs_x


# def train_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                 data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                 device: torch.device, epoch: int, loss_scaler, train_type, if_attack, max_norm: float=0,
#                 start_steps=None, lr_schedule_values=None, wd_schedule_values=None, 
#                 update_freq=None, print_freq=50):
#     model.train(True)
#     acc_meter = AverageMeter()
#     loss_meter = AverageMeter()
#     psnr_meter = AverageMeter()
#     ssim_meter = AverageMeter()
    
#     attack = None
#     if if_attack and hasattr(data_loader.dataset, 'args') and data_loader.dataset.args.channel_type == 'none':
#         attack = FGSM_REG(model, 8./255., 2./255., min_val=0, max_val=1, max_iters=4)

#     if loss_scaler is None:
#         model.zero_grad()
#         model.micro_steps = 0
#     else:
#         optimizer.zero_grad()

#     for data_iter_step, (samples, targets) in enumerate(data_loader):
#         step = data_iter_step // update_freq
#         it = start_steps + step
#         if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]
        
#         samples, bm_pos = samples
#         targets = targets.to(device, non_blocking=True)
#         original_samples = samples.clone().to(device, non_blocking=True)
#         samples = samples.to(device, non_blocking=True)
#         bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
#         if if_attack and hasattr(data_loader.dataset, 'args'):
#             channel_type = data_loader.dataset.args.channel_type
#             snr_db = data_loader.dataset.args.snr_db
#             if channel_type == 'awgn':
#                 samples = torch.complex(samples, torch.zeros_like(samples))
#                 samples = torch.from_numpy(channel(samples.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#             elif channel_type == 'rayleigh':
#                 samples = torch.complex(samples, torch.zeros_like(samples))
#                 samples = torch.from_numpy(channel_Rayleigh(samples.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#             elif channel_type == 'rician':
#                 samples = torch.complex(samples, torch.zeros_like(samples))
#                 samples = torch.from_numpy(channel_Rician(samples.cpu().numpy(), snr_db)).real.float().to(device)  # 转换为 float32
#             elif channel_type == 'none' and attack is not None:
#                 bum_pos = torch.zeros_like(bm_pos)
#                 bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
#                 samples = attack.perturb(samples, targets, bum_pos, 'mean', random_start=True, beta=beta)

#         batch_size = samples.size(0)
#         psnr_values = calc_psnr(samples.unbind(0), original_samples.unbind(0)) if if_attack else [float('nan')] * batch_size
#         ssim_values = calc_ssim(samples.unbind(0), original_samples.unbind(0)) if if_attack else [float('nan')] * batch_size
        
#         with torch.cuda.amp.autocast():
#             loss, outputs = train_class_batch(model, samples, targets, bm_pos, criterion, train_type)
#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
        
#         if loss_scaler is None:
#             loss /= update_freq
#             model.backward(loss)
#             model.step()
#         else:
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss /= update_freq
#             grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                     parameters=model.parameters(), create_graph=is_second_order,
#                                     update_grad=(data_iter_step + 1) % update_freq == 0)
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.zero_grad()

#         torch.cuda.synchronize()
#         min_lr, max_lr = 10., 0.
#         for group in optimizer.param_groups:
#             min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

#         acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
#         loss_meter.update(loss_value, 1)
#         if if_attack:
#             psnr_meter.update(sum(psnr_values) / len(psnr_values), n=batch_size)
#             ssim_meter.update(sum(ssim_values) / len(ssim_values), n=batch_size)
        
#         if data_iter_step % print_freq == 0:
#             print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e] [psnr: %.2f] [ssim: %.4f] [Channel: %s] [SNR: %.1f dB]' 
#                   % (epoch, batch_size * data_iter_step, len(data_loader.dataset),
#                      loss_meter.avg, acc_meter.avg * 100, max_lr, psnr_meter.avg, ssim_meter.avg,
#                      channel_type if if_attack else 'none', snr_db if if_attack else float('nan')))

#     train_stat = {
#         'loss': loss_meter.avg,
#         'acc': acc_meter.avg,
#         'psnr': psnr_meter.avg if if_attack else float('nan'),
#         'ssim': ssim_meter.avg if if_attack else float('nan'),
#         'channel': channel_type if if_attack and hasattr(data_loader.dataset, 'args') else 'none',
#         'snr': snr_db if if_attack and hasattr(data_loader.dataset, 'args') else float('nan')
#     }
#     return train_stat

def train_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, train_type, if_attack, max_norm: float=0,
                start_steps=None, lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=50):
    model.train(True)
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    channel = Channels()
    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        step = data_iter_step // update_freq
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        samples, bm_pos = samples
        targets = targets.to(device, non_blocking=True)
        original_samples = samples.clone().to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if if_attack:
            snr_db = 30  # 可调整为 -9 到 18 dB
            noise_var = torch.FloatTensor([1]).to(device) * 10**(-snr_db/20)
            samples = channel.Rayleigh(samples, noise_var.item())
        
        batch_size = samples.size(0)
        psnr_values = calc_psnr(samples.unbind(0), original_samples.unbind(0)) if if_attack else [float('nan')] * batch_size
        ssim_values = calc_ssim(samples.unbind(0), original_samples.unbind(0)) if if_attack else [float('nan')] * batch_size
        
        with torch.amp.autocast('cuda'):
            loss, outputs = train_class_batch(model, samples, targets, bm_pos, criterion, train_type)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

        acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
        loss_meter.update(loss_value, 1)
        if if_attack:
            psnr_meter.update(sum(psnr_values) / len(psnr_values), n=batch_size)
            ssim_meter.update(sum(ssim_values) / len(ssim_values), n=batch_size)
        
        if data_iter_step % print_freq == 0:
            print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e] [psnr: %.2f] [ssim: %.4f] [SNR: %.1f dB]' 
                  % (epoch, batch_size * data_iter_step, len(data_loader.dataset),
                     loss_meter.avg, acc_meter.avg * 100, max_lr, psnr_meter.avg, ssim_meter.avg, snr_db))

    train_stat = {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
        'psnr': psnr_meter.avg if if_attack else float('nan'),
        'ssim': ssim_meter.avg if if_attack else float('nan'),
        'snr': snr_db if if_attack else float('nan')
    }
    return train_stat


def train_epoch_wp(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, train_type, if_attack, wp_adver, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=50):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    attack = FGSM_REG(model, 4./255., 2./255., min_val=0, max_val=1, max_iters=4)

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    
    
    for data_iter_step, (samples ,targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples, bm_pos = samples
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if if_attack:
            bum_pos = torch.zeros_like(bm_pos)
            bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)     
            per_data = attack.perturb(samples, targets, bum_pos, 'mean', random_start=True, beta=beta)
            samples = per_data
        
        if epoch >= 5:
            awp = wp_adver.calc_awp(inputs_adv=samples,
                                            targets=targets)
            wp_adver.perturb(awp)
                                                                      
        batch_size = samples.size(0)
                                                   
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch(
                model, samples, targets, bm_pos, criterion, train_type)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if epoch >= 10:
            wp_adver.restore(awp)
                
        acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
        loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                    loss_meter.avg, acc_meter.avg*100, max_lr))
            
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 

