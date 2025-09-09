import torch, os
import torch.nn.functional as F
import numpy as np
"""Utility functions for training and validating GlobalMapper models.

This module wraps the PyTorch training loop into two functions.  The goal of
the comments added in this repository is to make the training procedure easy to
follow and to highlight where additional features (such as functional-zone
labels or building attributes) could influence the loss terms.
"""

from tensorboard_logger import configure, log_value
from time import gmtime, strftime
import warnings

# Suppress warnings from external libraries so the training log is cleaner.
warnings.filterwarnings("ignore")


def train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler):
    """One epoch of model training.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network being trained.
    epoch : int
        Current epoch index used purely for logging.
    train_loader : DataLoader
        Provides batches of training graphs.
    device : torch.device
        GPU or CPU device where tensors are allocated.
    opt : dict
        Training options loaded from the YAML configuration file.
    loss_dict : dict
        Mapping from loss name to instantiated loss functions.  Additional
        losses for new attributes can be inserted into this dictionary when the
        model is extended.
    optimizer : torch.optim.Optimizer
        Updates model parameters based on gradients.
    scheduler : torch.optim.lr_scheduler
        Learning-rate scheduler called every iteration.

    Returns
    -------
    Tuple[float, float]
        Mean existence accuracy and mean loss for the epoch.
    """

    model.train()  # switch the model to training mode
    loss_sum = 0   # track accumulated loss to compute average later
    ext_acc = 0    # accumulated existence accuracy
    iter_ct = 0    # number of batches processed
    batch_size = opt['batch_size']
    num_data = opt['num_data']

    for batch_idx, data in enumerate(train_loader):
        # Move batch to the target device and reset gradients.
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass returns predictions for node existence, position, size
        # and additional building descriptors.
        exist, pos, size, mu, log_var, b_shape, b_iou = model(data)

        # Ground-truth attributes prepared by ``UrbanGraphDataset``.  When new
        # per-block attributes are introduced (e.g. functional zone labels),
        # corresponding tensors should be extracted here as well.
        exist_gt = data.x[:, 0].unsqueeze(1)
        pos_gt = data.org_node_pos
        size_gt = data.org_node_size
        b_shape_gt = data.b_shape_gt
        b_iou_gt = data.b_iou

        # Convert logits into binary predictions for existence.
        exist_out = torch.ge(F.sigmoid(exist), 0.5).type(torch.uint8)
        # Penalize mismatch between the number of predicted and ground-truth
        # existing blocks.
        extsum_loss = loss_dict['ExtSumloss'](torch.sum(exist_out), torch.sum(exist_gt))
        exist_loss = loss_dict['ExistBCEloss'](exist, exist_gt)
        pos_loss = loss_dict['Posloss'](pos, pos_gt)
        size_loss = loss_dict['Sizeloss'](size, size_gt)
        shape_loss = loss_dict['ShapeCEloss'](b_shape, b_shape_gt)
        iou_loss = loss_dict['Iouloss'](b_iou, b_iou_gt)
        # Kullback–Leibler divergence term for the VAE latent space.
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # Combine all individual losses with weights from the configuration.
        # When extra attributes are added (e.g. functional zone classes) the
        # user can introduce a new loss term here and register its weight in
        # the YAML file.
        loss = opt['exist_weight'] * exist_loss + opt['pos_weight'] * pos_loss + opt['kld_weight'] * kld_loss +\
             size_loss * opt['size_weight'] + extsum_loss * opt['extsum_weight'] + \
              opt['shape_weight'] * shape_loss + opt['iou_weight'] * iou_loss

        # Backpropagate gradients.
        loss.backward()

        loss_sum += loss.item()     # accumulate loss for averaging
        optimizer.step()             # update network parameters
        scheduler.step()             # advance learning-rate scheduler

        if opt['save_record']:
            # When enabled, log individual loss components to TensorBoard for
            # monitoring training progress.
            log_value('train/all_loss', loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
            log_value('train/exist_loss', exist_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
            log_value('train/pos_loss', pos_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch  
            log_value('train/size_loss', size_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch         
            log_value('train/kld_loss', kld_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
            log_value('train/extsum_loss', extsum_loss, int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
            log_value('train/shape_loss', shape_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
            log_value('train/bldg_iou_loss', iou_loss.item(), int(epoch * (num_data / train_loader.batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch

        # Compute existence prediction accuracy for the batch.
        correct_ext = (exist_out == data.x[:,0].unsqueeze(1)).sum() /  torch.numel(data.x[:,0])
        ext_acc += correct_ext

        iter_ct += 1
    

    # Return mean accuracy and loss across the entire epoch.
    return ext_acc / np.float(iter_ct), loss_sum / np.float(iter_ct)




def validation(model, epoch, val_loader, device, opt, loss_dict, scheduler):
    """Evaluate the model on the validation set.

    The structure mirrors :func:`train` but gradients are disabled and only
    statistics are collected.
    """
    with torch.no_grad():
        model.eval()
        loss_all = 0
        ext_acc = 0
        iter_ct = 0
        batch_size = opt['batch_size']
        num_data = opt['num_data']
        loss_geo = 0.0

        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            exist, pos, size, mu, log_var, b_shape, b_iou = model(data)

            exist_gt = data.x[:, 0].unsqueeze(1)
            merge_gt = data.x[:, 1].unsqueeze(1)
            pos_gt = data.org_node_pos
            size_gt = data.org_node_size
            b_shape_gt = data.b_shape_gt
            b_iou_gt = data.b_iou

            exist_loss = loss_dict['ExistBCEloss'](exist, exist_gt) 
            pos_loss = loss_dict['Posloss'](pos, pos_gt)
            size_loss = loss_dict['Sizeloss'](size, size_gt)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            exist_out = torch.ge(exist, 0.5).type(torch.uint8)
            extsum_loss = loss_dict['ExtSumloss'](torch.sum(exist_out), torch.sum(exist_gt))
            shape_loss = loss_dict['ShapeCEloss'](b_shape, b_shape_gt)
            iou_loss = loss_dict['Iouloss'](b_iou, b_iou_gt)

            loss = opt['exist_weight'] * exist_loss + opt['pos_weight'] * pos_loss + opt['kld_weight'] * kld_loss +\
                size_loss * opt['size_weight'] + extsum_loss * opt['extsum_weight'] + \
                 opt['shape_weight'] * shape_loss + opt['iou_weight'] * iou_loss
            loss_all += loss.item()
            loss_geo += (pos_loss.item() + size_loss.item())

            if opt['save_record']:
                # Log validation metrics to TensorBoard for later inspection.
                log_value('val/val_all_loss', loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
                log_value('val/val_exist_loss', exist_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
                log_value('val/val_pos_loss', pos_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch  
                log_value('val/val_size_loss', size_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch         
                log_value('val/val_kld_loss', kld_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
                log_value('val/val_extsum_loss', extsum_loss, int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
                log_value('val/val_shape_loss', shape_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch
                log_value('val/val_bldg_iou_loss', iou_loss.item(), int(epoch * (num_data / batch_size) + batch_idx) ) # (num_data / batch_size) = how many batches per epoch

            correct_ext = (exist_out == data.x[:,0].unsqueeze(1)).sum() /  torch.numel(data.x[:,0])
            ext_acc += correct_ext

            iter_ct += 1

    return ext_acc / np.float(iter_ct), loss_all / np.float(iter_ct), loss_geo / np.float(iter_ct)