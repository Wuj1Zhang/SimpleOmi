import torch
import torch.nn as nn


def get_loss_func(loss_name):
    """
    Return the loss function.
    Parameters:
        loss_name (str)    -- the name of the loss function: BCE | MSE | L1 | CE
    """
    if loss_name == 'MSE':
        return nn.MSELoss(reduction='mean')
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction='mean')
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)


# def kl_loss(mean, log_var, reduction='mean'):
#     part_loss = 1 + log_var - mean.pow(2) - log_var.exp()
#     if reduction == 'mean':
#         loss = -0.5 * torch.mean(part_loss)
#     else:
#         loss = -0.5 * torch.sum(part_loss)
#     return loss
