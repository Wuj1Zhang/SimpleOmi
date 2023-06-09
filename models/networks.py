import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, activation=True, normalization=True, activation_name='LeakyReLU'):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            # FC block doesn't support InstanceNorm1d
            if isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d:
                norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y

# FcVae
class FcVaeABC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=384, dim_2B=256,
                 dim_1A=384, dim_2A=256, dim_1C=384, dim_2C=256, dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeABC, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.C_dim = omics_dims[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_latent = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_latent = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent = self.encode_fc_latent(level_4)

        return latent

    def decode(self, latent):
        level_1 = self.decode_fc_latent(latent)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean


    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        return latent, recon_x

class FcVaeB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for DNA methylation dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=512, dim_2B=256,
                 dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeB, self).__init__()

        self.B_dim = omics_dims[1]

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_latent = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # DECODER
        # Layer 1
        self.decode_fc_latent = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])

        level_3 = self.encode_fc_2B(level_2_B)

        level_4 = self.encode_fc_3(level_3)

        latent = self.encode_fc_latent(level_4)

        return latent

    def decode(self, latent):
        level_1 = self.decode_fc_latent(latent)

        level_2 = self.decode_fc_2(level_1)

        level_3_B = self.decode_fc_3B(level_2)

        recon_B = self.decode_fc_4B(level_3_B)

        return [None, recon_B]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        return latent, recon_x


class FcVaeA(nn.Module):
    """
        Defines a fully-connected variational autoencoder for gene expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1A=1024, dim_2A=1024,
                 dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeA, self).__init__()

        self.A_dim = omics_dims[0]

        # ENCODER
        # Layer 1
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_latent = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # DECODER
        # Layer 1
        self.decode_fc_latent = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_A = self.encode_fc_1A(x[0])

        level_3_A = self.encode_fc_2A(level_2_A)

        level_4 = self.encode_fc_3(level_3_A)

        latent = self.encode_fc_latent(level_4)

        return latent


    def decode(self, latent):
        level_1 = self.decode_fc_latent(latent)

        level_2 = self.decode_fc_2(level_1)

        level_3_A = self.decode_fc_3A(level_2)

        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        return latent, recon_x


class FcVaeC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1C=1024, dim_2C=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeC, self).__init__()

        self.C_dim = omics_dims[2]
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_latent = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # DECODER
        # Layer 1
        self.decode_fc_latent = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_C = self.encode_fc_1C(x[2])

        level_3_C = self.encode_fc_2C(level_2_C)

        level_4 = self.encode_fc_3(level_3_C)

        latent = self.encode_fc_latent(level_4)

        return latent


    def decode(self, latent):
        level_1 = self.decode_fc_latent(latent)

        level_2 = self.decode_fc_2(level_1)

        level_3_C = self.decode_fc_3C(level_2)

        recon_C = self.decode_fc_4C(level_3_C)

        return [None, None, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        return latent, recon_x



# Class for downstream task
class MultiFcClassifier(nn.Module):
    """
    Defines a multi-layer fully-connected classifier
    """
    def __init__(self, class_num=2, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 class_dim_1=128, class_dim_2=64, layer_num=3):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            class_num (int)         -- the number of class
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcClassifier, self).__init__()

        self.input_fc = FCBlock(latent_dim, class_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = class_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, class_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = class_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(class_dim_2, class_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y

# Class for the OmiEmbed combined network
class OmiEmbed(nn.Module):
    """
    Defines the OmiEmbed combined network
    """
    def __init__(self, net_down, omics_dims, omics_mode='multi_omics', norm_layer=nn.InstanceNorm1d, kernel_size=9,
                 leaky_slope=0.2, dropout_p=0, latent_dim=64, class_num=2, time_num=256, task_num=7):
        """
            Construct the OmiEmbed combined network
            Parameters:
                net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
                omics_dims (list)       -- the list of input omics dimensions
                omics_mode (str)        -- omics types would like to use in the model
                norm_layer              -- normalization layer
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
                class_num (int/list)    -- the number of classes
                time_num (int)          -- the number of time intervals
                task_num (int)          -- the number of downstream tasks
        """
        super(OmiEmbed, self).__init__()

        self.vae = None
        if omics_mode == 'abc':
            self.vae = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'b':
            self.vae = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'a':
            self.vae = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'c':
            self.vae = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

        self.net_down = net_down
        self.down = None
        if net_down == 'multi_FC_classifier':
            self.down = MultiFcClassifier(class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        else:
            raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    def get_last_encode_layer(self):
        return self.vae.get_last_encode_layer()

    def forward(self, x):
        z, recon_x, mean, log_var = self.vae(x)
        y_out = self.down(mean)
        return z, recon_x, mean, log_var, y_out


def get_norm_layer(norm_type='batch'):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization method [%s] is not found' % norm_type)
    return norm_layer


def define_net(net_down, omics_dims, omics_mode='multi_omics', norm_type='batch',
               leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256, task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the OmiEmbed network

    Parameters:
        net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch 
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        class_num (int)         -- the number of classes
        time_num (int)          -- the number of time intervals
        task_num (int)          -- the number of downstream tasks
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns the OmiEmbed network

    The network has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    net = OmiEmbed(net_down, omics_dims, omics_mode, norm_layer, leaky_slope, dropout_p,
                   latent_dim, class_num, time_num, task_num)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_VAE(omics_dims, omics_mode='multi_omics', norm_type='batch', leaky_slope=0.2, dropout_p=0,
               latent_dim=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the VAE network

    Parameters:
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns a VAE

    The default backbone of the VAE is one dimensional convolutional layer.

    The generator has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)
    
    if omics_mode == 'abc':
        net = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif omics_mode == 'b':
        net = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif omics_mode == 'a':
        net = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif omics_mode == 'c':
        net = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_down(net_down, norm_type='batch', leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256,
                task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
        Create the downstream task network

        Parameters:
            net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
            norm_type (str)         -- the name of normalization layers used in the network, default: batch
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            class_num (int)         -- the number of class
            time_num (int)          -- the number of time intervals
            task_num (int)          -- the number of downstream tasks
            init_type (str)         -- the name of our initialization method
            init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
            gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

        Returns a downstream task network

        The default downstream task network is a multi-layer fully-connected classifier.

        The generator has been initialized by <init_net>.
        """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if net_down == 'multi_FC_classifier':
        net = MultiFcClassifier(class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    else:
        raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    return init_net(net, init_type, init_gain, gpu_ids)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # multi-GPUs
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    """
    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, param):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer (opt class)     -- the optimizer of the network
        param (params class)      -- param.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <param.niter> epochs and linearly decay the rate to zero
    over the next <param.niter_decay> epochs.

    """
    if param.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_lambda = 1.0 - max(0, epoch + param.epoch_count - param.epoch_num + param.epoch_num_decay) / float(param.epoch_num_decay + 1)
            return lr_lambda
        # lr_scheduler is imported from torch.optim
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif param.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=param.decay_step_size, gamma=0.1)
    elif param.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.epoch_num, eta_min=0)
    else:
        return NotImplementedError('Learning rate policy [%s] is not found', param.lr_policy)
    return scheduler
