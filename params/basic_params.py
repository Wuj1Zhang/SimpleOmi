import time
import argparse
import torch
import os
import models
from util import util


class BasicParams:
    """
    This class define the console parameters
    """

    def __init__(self):
        """
        Reset the class. Indicates the class hasn't been initialized
        """
        self.initialized = False
        self.isTrain = True
        self.isTest = True

    def initialize(self, parser):
        """
        Define the common console parameters
        """
        parser.add_argument('--gpu_ids', type=str, default='1',
                            help='which GPU would like to use: e.g. 0 or 0,1, -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='/gscratch/stf/hzhang33/expOmi',
                            help='models, settings and intermediate results are saved in folder in this directory')
        parser.add_argument('--exp_name', type=str, default=None,
                            help='name of the folder in the checkpoint directory')
        parser.add_argument('--exp_dir', type=str, default='/gscratch/stf/hzhang33/omiExp',
                            help='name of the folder in the checkpoint directory')
        parser.add_argument('--exp_type', type=str, default='try',
                            help='the type of experiment')

        # Dataset parameters
        parser.add_argument('--omics_mode', type=str, default='abc',
                            help='omics types would like to use in the model, options: [abc | a | b | c]')
        parser.add_argument('--data_root', type=str, default='/gscratch/stf/hzhang33/data',
                            help='path to input data')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input data batch size')
        parser.add_argument('--num_threads', default=12, type=int,
                            help='number of threads for loading data')
        parser.add_argument('--use_sample_list',  default=True, action='store_true',
                            help='provide a subset sample list of the dataset, store in the path data_root/sample_list.tsv, if False use all the samples')
        parser.add_argument('--file_format', type=str, default='tsv',
                            help='file format of the omics data, options: [tsv | csv]')

        # Model parameters
        parser.add_argument('--net_down', type=str, default='multi_FC_classifier',
                            help='specify the backbone of the downstream task network, default is the multi-layer FC classifier, options: [multi_FC_classifier | multi_FC_regression | multi_FC_survival | multi_FC_multitask]')
        parser.add_argument('--norm_type', type=str, default='batch',
                            help='the type of normalization applied to the model, default to use batch normalization, options: [batch]')
        parser.add_argument('--dropout_p', type=float, default=0.0,
                            help='probability of an element to be zeroed in a dropout layer, default is 0 which means no dropout.')
        parser.add_argument('--leaky_slope', type=float, default=0.2,
                            help='the negative slope of the Leaky ReLU activation function')
        parser.add_argument('--latent_space_dim', type=int, default=128,
                            help='the dimensionality of the latent space')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='choose the method of network initialization, options: [normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal initialization methods')

        # Loss parameters
        parser.add_argument('--recon_loss', type=str, default='MSE',
                            help='chooses the reconstruction loss function, options: [MSE | L1]')
        parser.add_argument('--k_embed', type=float, default=1.0,
                            help='weight for the embedding loss')
        parser.add_argument('--k_reconA', type=float, default=1.0,
                            help='weight for the reconstruction loss of omics-A')       
        parser.add_argument('--k_reconB', type=float, default=1.0,
                            help='weight for the reconstruction loss of omics-A')   
        parser.add_argument('--k_reconC', type=float, default=1.0,
                            help='weight for the reconstruction loss of omics-A')                
        parser.add_argument('--w_down', type=float, default=1.0,
                            help='weight of downstream loss')

        parser.add_argument('--detail', action='store_true', default=False,
                            help='print more detailed information if set true')
        parser.add_argument('--experiment_to_load', type=str, default='test',
                            help='the experiment to load')

        self.initialized = True  # set the initialized to True after we define the parameters of the project
        return parser

    def get_params(self):
        """
        Initialize our parser with basic parameters once.
        Add additional model-specific parameters.
        """
        if not self.initialized:  # check if this object has been initialized
            # if not create a new parser object
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            #  use our method to initialize the parser with the predefined arguments
            parser = self.initialize(parser)

        # get the basic parameters
        param, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_params(self, param):
        """
        Print welcome words and command line parameters.
        Save the command line parameters in a txt file to the disk
        """
        message = ''
        message += '\nSimlified Omiembed framework\n'
        message += '----------------------Parameters-----------------------\n'
        for key, value in sorted(vars(param).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>18}: {:<15}{}\n'.format(str(key), str(value), comment)
        message += '----------------------------------------------------------------\n'
        print(message)

        # Save the running parameters setting in the disk
        param.exp_name = param.exp_name if param.exp_name else self.get_experiment_name(param)
        experiment_dir = os.path.join(param.checkpoints_dir, param.exp_type, param.exp_name)

        util.mkdir(experiment_dir)
        file_name = os.path.join(experiment_dir, 'cmd_parameters.txt')
        with open(file_name, 'w') as param_file:
            now = time.strftime('%c')
            param_file.write('{:s}\n'.format(now))
            param_file.write(message)
            param_file.write('\n')

    def get_experiment_name(self, param):
        name = f"{param.omics_mode}_bs_{param.batch_size}_dr_{param.dropout_p}_lr_{param.lr}"
        name += f'_p1_{param.epoch_num_p1}_p2_{param.epoch_num_p2}_p3_{param.epoch_num_p3}'
        if param.lr_policy is not None:
            name += f'_LRsched_{param.lr_policy}'
        if param.init_type is not None:
            name += f'_init_{param.init_type}'
        if param.k_embed != 0.01:
            name += f'_woe_{param.k_embed}'
        if param.w_down != 0.01:
            name += f'_wod_{param.w_down}'
        if param.k_reconA != 0.01:
            name += f'_woa_{param.k_reconA}'
        if param.k_reconB != 0.01:
            name += f'_wob_{param.k_reconB}'
        if param.k_reconC != 0.01:
            name += f'_woc_{param.k_reconC}'
        return name

    def parse(self):
        """
        Parse the parameters of our project. Set up GPU device. Print the welcome words and list parameters in the console.
        """
        param = self.get_params()  # get the parameters to the object param
        param.isTrain = self.isTrain
        param.isTest = self.isTest

        # Print welcome words and command line parameters
        self.print_params(param)

        # Set the internal parameters
        # epoch_num: the total epoch number
        if self.isTrain:
            param.epoch_num = param.epoch_num_p1 + param.epoch_num_p2 + param.epoch_num_p3
        # downstream_task: for the classification task a labels.tsv file is needed
        param.downstream_task = 'classification'
        # add_channel: add one extra dimension of channel for the input data, used for convolution layer
        # ch_separate: separate the DNA methylation matrix base on the chromosome
        param.add_channel = False
        param.ch_separate = False
        # omics_num: the number of omics types
        param.omics_num = len(param.omics_mode)

        # Set up GPU
        str_gpu_ids = param.gpu_ids.split(',')
        param.gpu_ids = []
        for str_gpu_id in str_gpu_ids:
            int_gpu_id = int(str_gpu_id)
            if int_gpu_id >= 0:
                param.gpu_ids.append(int_gpu_id)
        if len(param.gpu_ids) > 0:
            torch.cuda.set_device(param.gpu_ids[0])

        self.param = param
        return self.param
