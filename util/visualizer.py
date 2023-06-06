import os
import time
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import label_binarize
from util import util
from util import metrics
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    This class print/save logging information
    """

    def __init__(self, param):
        """
        Initialize the Visualizer class
        """
        self.param = param
        self.output_path = os.path.join(param.checkpoints_dir, param.exp_type, param.exp_name)
        path = os.path.join(param.exp_dir, param.exp_type)
        self.log_path = os.path.join(path, param.exp_name)
        util.mkdir(self.log_path)

        # Create a logging file to store training losses
        self.train_log_filename = os.path.join(self.output_path, 'train_log.txt')
        with open(self.train_log_filename, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('----------------------- Training Log ({:s}) -----------------------\n'.format(now))

        self.train_summary_filename = os.path.join(self.output_path, 'train_summary.txt')
        with open(self.train_summary_filename, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('----------------------- Training Summary ({:s}) -----------------------\n'.format(now))

        self.test_log_filename = os.path.join(self.output_path, 'test_log.txt')
        with open(self.test_log_filename, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('----------------------- Testing Log ({:s}) -----------------------\n'.format(now))

        self.test_summary_filename = os.path.join(self.output_path, 'test_summary.txt')
        with open(self.test_summary_filename, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('----------------------- Testing Summary ({:s}) -----------------------\n'.format(now))

        self.writer = SummaryWriter(log_dir=self.log_path)

    def get_experiment_name(self, param):
        name = f"{param.omics_mode}_{param.net_VAE}_bs_{param.batch_size}_dr_{param.dropout_p}_lr_{param.lr}"
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
        
    def print_train_log(self, epoch, iteration, losses_dict, metrics_dict, load_time, comp_time, batch_size, dataset_size, with_time=True):
        """
        print train log on console and save the message to the disk

        Parameters:
            epoch (int)                     -- current epoch
            iteration (int)                 -- current training iteration during this epoch
            losses_dict (OrderedDict)       -- training losses stored in the ordered dict
            metrics_dict (OrderedDict)      -- metrics stored in the ordered dict
            load_time (float)               -- data loading time per data point (normalized by batch_size)
            comp_time (float)               -- computational time per data point (normalized by batch_size)
            batch_size (int)                -- batch size of training
            dataset_size (int)              -- size of the training dataset
            with_time (bool)                -- print the running time or not
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        if with_time:
            message = '[TRAIN] [Epoch: {:3d}   Iter: {:4d}   Load_t: {:.3f}   Comp_t: {:.3f}]   '.format(epoch, data_point_covered, load_time, comp_time)
        else:
            message = '[TRAIN] [Epoch: {:3d}   Iter: {:4d}]\n'.format(epoch, data_point_covered)
        for name, loss in losses_dict.items():
            message += '{:s}: {:.3f}   '.format(name, loss[-1])
        for name, metric in metrics_dict.items():
            message += '{:s}: {:.3f}   '.format(name, metric)

        print(message)  # print the message

        with open(self.train_log_filename, 'a') as log_file:
            log_file.write(message + '\n')  # save the message

    def print_train_summary(self, epoch, losses_dict, output_dict, train_time, current_lr):
        """
        print the summary of this training epoch

        Parameters:
            epoch (int)                             -- epoch number of this training model
            losses_dict (OrderedDict)               -- the losses dictionary
            output_dict (OrderedDict)               -- the downstream output dictionary
            train_time (float)                      -- time used for training this epoch
            current_lr (float)                      -- the learning rate of this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TRAIN] [Epoch: {:3d}]  '.format(int(epoch))

        for name, loss in losses_dict.items():
            write_message += '{:.6f}\t'.format(np.mean(loss))
            print_message += name + ': {:.3f}  '.format(np.mean(loss))
            self.writer.add_scalar('train_loss_'+name, np.mean(loss), epoch)

        metrics_dict = self.get_epoch_metrics(output_dict)
        for name, metric in metrics_dict.items():
            write_message += '{:.6f}\t'.format(metric)
            print_message += name + ': {:.3f}  '.format(metric)
            self.writer.add_scalar('train_'+name, metric, epoch)

        current_lr_msg = 'lr: {:.7f}'.format(current_lr)
        print_message += current_lr_msg
        self.writer.add_scalar('lr', current_lr, epoch)

        with open(self.train_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        print(print_message)

    def print_test_log(self, epoch, iteration, losses_dict, metrics_dict, batch_size, dataset_size):
        """
        print performance metrics of this iteration on console and save the message to the disk

        Parameters:
            epoch (int)                     -- epoch number of this testing model
            iteration (int)                 -- current testing iteration during this epoch
            losses_dict (OrderedDict)       -- training losses stored in the ordered dict
            metrics_dict (OrderedDict)      -- metrics stored in the ordered dict
            batch_size (int)                -- batch size of testing
            dataset_size (int)              -- size of the testing dataset
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        message = '[TEST] [Epoch: {:3d}   Iter: {:4d}]   '.format(int(epoch), data_point_covered)
        for name, loss in losses_dict.items():
            message += '{:s}: {:.3f}   '.format(name, loss[-1])
        for name, metric in metrics_dict.items():
            message += '{:s}: {:.3f}   '.format(name, metric)

        print(message)

        with open(self.test_log_filename, 'a') as log_file:
            log_file.write(message + '\n')

    def print_test_summary(self, epoch, losses_dict, output_dict, test_time):
        """
        print the summary of this testing epoch

        Parameters:
            epoch (int)                             -- epoch number of this testing model
            losses_dict (OrderedDict)               -- the losses dictionary
            output_dict (OrderedDict)               -- the downstream output dictionary
            test_time (float)                       -- time used for testing this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TEST]  [Epoch: {:3d}]  '.format(int(epoch))

        for name, loss in losses_dict.items():
            # write_message += '{:.6f}\t'.format(np.mean(loss))
            print_message += name + ': {:.3f}  '.format(np.mean(loss))
            self.writer.add_scalar('test_loss_'+name, np.mean(loss), epoch)

        metrics_dict = self.get_epoch_metrics(output_dict)

        for name, metric in metrics_dict.items():
            write_message += '{:.6f}\t'.format(metric)
            print_message += name + ': {:.3f}  '.format(metric)
            self.writer.add_scalar('test_' + name, metric, epoch)

        with open(self.test_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        print(print_message)

    def get_epoch_metrics(self, output_dict):
        """
        Get the downstream task metrics for whole epoch

        Parameters:
            output_dict (OrderedDict)  -- the output dictionary used to compute the downstream task metrics
        """
        if self.param.downstream_task == 'classification':
            y_true = output_dict['y_true'].cpu().numpy()
            y_true_binary = label_binarize(y_true, classes=range(self.param.class_num))
            y_pred = output_dict['y_pred'].cpu().numpy()
            y_prob = output_dict['y_prob'].cpu().numpy()
            if self.param.class_num == 2:
                y_prob = y_prob[:, 1]

            accuracy = sk.metrics.accuracy_score(y_true, y_pred)
            precision = sk.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = sk.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = sk.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
            try:
                auc = sk.metrics.roc_auc_score(y_true_binary, y_prob, multi_class='ovo', average='macro')
            except ValueError:
                auc = -1
                print('ValueError: ROC AUC score is not defined in this case.')

            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

    def save_output_dict(self, output_dict):
        """
        Save the downstream task output to disk

        Parameters:
            output_dict (OrderedDict)  -- the downstream task output dictionary to be saved
        """
        down_path = os.path.join(self.output_path, 'down_output')
        util.mkdir(down_path)
        if self.param.downstream_task == 'classification':
            # Prepare files
            index = output_dict['index'].numpy()
            y_true = output_dict['y_true'].cpu().numpy()
            y_pred = output_dict['y_pred'].cpu().numpy()
            y_prob = output_dict['y_prob'].cpu().numpy()

            sample_list = self.param.sample_list[index]

            # Output files
            y_df = pd.DataFrame({'sample': sample_list, 'y_true': y_true, 'y_pred': y_pred}, index=index)
            y_df_path = os.path.join(down_path, 'y_df.tsv')
            y_df.to_csv(y_df_path, sep='\t')

            prob_df = pd.DataFrame(y_prob, columns=range(self.param.class_num), index=sample_list)
            y_prob_path = os.path.join(down_path, 'y_prob.tsv')
            prob_df.to_csv(y_prob_path, sep='\t')


    def save_latent_space(self, latent_dict, sample_list):
        """
            save the latent space matrix to disc

            Parameters:
                latent_dict (OrderedDict)          -- the latent space dictionary
                sample_list (ndarray)               -- the sample list for the latent matrix
        """
        reordered_sample_list = sample_list[latent_dict['index'].astype(int)]
        latent_df = pd.DataFrame(latent_dict['latent'], index=reordered_sample_list)
        output_path = os.path.join(self.param.checkpoints_dir, self.param.exp_name, 'latent_space.tsv')
        print('Saving the latent space matrix...')
        latent_df.to_csv(output_path, sep='\t')


    @staticmethod
    def print_phase(phase):
        """
        print the phase information

        Parameters:
            phase (int)             -- the phase of the training process
        """
        if phase == 'p1':
            print('PHASE 1: Unsupervised Phase')
        elif phase == 'p2':
            print('PHASE 2: Supervised Phase')
        elif phase == 'p3':
            print('PHASE 3: Supervised Phase')
