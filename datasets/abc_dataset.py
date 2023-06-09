import os.path
from datasets import load_file
from datasets import get_survival_y_true
from datasets.basic_dataset import BasicDataset
# from util import preprocess
import numpy as np
import pandas as pd
import torch


class ABCDataset(BasicDataset):
    """
    A dataset class for multi-omics dataset.
    For gene expression data, file should be prepared as '/path/to/data/A.tsv'.
    For DNA methylation data, file should be prepared as '/path/to/data/B.tsv'.
    For miRNA expression data, file should be prepared as '/path/to/data/C.tsv'.
    For each omics file, each columns should be each sample and each row should be each molecular feature.
    """

    def __init__(self, param):
        """
        Initialize this dataset class.
        """
        BasicDataset.__init__(self, param)
        self.omics_dims = []

        # Load data for A
        A_df = load_file(param, 'A')
        # Get the sample list
        if param.use_sample_list:
            sample_list_path = os.path.join(param.data_root, 'sample_list.tsv')       # get the path of sample list
            self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        else:
            self.sample_list = A_df.columns
        # Get the feature list for A
        feature_list_A = A_df.index
        A_df = A_df.loc[feature_list_A, self.sample_list]
        self.A_dim = A_df.shape[0]
        self.sample_num = A_df.shape[1]
        A_array = A_df.values
        if self.param.add_channel:
            # Add one dimension for the channel
            A_array = A_array[np.newaxis, :, :]
        self.A_tensor_all = torch.Tensor(A_array)
        self.omics_dims.append(self.A_dim)

        # Load data for B
        B_df = load_file(param, 'B')
        # Get the feature list for B
        feature_list_B = B_df.index
        B_df = B_df.loc[feature_list_B, self.sample_list]
        if param.ch_separate:
            B_df_list, self.B_dim = preprocess.separate_B(B_df)
            self.B_tensor_all = []
            for i in range(0, 23):
                B_array = B_df_list[i].values
                if self.param.add_channel:
                    # Add one dimension for the channel
                    B_array = B_array[np.newaxis, :, :]
                B_tensor_part = torch.Tensor(B_array)
                self.B_tensor_all.append(B_tensor_part)
        else:
            self.B_dim = B_df.shape[0]
            B_array = B_df.values
            if self.param.add_channel:
                # Add one dimension for the channel
                B_array = B_array[np.newaxis, :, :]
            self.B_tensor_all = torch.Tensor(B_array)
        self.omics_dims.append(self.B_dim)

        # Load data for C
        C_df = load_file(param, 'C')
        # Get the feature list for C
        feature_list_C = C_df.index
        C_df = C_df.loc[feature_list_C, self.sample_list]
        self.C_dim = C_df.shape[0]
        C_array = C_df.values
        if self.param.add_channel:
            # Add one dimension for the channel
            C_array = C_array[np.newaxis, :, :]
        self.C_tensor_all = torch.Tensor(C_array)
        self.omics_dims.append(self.C_dim)

        self.class_num = 0
        if param.downstream_task == 'classification':
            # Load labels
            labels_path = os.path.join(param.data_root, 'labels.tsv')       # get the path of the label
            labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.labels_array = labels_df.iloc[:, -1].values
            # Get the class number
            self.class_num = len(labels_df.iloc[:, -1].unique())
        elif param.downstream_task == 'regression':
            # Load target values
            values_path = os.path.join(param.data_root, 'values.tsv')  # get the path of the target value
            values_df = pd.read_csv(values_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.values_array = values_df.iloc[:, -1].astype(float).values
            self.values_max = self.values_array.max()
            self.values_min = self.values_array.min()
        elif param.downstream_task == 'survival':
            # Load survival data
            survival_path = os.path.join(param.data_root, 'survival.tsv')   # get the path of the survival data
            survival_df = pd.read_csv(survival_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.survival_T_array = survival_df.iloc[:, -2].astype(float).values
            self.survival_E_array = survival_df.iloc[:, -1].values
            self.survival_T_max = self.survival_T_array.max()
            self.survival_T_min = self.survival_T_array.min()
            if param.survival_loss == 'MTLR':
                self.y_true_tensor = get_survival_y_true(param, self.survival_T_array, self.survival_E_array)
            if param.stratify_label:
                labels_path = os.path.join(param.data_root, 'labels.tsv')  # get the path of the label
                labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
                self.labels_array = labels_df.iloc[:, -1].values
        elif param.downstream_task == 'multitask':
            # Load labels
            labels_path = os.path.join(param.data_root, 'labels.tsv')  # get the path of the label
            labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.labels_array = labels_df.iloc[:, -1].values
            # Get the class number
            self.class_num = len(labels_df.iloc[:, -1].unique())

            # Load target values
            values_path = os.path.join(param.data_root, 'values.tsv')  # get the path of the target value
            values_df = pd.read_csv(values_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.values_array = values_df.iloc[:, -1].astype(float).values
            self.values_max = self.values_array.max()
            self.values_min = self.values_array.min()

            # Load survival data
            survival_path = os.path.join(param.data_root, 'survival.tsv')  # get the path of the survival data
            survival_df = pd.read_csv(survival_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.survival_T_array = survival_df.iloc[:, -2].astype(float).values
            self.survival_E_array = survival_df.iloc[:, -1].values
            self.survival_T_max = self.survival_T_array.max()
            self.survival_T_min = self.survival_T_array.min()
            if param.survival_loss == 'MTLR':
                self.y_true_tensor = get_survival_y_true(param, self.survival_T_array, self.survival_E_array)
        elif param.downstream_task == 'alltask':
            # Load labels
            self.labels_array = []
            self.class_num = []
            for i in range(param.task_num-2):
                labels_path = os.path.join(param.data_root, 'labels_'+str(i+1)+'.tsv')  # get the path of the label
                labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
                self.labels_array.append(labels_df.iloc[:, -1].values)
                # Get the class number
                self.class_num.append(len(labels_df.iloc[:, -1].unique()))

            # Load target values
            values_path = os.path.join(param.data_root, 'values.tsv')  # get the path of the target value
            values_df = pd.read_csv(values_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.values_array = values_df.iloc[:, -1].astype(float).values
            self.values_max = self.values_array.max()
            self.values_min = self.values_array.min()

            # Load survival data
            survival_path = os.path.join(param.data_root, 'survival.tsv')  # get the path of the survival data
            survival_df = pd.read_csv(survival_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.survival_T_array = survival_df.iloc[:, -2].astype(float).values
            self.survival_E_array = survival_df.iloc[:, -1].values
            self.survival_T_max = self.survival_T_array.max()
            self.survival_T_min = self.survival_T_array.min()
            if param.survival_loss == 'MTLR':
                self.y_true_tensor = get_survival_y_true(param, self.survival_T_array, self.survival_E_array)

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Returns a dictionary that contains A_tensor, B_tensor, C_tensor, label and index
            input_omics (list)              -- a list of input omics tensor
            label (int)                     -- label of the sample
            index (int)                     -- the index of this data point
        """
        # Get the tensor of A
        if self.param.add_channel:
            A_tensor = self.A_tensor_all[:, :, index]
        else:
            A_tensor = self.A_tensor_all[:, index]

        # Get the tensor of B
        if self.param.ch_separate:
            B_tensor = []
            for i in range(0, 23):
                if self.param.add_channel:
                    B_tensor_part = self.B_tensor_all[i][:, :, index]
                else:
                    B_tensor_part = self.B_tensor_all[i][:, index]
                B_tensor.append(B_tensor_part)
            # Return a list of tensor
        else:
            if self.param.add_channel:
                B_tensor = self.B_tensor_all[:, :, index]
            else:
                B_tensor = self.B_tensor_all[:, index]
            # Return a tensor

        # Get the tensor of C
        if self.param.add_channel:
            C_tensor = self.C_tensor_all[:, :, index]
        else:
            C_tensor = self.C_tensor_all[:, index]

        if self.param.downstream_task == 'classification':
            # Get label
            label = self.labels_array[index]
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'label': label, 'index': index}
        elif self.param.downstream_task == 'regression':
            # Get target value
            value = self.values_array[index]
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'value': value, 'index': index}
        elif self.param.downstream_task == 'survival':
            # Get survival T and E
            survival_T = self.survival_T_array[index]
            survival_E = self.survival_E_array[index]
            y_true = self.y_true_tensor[index, :]
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'survival_T': survival_T, 'survival_E': survival_E, 'y_true': y_true, 'index': index}
        elif self.param.downstream_task == 'multitask':
            # Get label
            label = self.labels_array[index]
            # Get target value
            value = self.values_array[index]
            # Get survival T and E
            survival_T = self.survival_T_array[index]
            survival_E = self.survival_E_array[index]
            y_true = self.y_true_tensor[index, :]
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'label': label, 'value': value,
                    'survival_T': survival_T, 'survival_E': survival_E, 'y_true': y_true, 'index': index}
        elif self.param.downstream_task == 'alltask':
            # Get label
            label = []
            for i in range(self.param.task_num - 2):
                label.append(self.labels_array[i][index])
            # Get target value
            value = self.values_array[index]
            # Get survival T and E
            survival_T = self.survival_T_array[index]
            survival_E = self.survival_E_array[index]
            y_true = self.y_true_tensor[index, :]
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'label': label, 'value': value,
                    'survival_T': survival_T, 'survival_E': survival_E, 'y_true': y_true, 'index': index}
        else:
            return {'input_omics': [A_tensor, B_tensor, C_tensor], 'index': index}

    def __len__(self):
        """
        Return the number of data points in the dataset.
        """
        return self.sample_num

