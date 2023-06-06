"""
Training and testing for OmiEmbed
"""
import time
import warnings
from util import util
from params.train_test_params import TrainTestParams
from datasets import create_separate_dataloader
from models import vae_classifier_model
from util.visualizer import Visualizer


if __name__ == "__main__":
    '''python train_test.py --gpu_ids 7 --experiment_name AbC_fc --omics_mode abc --num_threads 8 '''
    '''python train_test.py --gpu_ids 7 --experiment_name temp --omics_mode c --num_threads 12 '''
    warnings.filterwarnings('ignore')
    full_start_time = time.time()
    # Get parameters
    param = TrainTestParams().parse()

    # Dataset related
    full_dataloader, train_dataloader, val_dataloader, test_dataloader = create_separate_dataloader(param)
    print('The size of training set is {}'.format(len(train_dataloader)))
    param.sample_list = full_dataloader.get_sample_list()
    param.omics_dims = full_dataloader.get_omics_dims()
    param.class_num = full_dataloader.get_class_num()
    print('The number of classes: {}'.format(param.class_num))

    # Create a model given param.model and other parameters
    model = vae_classifier_model.VaeClassifierModel(param)     
    # Regular setup for the model: load and print networks, create schedulers
    model.setup(param)


    # Create a visualizer to print results
    '''------------------------------NEED TO BE IMPROVE!!!!----------------------------'''
    visualizer = Visualizer(param)  # Create a visualizer to print results
    '''------------------------------NEED TO BE IMPROVE!!!!----------------------------'''



    # Start the epoch loop
    visualizer.print_phase(model.phase)
    for epoch in range(param.epoch_count, param.epoch_num + 1):     # outer loop for different epochs
        epoch_start_time = time.time()                              # Start time of this epoch
        model.epoch = epoch

        ''' TRAINING STARTING HERE '''
        model.set_train()                                           # Set train mode for training
        iter_load_start_time = time.time()                          # Start time of data loading for this iteration
        output_dict, losses_dict, metrics_dict = model.init_log_dict()          # Initialize the log dictionaries
        if epoch == param.epoch_num_p1 + 1:
            model.phase = 'p2'                                      # Change to supervised phase
            visualizer.print_phase(model.phase)
        if epoch == param.epoch_num_p1 + param.epoch_num_p2 + 1:
            model.phase = 'p3'                                      # Change to supervised phase
            visualizer.print_phase(model.phase)

        # Start training loop
        for i, data in enumerate(train_dataloader):                 # Inner loop for different iteration within one epoch
            model.iter = i
            dataset_size = len(train_dataloader)
            actual_batch_size = len(data['index'])
            iter_start_time = time.time()
            model.set_input(data)                                   # Unpack input data from the output dictionary of the dataloader
            model.update()                                          # Calculate losses, gradients and update network parameters
            model.update_log_dict(output_dict, losses_dict, metrics_dict, actual_batch_size)       # Update the log dictionaries
            iter_load_start_time = time.time()

        train_time = time.time() - epoch_start_time
        current_lr = model.update_learning_rate()  # update learning rates at the end of each epoch
        visualizer.print_train_summary(epoch, losses_dict, output_dict, train_time, current_lr)

        ''' TESTING STARTING HERE !!!!!'''
        model.set_eval()                                            # Set eval mode for testing
        test_start_time = time.time()                               # Start time of testing
        output_dict, losses_dict, metrics_dict = model.init_log_dict()  # Initialize the log dictionaries

        # Start testing loop
        for i, data in enumerate(test_dataloader):
            dataset_size = len(test_dataloader)
            actual_batch_size = len(data['index'])
            model.set_input(data)                                   # Unpack input data from the output dictionary of the dataloader
            model.test()                                            # Run forward to get the output tensors
            model.update_log_dict(output_dict, losses_dict, metrics_dict, actual_batch_size)  # Update the log dictionaries

        test_time = time.time() - test_start_time
        visualizer.print_test_summary(epoch, losses_dict, output_dict, test_time)
        if epoch == param.epoch_num:
            visualizer.save_output_dict(output_dict)


    full_time = time.time() - full_start_time
    print('Full running time: {:.3f}s'.format(full_time))
