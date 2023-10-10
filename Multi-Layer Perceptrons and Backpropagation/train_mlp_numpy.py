################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
from modules import LinearModule
import cifar10_utils

import torch
import os.path
from os import path
import csv
import pandas as pd
import json

def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    # argmax will return index from [0,C-1]
    prediction_class = np.argmax(predictions,axis = 1)
    conf_mat = np.zeros((predictions.shape[1],predictions.shape[1]))
    
    for i in range(len(targets)):
        conf_mat[prediction_class[i]][targets[i]] += 1 
    
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """

    metrics = dict()

    metrics['accuracy'] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    metrics['precision'] = np.diag(confusion_matrix)/np.sum(confusion_matrix,axis = 1)
    metrics['recall'] = np.diag(confusion_matrix)/np.sum(confusion_matrix,axis = 0)
    metrics['f1_beta'] = ((1+beta**2)*metrics['precision']*metrics['recall'])/((beta**2*metrics['precision']) + metrics['recall'])
 
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """
    input_features = 32*32*3
    conf_matrix = np.zeros((num_classes,num_classes))
    for data_inputs, data_labels in data_loader:
        
        # get predictions
        input_data = data_inputs.flatten().reshape(data_inputs.shape[0], input_features)
        preds= model.forward(input_data)
        
        # get confusion matrix
        # data labels not one hot encoded therefore conversion into 1-D array not required
        #data_labels_1D = np.argmax(data_labels,axis = 1)
        conf_matrix += confusion_matrix(preds, data_labels)

    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    # TODO: Initialize model and loss module
    
    input_features = 32*32*3
    num_classes = 10
    
    model = MLP(input_features, hidden_dims, num_classes)
    loss_module = CrossEntropyModule()
    
    
    # TODO: Training loop including validation
    
    best_validation_accuracy = 0.0
    val_accuracies = []
    loss_train_per_step = []
    # Training loop
    for epoch in range(epochs):
        loss_train = []
        #print("epoch : ",epoch) 
        for data_inputs, data_labels in cifar10_loader['train']: 
        
            
            input_data = data_inputs.flatten().reshape(data_inputs.shape[0], input_features)

            ## Step 1: Run the model on the input data
            preds = model.forward(input_data)
            # S*N pred matrix 
        
            ## Step 2: Calculate the loss
            loss = loss_module.forward(preds, data_labels)

            loss_train_per_step.append(loss)
            ## Step 3: Perform backpropagation
            dout = loss_module.backward(preds,data_labels)
            model.backward(dout)

            ## Step 4: Update the parameters
            for layer in model.layers:
                if(isinstance(layer, LinearModule)):
                    layer.params['weight'] = layer.params['weight'] - lr*layer.grads['weight']
                    layer.params['bias'] = layer.params['bias'] - lr*layer.grads['bias']
                    #print("---")
                    #print(layer.params['weight'].shape)
                    #print("---")
        ##############
        # Validation #
        ##############
        train_metrics = evaluate_model(model, cifar10_loader['train'], num_classes)
        val_metrics = evaluate_model(model, cifar10_loader['validation'], num_classes)
        

        val_accuracies.append(val_metrics['accuracy'])
        ## Reference : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
       
        #print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_metrics['accuracy']*100.0:05.2f}%, Validation accuracy: {val_metrics['accuracy']*100.0:05.2f}%")
        
        if(val_metrics['accuracy'] > best_validation_accuracy):
            best_validation_accuracy = val_metrics['accuracy']
            print("\t (New best performance, saving model...)")
            best_model = deepcopy(model)
            print("\t (deepcopying done)")
            best_val_epoch = epoch
            print("\t Best epoch : ", best_val_epoch)
            print("\t Best val accuracy :", best_validation_accuracy)

   
    print("Training finished \n")

    # TODO: Test best model

    print("Now test data ---")

    test_metrics = evaluate_model(best_model, cifar10_loader['test'], num_classes)
    test_accuracy = test_metrics['accuracy']
    print("Test Accuracy : ", test_accuracy)

    print("\n Testing finished \n")


    # TODO: Add any information you might want to save for plotting
    #logging_info = ...
    #######################
    # END OF YOUR CODE    #
    #######################
    logging_info = {"best_val_acc": max(val_accuracies),"train_loss":loss_train_per_step,'epochs': epochs ,'lr':lr,'hidden_layers':hidden_dims}

    return best_model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    #train(**kwargs)
    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    print("Final Test Accuracy : ", test_accuracy)


    results = {'learning_rate': logging_info['lr'],
          'hidden_layers': logging_info['hidden_layers'],
          'epochs': logging_info['epochs'],
          'optimizer': 'sgd',
          'test_accuracy' : test_accuracy,
          'best_val_accuracy' : logging_info['best_val_acc'],
          'val_accuracy': val_accuracies,
          'train_loss':logging_info['train_loss']
            }

    a = []

    # Serializing json
    json_object = json.dumps(results, indent=4)

    if not os.path.isfile("results_numpy.json"):
        a.append(json_object)
        with open("results_numpy.json", mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open("results_numpy.json") as feedsjson:
            feeds = json.load(feedsjson)
            feeds.append(json_object)
        with open("results_numpy.json", mode='w') as f:
            f.write(json.dumps(feeds, indent=2))
    
    # Feel free to add any additional functions, such as plotting of the loss curve here
    
