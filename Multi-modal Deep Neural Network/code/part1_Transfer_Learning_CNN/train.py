################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.optim as optim
import copy

from cifar100_utils import get_train_validation_set, get_test_set,get_gaussian_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """


    model_res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model_res.fc = nn.Linear(512, 100)

    ## last layer reinitialization
    torch.nn.init.normal_(model_res.fc.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(model_res.fc.bias, 0.0)

    #set gradients = false: 
    params = list(model_res.parameters())

    # skip the last layer : weight and bias 
    for i in range(0,len(params)-2):
        params[i].requires_grad = False


    return model_res


def train_model(model, lr, batch_size, epochs, data_dir, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    """

    
    train_dataset,val_dataset = get_train_validation_set(data_dir,augmentation_name=augmentation_name)

    dataloaders = {}

    dataloaders['train'] = data.DataLoader(train_dataset, batch_size=batch_size)
    dataloaders['validation'] = data.DataLoader(val_dataset, batch_size=batch_size)

    
    # parameters to update during training
    params_to_update=[]
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.Adam(params_to_update)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    
    val_acc_history = []
    best_acc = 0.0

    # Reference : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


    for epoch in range(epochs):
        print('Epoch {}'.format(epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

            
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # get the class with max prob.
                preds = outputs.argmax(dim=-1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                #print(running_corrects)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model


            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                print("best val Accuracy:",epoch_acc)
            if phase == 'validation':
                val_acc_history.append(epoch_acc)

    

    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

    
    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            output = model(data_inputs)
            # get the class with max prob.
            pred_labels = outputs.argmax(dim=-1)

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")
   



def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """

    set_seed(seed)
    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = get_model()
    # Send the model to GPU
    model.to(device)

    best_model = train_model(model,lr,batch_size,epochs,data_dir,device,augmentation_name)

    test_data = get_test_set(data_dir)
    test_data_with_noise = get_gaussian_test_set(data_dir)

    dataloader_test = data.DataLoader(test_data, batch_size=batch_size,return_numpy=False)
    dataloader_test_noise = data.DataLoader(test_data_with_noise, batch_size=batch_size,return_numpy=False)

    print("Testing our best model -----")
    print("without noise ---")
    evaluate_model(model, dataloader_test, device)
    print("--------------------")
    print("with noise -----")
    evaluate_model(model, dataloader_test_noise, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    print(args)
    kwargs = vars(args)
    main(**kwargs)
