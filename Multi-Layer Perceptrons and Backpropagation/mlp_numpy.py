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
# Date Created: 2022-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

         # 2 because of one input layer and one output layer
        num_layers = len(n_hidden) + 2
        
        dimensions_MLP = [n_inputs] + n_hidden + [n_classes]
        
        self.layers = [] 
        is_input_layer = True
        print(dimensions_MLP)
        for i in range(num_layers-1):
            
            self.layers.append(LinearModule(dimensions_MLP[i],dimensions_MLP[i+1],input_layer=is_input_layer))
            
            # Here RELU/ELU activation would be applied
            if(i < num_layers - 2):
                self.layers.append(ELUModule())
                    
            
            # The layer before the output : Here, softmax would be applied
            if(i == num_layers - 2):
                self.layers.append(SoftMaxModule())
                
            # It would be true only for the first run for rest of the layers this will be False 
            is_input_layer = False
        

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        for layer in self.layers : 
            
            # This new x is the transformed x
            out = layer.forward(x)
            x  = np.copy(out)    


        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        for layer in reversed(self.layers): 
            
            # use gradients from previous module(in Backward pass)
            out = layer.backward(dout)
            dout  = np.copy(out) 

        pass

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################
