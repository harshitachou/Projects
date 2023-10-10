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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        self.params = {'weight': None, 'bias': None}

        # d = number of input variables to the layer
        d = in_features
        self.params['weight'] = np.random.normal(0,2/d, size = (out_features,in_features))
        self.params['bias'] = np.zeros(out_features)

        self.grads = {'weight': None, 'bias': None}

        self.grads['weight'] = np.zeros((out_features,in_features))
        self.grads['bias'] =  np.zeros(out_features)
        self.input_layer = input_layer

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x = x
        out =  x@self.params['weight'].T + np.tile(self.params['bias'],(x.shape[0], 1))

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        self.grads['weight'] =  dout.T @ self.x
        
        # add gradients for each output dimension(over the samples)
        self.grads['bias'] = np.sum(dout,axis = 0)
        
        dx = dout@self.params['weight']
        # Basically dy/dx of linear layer.
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        
        out = np.where(x>0 , x , np.exp(x)-1)
        self.out = out

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        # derivative of ELU(y) wrt y
        dELU_dy = np.where(self.out > 0, 1 , self.out + 1)
        
        # element wise product
        dx = np.multiply(dout, dELU_dy)
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # newaxis : to convert it into a column vector                 
        b = np.amax(x,axis=1)[:, np.newaxis]
        out = np.exp(x-b) / (np.sum(np.exp(x-b),axis = 1)[:, np.newaxis])
        self.x = out
        self.out = out
        
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        z = np.multiply(dout,self.out) @ np.ones((self.out.shape[1],self.out.shape[1]))
        u = dout - z
        dx = np.multiply(self.out,u)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
       
        num_classes = x.shape[1]

        # convert labels into one-hot encoded matrix
        label_matrix = np.eye(num_classes)[y]

        logx = np.log(x)
        loss = np.sum(np.multiply(-label_matrix, logx),axis = 1)
        sum_all_losses = np.sum(loss) 
        out = sum_all_losses/len(loss)

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        num_sample = x.shape[0]
        num_classes = x.shape[1]

        # convert labels into one-hot encoded matrix
        label_matrix = np.eye(num_classes)[y]

        dx = -1/num_sample*(np.divide(label_matrix,x))

        return dx