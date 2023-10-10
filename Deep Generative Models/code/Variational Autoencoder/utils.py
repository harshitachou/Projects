################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    e = torch.normal(mean=0, std=1, size = mean.shape).to(device)

    z = mean+std*e

    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    mean_square = torch.square(mean)
    KLD_one_dim =  torch.exp(2*log_std) + mean_square - 1 - 2*log_std
    KLD = 0.5*torch.sum(KLD_one_dim,dim = -1) # per image

    # return of shape [batch]
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """

    bpd = elbo * (torch.ones(elbo.shape[0]).to(device)*torch.log2(torch.exp(torch.tensor(1).to(device)))) / (img_shape[1]*img_shape[2]*img_shape[3])

    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values 
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    # Construct grid of latent variable values
    import torch
    z_percen = torch.arange(start=0.5,end=grid_size,step=1)/grid_size
    normal = torch.distributions.Normal(0,1)
    
    x,y = torch.meshgrid(normal.icdf(z_percen), normal.icdf(z_percen)).to(device)

    x = x.reshape(-1)
    y = y.reshape(-1)

    # shape : [batch,z_dim]
    z_sampled = torch.stack((x, y), dim=1)
    x_samples = decoder(z_sampled)
    m = nn.Softmax(dim=1)

    # apply softmax
    x_samples = m(x_samples) 

    #eg. : [10,16,28,28] -> [10*28*28,16] 
    reshape_x = x_samples_softmax.view(batch_size*x_samples.shape[2]*x_samples.shape[3],x_samples.shape[1])
    # shape : [10*28*28,1]
    x_images = torch.multinomial(reshape_x, 1)
    x_images = x_images.view(batch_size,1,x_samples.shape[2],x_samples.shape[3])

    x_images = x_images.float() / 15  # Converting 4-bit images to values between 0 and 1
    img_grid = make_grid(x_images, nrow=grid_size, normalize=True, value_range=(0, 1), pad_value=0.5)

    return img_grid

