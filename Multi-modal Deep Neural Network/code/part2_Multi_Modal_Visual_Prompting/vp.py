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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np

class VerticalLines(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(VerticalLines, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        # pad size 15*15 and gap 15
        # create 7 such patches
        self.image_size = image_size
        self.patch_1 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_2 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_3 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_4 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_5 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_6 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.patch_7 = nn.Parameter(torch.randn(1, 3, image_size, 15),requires_grad=True)
        self.device = args.device

    def forward(self, x):
       
        prompt = torch.zeros(1,3,self.image_size,self.image_size).to(self.device)
      
        index = 0
        prompt[:,:,:,index:index+15] = self.patch_1
        index = index + 30

        prompt[:,:,:,index:index+15] = self.patch_2
        index = index + 30


        prompt[:,:,:,index:index+15] = self.patch_3
        index = index + 30

        prompt[:,:,:,index:index+15] = self.patch_4
        index = index + 30

        prompt[:,:,:,index:index+15] = self.patch_5
        index = index + 30

        prompt[:,:,:,index:index+15] = self.patch_6
        index = index + 30

        prompt[:,:,:,index:index+15] = self.patch_7
        index = index + 30

        prompt = torch.tile(prompt,(x.shape[0],1,1,1))

        return torch.mul(x, prompt) 

class CrossPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(CrossPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # to be used in forward function
        self.image_size = image_size
        self.patch_size = pad_size

        self.device = args.device

        self.pad_vertical = nn.Parameter(torch.randn(1, 3, image_size, 2*pad_size),requires_grad=True)
        self.pad_horizontal = nn.Parameter(torch.randn(1, 3,2*pad_size, image_size),requires_grad=True)


    def forward(self, x):
        

        prompt = torch.zeros(1,3,self.image_size,self.image_size).to(self.device)

        mid = int(self.image_size/2)

        prompt[:,:,:,mid-self.patch_size:mid+self.patch_size] = self.pad_vertical
        prompt[:,:,mid-self.patch_size:mid+self.patch_size,:] = self.pad_horizontal

        # The above prompt is for one batch. We need to make for x.shape[0] batches
        # repeat x.shape[0] times along 0th dimension
        prompt = torch.tile(prompt,(x.shape[0],1,1,1))

        return x+prompt

class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # to be used in forward function
        self.mid = image_size-2*pad_size

        self.device = args.device

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down
        self.pad_up = nn.Parameter(torch.randn(1, 3, pad_size, image_size),requires_grad=True)
        self.pad_down = nn.Parameter(torch.randn(1, 3, pad_size, image_size),requires_grad=True)
        self.pad_left = nn.Parameter(torch.randn(1, 3, image_size-2*pad_size, pad_size),requires_grad=True)
        self.pad_right = nn.Parameter(torch.randn(1, 3, image_size-2*pad_size, pad_size),requires_grad=True)

        

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
            # Declaring them as parameters of nn.module . So that they can be accessed and learnt.
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right

    def forward(self, x):
        # TODO: For a given batch of images, add the prompt as a padding to the image.
        prompt = torch.zeros(1,3,self.mid,self.mid).to(self.device)

        # along column : last dimension
        prompt = torch.cat([self.pad_left,prompt,self.pad_right],dim=3)

        # along rows
        prompt = torch.cat([self.pad_up,prompt,self.pad_down],dim=2)

        # The above prompt is for one batch. We need to make for x.shape[0] batches
        # repeat x.shape[0] times along 0th dimension
        prompt = torch.tile(prompt,(x.shape[0],1,1,1))

        return x+prompt 

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.



class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.
        # To be used in forward function
        self.i_size = args.image_size
        self.patch_size = args.prompt_size 
    
        self.patch = nn.Parameter(torch.randn(1,3,self.patch_size,self.patch_size))
        self.device = args.device



        
        
        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

    
    def forward(self, x):
      
        # along rows
        prompt = torch.cat([self.patch,torch.zeros(1,3,self.i_size-self.patch_size,self.patch_size).to(self.device)],dim=2)

        #along columns
        prompt = torch.cat([prompt,torch.zeros(1,3,self.i_size,self.i_size-self.patch_size).to(self.device)],dim=3)

        prompt = torch.tile(prompt,(x.shape[0],1,1,1))

        return x+prompt 
        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # To be used in forward function
        self.image_size = args.image_size
        self.patch_size = args.prompt_size
        self.device = args.device

        self.patch = nn.Parameter(torch.randn(1,3,self.patch_size,self.patch_size),requires_grad=True)

        
        
        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # random location : top left corner of patch : row and column number
        top_left = torch.randint(0,self.image_size-1,(2,)).to(self.device)
        #print(top_left)

        prompt = torch.zeros(1,3,self.image_size,self.image_size).to(self.device)
        prompt[:,:,top_left[0]:self.patch_size+top_left[0],top_left[1]:self.patch_size+top_left[1]] = self.patch
        #print(prompt)


        return x+prompt

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.



