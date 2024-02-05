# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from agents.layers import DoubleConv, Down, Up, OutConv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class UNet(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16):
        super(UNet, self).__init__()      
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        input_dim = state_dim + action_dim + t_dim
        
        self.inc = DoubleConv(input_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, action_dim)
        
    def forward(self, x, time, state):
    
    	t = self.time_mlp(time)
    	x = torch.cat([x, t, state], dim=1)
    	x1 = self.inc(x)
    	x2 = self.down1(x1)
    	x3 = self.down2(x2)
    	x4 = self.down3(x3)
    	x5 = self.down4(x4)
    	x = self.up1(x5, x4)
    	x = self.up2(x, x3)
    	x = self.up3(x, x2)
    	x = self.up4(x, x1)
    	logits = self.outc(x)
    	return logits
    	
