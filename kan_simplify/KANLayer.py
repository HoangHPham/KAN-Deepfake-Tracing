import numpy as np

import torch
import torch.nn as nn

from .spline import *



class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        G: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            w_b: magnitude of the residual activation function b(x) is drawn from N(mu, sigma^2), mu = scale_base_mu. See page 6 KAN 1.0.
        scale_base_sigma: float
            w_b: magnitude of the residual activation function b(x) is drawn from N(mu, sigma^2), sigma = scale_base_sigma. See page 6 KAN 1.0.
        scale_sp: float
            w_s: mangitude of the spline function spline(x). See page 6 KAN 1.0.
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """
    
    def __init__(self, in_dim=3, out_dim=2, G=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu'):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            G : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                w_b: the scale of the residual activation function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2). See page 6 KAN 1.0.
            scale_base_sigma : float
                w_b: the scale of the residual activation function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2). See page 6 KAN 1.0.
            scale_sp : float
                w_s: the scale of the base function spline(x). See page 6 KAN 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp (w_s) is trainable
            sb_trainable : bool
                If true, scale_base (w_b) is trainable
            device : str
                device
            
        Returns:
        --------
            self
            
        '''
        
        super(KANLayer, self).__init__()
        
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.G = G
        self.k = k # order = degree + 1
        
        # grid
        grid = torch.linspace(grid_range[0], grid_range[1], steps=G + 1).unsqueeze(dim=0).expand(self.in_dim, G + 1) # shape = [n_l, G + 1]
        grid = extend_grid(grid, k_extend=k) # extend grid points to [t_(-k),...,t_(G+k)] => shape = [n_l, G+1+2*k]
        self.grid = torch.nn.Parameter(grid).requires_grad_(False) # register as trainable parameters
        
        # noise to evaluate spline
        noises = (torch.rand(self.G + 1, self.in_dim, self.out_dim) - 1/2) * noise_scale / G # shape = [n_grid_points, n_l, n_(l+1)]
        
        # grid[:, k:-k] : t_0 -> t_G
        # register spline coefficients as trainable parameters
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k)) # shape = [n_l, n_(l+1), G+k] 
        
        self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        # Initialize with random from Uniform distribution - Xavier initialization
        # scale of residual activation function b(x): w_b
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        
        # b(x)
        self.base_fun = base_fun # SiLU

        # scale of the base function spline(x): w_s
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        
        # a hyperparameter used in update_grid_from_samples.
        self.grid_eps = grid_eps

        self.to(device)
        
    
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self


    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                pre-activation
                fan out x into activations, shape (number of samples, output dimension, input dimension)
            postacts : 3D torch.float
                post-activation phi(x) = w_b.b(x) + w_s.spline(x)
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                spline(x)
                the outputs of spline functions with preacts as inputs
        '''
        
        batch = x.shape[0] # n_samples
        
        # prepare input for n_l*n_(l+1) activations (each input node has n_(l+1) activation)
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim) # shape = [n_samples, n_(l+1), n_l]
        
        # b(x)
        base = self.base_fun(x) # (batch, in_dim) = (n_samples, n_l)

        # spline(x)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k) # [n_samples, n_l, n_(l+1)]
        postspline = y.clone().permute(0,2,1) # [n_samples, n_(l+1), n_l]

        # phi(x) = w_b.b(x) + w_s.spline(x)
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1) # [n_samples, n_(l+1), n_l]

        # additive node
        y = torch.sum(y, dim=1) # shape = [n_samples, n_(l+1)]
        
        return y, preacts, postacts, postspline


    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        '''
        
        batch = x.shape[0] # n_samples

        # sort x ascending
        x_pos = torch.sort(x, dim=0)[0] # shape = [n_samples, in_dim]
        
        # evaluate x on B-spline curves
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k) # shape = [n_samples, n_l, n_(l+1)]
        
        num_interval = self.grid.shape[1] - 1 - 2*self.k # current G (not count the extended grid points)

        def get_grid(num_interval):
            # split n_samples into equal ranges corresponding to ranges of grid intervals
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1] # => There are G ranges
            
            # percentiles of samples: select samples corresponding to ranges of G intervals
            grid_adaptive = x_pos[ids, :].permute(1,0)  # shape = [in_dim, G+1]
            
            # calculate value of each grid interval
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval # (max - min)/G
            
            # grid is uniform: create grid corresponding to the range value of x (min, max) with G intervals
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            
            # When grid_eps = 1, the grid is uniform; 
            # when grid_eps = 0, the grid is partitioned using percentiles of samples;
            # when 0 < grid_eps < 1 interpolates between the two extremes.
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive # exponentially weighted average (EWA) formula
            return grid
        
        grid = get_grid(num_interval=num_interval)

        # extend from G+1 grid points to (G+1)+2*k grid points
        self.grid.data = extend_grid(grid, k_extend=self.k) # shape = [n_l, G+1+2*k]
        
        # create new spline coefficients
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k) # shape = [n_l, n_(l+1), G+k]