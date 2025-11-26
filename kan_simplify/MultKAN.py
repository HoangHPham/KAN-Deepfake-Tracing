import os
import copy
import yaml
import sympy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy import *
from tqdm import tqdm

import torch
import torch.nn as nn

from .KANLayer import KANLayer
from .spline import curve2coef


class MultKAN(nn.Module):
    '''
    KAN class
    
    Attributes:
    -----------
        grid : int
            the number of grid intervals
        k : int
            spline order
        act_fun : a list of KANLayers
        depth : int
            depth of KAN
        width : list
            number of neurons in each layer.
            Without multiplication nodes, [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
            With multiplication nodes, [2,[5,3],[5,1],3] means besides the [2,5,5,3] KAN, there are 3 (1) mul nodes in layer 1 (2). 
        mult_arity : int, or list of int lists
            multiplication arity for each multiplication node (the number of numbers to be multiplied)
        grid : int
            the number of grid intervals
        base_fun : fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        width_in : list
            The number of input neurons for each layer
        width_out : list
            The number of output neurons for each layer
        base_fun_name : str
            Name of the base function b(x)
        grip_eps : float
            The parameter that interpolates between uniform grid and adaptive grid (based on sample quantile)
        node_bias : a list of 1D torch.float (bias of node scaling affine transformation: y = Ax + b)
        node_scale : a list of 1D torch.float (scale of node scaling affine transformation: y = Ax + b)
        subnode_bias : a list of 1D torch.float (bias of subnode scaling affine transformation: y = Ax + b)
        subnode_scale : a list of 1D torch.float (scale of subnode scaling affine transformation: y = Ax + b)
        affine_trainable : bool
            indicate whether affine parameters are trainable (node_bias, node_scale, subnode_bias, subnode_scale)
        sp_trainable : bool w_s
            indicate whether the overall magnitude of splines is trainable
        sb_trainable : bool w_b
            indicate whether the overall magnitude of base function is trainable
        save_act : bool
            indicate whether intermediate activations are saved in forward pass
        node_scores : None or list of 1D torch.float
            node attribution score
        edge_scores : None or list of 2D torch.float
            edge attribution score
        subnode_scores : None or list of 1D torch.float
            subnode attribution score
        cache_data : None or 2D torch.float
            cached input data
        acts : None or a list of 2D torch.float
            activations on nodes
        auto_save : bool
            indicate whether to automatically save a checkpoint once the model is modified
        state_id : int
            the state of the model (used to save checkpoint)
        ckpt_path : str
            the folder to store checkpoints
        round : int
            the number of times rewind() has been called
        device : str
    '''
    
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, auto_save=True, first_init=True, ckpt_path='./saved_kan_model', state_id=0, round=0, device='cpu'):
        '''
        initalize a KAN model
        
        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default update_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str
            
        Returns:
        --------
            self
        '''
        
        super(MultKAN, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        ### initializing the numerical front ###
        
        self.depth = len(width) - 1 # dont count input layer
        
        for i in range(len(width)):
            if type(width[i]) == int or type(width[i]) == np.int64:
                width[i] = [width[i], 0] # n_add_nodes -> [n_add_nodes, n_mult_nodes]
        self.width = width
        
        # mult_arity: the number of nodes to be multiplied for each multiplication node
        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required. 
        self.mult_arity = mult_arity
        
        width_in = self.width_in
        width_out = self.width_out
        
        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.
            
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        
        self.act_fun = []
        for l in range(self.depth):
            # splines
            if isinstance(grid, list):
                grid_l = grid[l]
            else:
                grid_l = grid
                
            if isinstance(k, list):
                k_l = k[l]
            else:
                k_l = k
                
            sp_batch = KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], G=grid_l, k=k_l, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable)
            self.act_fun.append(sp_batch) # KANLayer includes learnable activation functions
        self.act_fun = nn.ModuleList(self.act_fun)
        
        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        # Register node_bias, node_scale, subnode_bias, subnode_scale as parameters
        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')
        
        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        
        self.save_act = save_act
            
        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None
        
        self.cache_data = None
        self.acts = None
        
        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round
        
        self.device = device
        self.to(device)
        
        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path+'/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path+'/'+'0.0')
            else:
                self.state_id = state_id
            
        self.input_id = torch.arange(self.width_in[0],)
    
    
    def to(self, device):
        '''
        move the model to device
        
        Args:
        -----
            device : str or device

        Returns:
        --------
            self
        '''
        super(MultKAN, self).to(device)
        self.device = device
        
        for kanlayer in self.act_fun:
            kanlayer.to(device)
            
        return self
    
    
    
    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0] + width[l][1] for l in range(len(width))] # n_add_nodes + n_mult_nodes (n_nodes before activations of the next layer)
        return width_in
    
    
    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        width = self.width
        if self.mult_homo == True: # if mult_arity is a scalar
            # width[l][0]: number of additive nodes
            # width[l][1]: number of multiplicative nodes
            # mult_arity: number of numbers for multiplication of each multiplicative node
            # ==> mult_arity * width[l][1] = number of subnodes for all multiplicative nodes
            width_out = [width[l][0] + self.mult_arity * width[l][1] for l in range(len(width))] # (n_nodes after activation of the current layer)
        else: # if mult_arity is a list of number of multiplications for each multiplicative node
            width_out = [width[l][0] + int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out
    
    
    @property
    def n_sum(self):
        '''
        The number of addition nodes for each layer
        '''
        width = self.width
        n_sum = [width[l][0] for l in range(1, len(width)-1)]
        return n_sum
    
    
    @property
    def n_mult(self):
        '''
        The number of multiplication nodes for each layer
        '''
        width = self.width
        n_mult = [width[l][1] for l in range(1, len(width)-1)]
        return n_mult
    
    @property
    def feature_score(self):
        '''
        attribution scores for inputs
        '''
        self.attribute()
        if self.node_scores == None:
            return None
        else:
            return self.node_scores[0]
        
    
    def update_grid(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None
        ''' 
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])
            
    
    def saveckpt(self, path='saved_kan_model'):
        '''
        save the current model to files (configuration file and state file)
        
        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            None
        '''
    
        model = self
        
        dic = dict(
            width = model.width,
            grid = model.grid,
            k = model.k,
            mult_arity = model.mult_arity,
            base_fun_name = model.base_fun_name,
            affine_trainable = model.affine_trainable,
            grid_eps = model.grid_eps,
            grid_range = model.grid_range,
            sp_trainable = model.sp_trainable,
            sb_trainable = model.sb_trainable,
            state_id = model.state_id,
            auto_save = model.auto_save,
            ckpt_path = model.ckpt_path,
            round = model.round,
            device = str(model.device)
        )
        
        if dic["device"].isdigit():
            dic["device"] = int(model.device)

        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state')
        torch.save(model.cache_data, f'{path}_cache_data')
        
    
    @staticmethod
    def loadckpt(path='saved_kan_model'):
        '''
        load checkpoint from path
        
        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MultKAN
        '''
        
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
            
        state = torch.load(f'{path}_state')
        
        # create new model from pre-defined config
        model_load = MultKAN(width=config['width'], 
                     grid=config['grid'], 
                     k=config['k'], 
                     mult_arity = config['mult_arity'], 
                     base_fun=config['base_fun_name'], 
                     affine_trainable=config['affine_trainable'], 
                     grid_eps=config['grid_eps'], 
                     grid_range=config['grid_range'], 
                     sp_trainable=config['sp_trainable'],
                     sb_trainable=config['sb_trainable'],
                     state_id=config['state_id'],
                     auto_save=config['auto_save'],
                     first_init=False,
                     ckpt_path=config['ckpt_path'],
                     round = config['round']+1,
                     device = config['device'])
        
        model_load.load_state_dict(state)
        model_load.cache_data = torch.load(f'{path}_cache_data')
        
        return model_load
    
    
    def log_history(self, method_name): 
        if self.auto_save:
            # save to log file
            with open(self.ckpt_path + '/history.txt', 'a') as file:
                file.write(str(self.round) + '.' + str(self.state_id) + ' => ' + method_name + ' => ' + str(self.round) + '.' + str(self.state_id+1) + '\n')

            # update state_id
            self.state_id += 1

            # save to ckpt
            self.saveckpt(path=self.ckpt_path + '/' + str(self.round) + '.' + str(self.state_id))
            print('saving model version ' + str(self.round) + '.' + str(self.state_id))
            
    
    def attribute(self, l=None, i=None, out_score=None, plot=True):
        '''
        get attribution scores
        Docs: 4.1. Identifying important features from KANs (Page 8 KAN 2.0)

        Args:
        -----
            l : None or int
                layer index (1-indexed)
            i : None or int
                neuron index
            out_score : None or 1D torch.float
                specify output scores
            plot : bool
                when plot = True, display the bar show
            
        Returns:
        --------
            attribution scores, shape = [out_dim, in_dim]
        '''
        
        device = self.act_fun[0].grid.device
        
        if l != None:
            self.attribute()
            out_score = self.node_scores[l]
            
        if self.acts == None:
            self.get_act()
            
        def score_node2subnode(node_score, width, mult_arity, out_dim):

            assert np.sum(width) == node_score.shape[1]

            # scores of subnodes for additive nodes
            subnode_score = node_score[:, :width[0]] 
            
            # scores of subnodes for multiplicative nodes
            if isinstance(mult_arity, int):
                subnode_score = torch.cat([subnode_score, node_score[:, width[0]:][:,:,None].expand(out_dim, node_score[:, width[0]:].shape[1], mult_arity).reshape(out_dim, -1)], dim=1)
            else:
                for i in range(len(mult_arity)):
                    subnode_score = torch.cat([subnode_score, node_score[:, width[0] + i].expand(out_dim, mult_arity[i])], dim=1)
            return subnode_score
        
        node_scores = []
        subnode_scores = []
        edge_scores = []
        
        l_query = l
        if l == None:
            l_end = self.depth
        else:
            l_end = l
            
        # back propagate from the queried layer
        out_dim = self.width_in[l_end] # The number of input nodes in l_end^th layer
        if out_score == None:
            node_score = torch.eye(out_dim).requires_grad_(True) # 1 for all scores (the last layer)
        else:
            node_score = torch.diag(out_score).requires_grad_(True)
        node_scores.append(node_score) 
        
        for l in range(l_end, 0, -1):
            # node to subnode 
            if isinstance(self.mult_arity, int):
                subnode_score = score_node2subnode(node_score, self.width[l], self.mult_arity, out_dim=out_dim)
            else:
                mult_arity = self.mult_arity[l]
                subnode_score = score_node2subnode(node_score, self.width[l], mult_arity, out_dim=out_dim)

            subnode_scores.append(subnode_score)
            
            ## Equation (9) KAN 2.0
            # subnode to edge
            edge_score = torch.einsum('ij,ki,i->kij', self.edge_actscale[l-1], subnode_score.to(device), 1 / (self.subnode_actscale[l-1] + 1e-4))
            edge_scores.append(edge_score)

            # edge to node
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)
            
        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
        self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
        self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]
        
        # return
        if l_query != None:
            if i == None:
                return self.node_scores_all[0]
            else:
                # plot
                if plot:
                    in_dim = self.width_in[0]
                    plt.figure(figsize=(1*in_dim, 3))
                    plt.bar(range(in_dim), self.node_scores_all[0][i].cpu().detach().numpy())
                    plt.xticks(range(in_dim))

                return self.node_scores_all[0][i]
         
            
    def node_attribute(self):
        self.node_attribute_scores = []
        for l in range(1, self.depth+1):
            node_attr = self.attribute(l)
            self.node_attribute_scores.append(node_attr)
            
    
    def get_act(self, x=None):
        '''
        collect intermidate activations
        '''
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = save_act
        
    
    def forward(self, x):
        '''
        forward pass
        
        Args:
        -----
            x : 2D torch.tensor
                inputs [#n_samples, #in_dim]
        Returns:
        --------
            None
        '''
        
        x = x[:, self.input_id.long()]
        assert x.shape[1] == self.width_in[0]
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        
        self.acts.append(x)  # acts shape: (batch, width[l])
        
        for l in range(self.depth):
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x) # KANLayer forward # [#n_samples, #n_(l+1)]
            
            # subnodes
            x = x_numerical # adding auxiliary variable
            
            if self.save_act:
                # save subnode_scale
                if x.shape[0] != 1:
                    self.subnode_actscale.append(torch.std(x, dim=0).detach()) # std of samples
                else:
                    self.subnode_actscale.append(torch.abs(x[0]).detach()) # if batch size = 1, use abs value instead of std

            # subnode affine transform
            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :] # x = Ax + b
            
            if self.save_act:
                
                postacts = postacts_numerical

                if preacts.shape[0] != 1:
                    input_range = torch.std(preacts, dim=0) + 0.1
                else:
                    input_range = torch.abs(preacts[0])
                    
                if postacts_numerical.shape[0] != 1:
                    output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                else:
                    output_range_spline = torch.abs(postacts_numerical[0])
                    
                if postacts.shape[0] != 1:
                    output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from spline
                else:
                    output_range = torch.abs(postacts[0])
                
                # save edge_scale
                self.edge_actscale.append(output_range)
                
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())
                
                self.acts_premult.append(x.detach())
                
            # multiplicative nodes
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True: # if mult_arity is a scalar
                for i in range(self.mult_arity - 1):
                    if i == 0:
                        x_mult = x[:, dim_sum::self.mult_arity] * x[:, dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:, dim_sum+i+1::self.mult_arity]
            else: # if mult_arity is a list of number of multiplications for each multiplicative node
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j]) # total number of subnodes
                    for i in range(self.mult_arity[l+1][j] - 1):
                        if i == 0:
                            x_mult_j = x[:, [acml_id]] * x[:, [acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:, [acml_id+i+1]]
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                        
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:, :dim_sum], x_mult], dim=1) # nodes (additive nodes is kept unchanged from subnodes + multiplicative nodes)
                
            # x = Ax + b
            # node affine transform
            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]
            
            self.acts.append(x.detach())
            
        return x
    
    
    def get_reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff):
        '''
        Get regularization
        Docs: 2.5.1 Simplification techniques - Sparsification regularization (Page 11 KAN 1.0)
        
        Args:
        -----
            reg_metric : the regularization metric
                'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient penalty strength
            lamb_coefdiff : float
                coefficient smoothness strength
        
        Returns:
        --------
            reg_ : torch.float
        '''
        
        if reg_metric == 'edge_forward_spline_n':
            acts_scale = self.acts_scale_spline
            
        elif reg_metric == 'edge_forward_sum':
            acts_scale = self.acts_scale
            
        elif reg_metric == 'edge_forward_spline_u':
            acts_scale = self.edge_actscale
            
        elif reg_metric == 'edge_backward':
            acts_scale = self.edge_scores
            
        elif reg_metric == 'node_backward':
            acts_scale = self.node_attribute_scores
            
        else:
            raise Exception(f'reg_metric = {reg_metric} not recognized!')
        
        reg_ = 0.
        for i in range(len(acts_scale)):
            vec = acts_scale[i]

            # Sparsification regularization (see Page 11 KAN 1.0)
            l1 = torch.sum(vec)
            p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
            p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
            entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
            reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row + entropy_col)  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(self.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_
    
    
    def fit(self, dataset, opt="Adam", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
              metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', reg_metric='edge_forward_spline_n', display_metrics=None):
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics
        '''
        
        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2) # MSE
        else:
            loss_fn = loss_fn_eval = loss_fn
            
        grid_update_freq = int(stop_grid_update_step / grid_update_num) # update grid after every grid_update_freq steps
        
        # Optimization
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # save results
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []
                
        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0] # full batch
            batch_size_test = dataset['test_input'].shape[0] # full batch
        else:
            batch_size = batch # mini-batch
            batch_size_test = batch # mini-batch
            
        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
              
        global train_loss, reg_
        
        pbar = tqdm(range(steps), desc='description', ncols=100) # progress bar
        for _ in pbar:
            if _ == steps-1 and self.save_act:
                self.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
        
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False) # random batch of training samples
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)  # random batch of testing samples

            # if it is still able to update grid
            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id]) 
                
            pred = self.forward(dataset['train_input'][train_id]) # MultKAN model forward
            train_loss = loss_fn(pred, dataset['train_label'][train_id]) # prediction train loss
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute() # edge scores
                if reg_metric == 'node_backward':
                    self.node_attribute() # node scores
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff) # total regularization
            else:
                reg_ = torch.tensor(0.)
            loss = train_loss + lamb * reg_ # total loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id]) # prediction test loss
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item()) # save metric scores

            # save results
            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())
            
            # setup progress bar
            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
            # save figure
            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act
                
        # write log
        self.log_history('fit')
        return results
    
    
    def module_inFeatures(self, modular_structures):
        
        print("Adding modular structure for KAN")
        
        cl = 0
        
        in_dim  = self.width_in[cl]
        out_dim = self.width_out[cl+1]

        # Khởi tạo mask = 0
        mask_np = np.zeros((in_dim, out_dim), dtype=np.float32)

        # Bật các kết nối hợp lệ (cartesian product input-group x output-group)
        for ins, outs in modular_structures:
            mask_np[np.ix_(ins, outs)] = 1.0

        # Gán lại một lần cho layer
        mask_t = torch.as_tensor(mask_np, device=self.act_fun[cl].mask.device, dtype=self.act_fun[cl].mask.dtype)
        self.act_fun[cl].mask.data.copy_(mask_t)
        
        self.log_history('module')
        
    
    def get_fun(self, l, i, j):
        '''
        get function (l,i,j)
        '''
        inputs = self.spline_preacts[l][:,j,i].cpu().detach().numpy()
        outputs = self.spline_postacts[l][:,j,i].cpu().detach().numpy()
        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]
        plt.figure(figsize=(3,3))
        plt.plot(inputs, outputs, marker="o")
        return inputs, outputs
    
    
    def history(self, k='all'):
        '''
        get history
        '''
        with open(self.ckpt_path+'/history.txt', 'r') as f:
            data = f.readlines()
            n_line = len(data)
            if k == 'all':
                k = n_line

            data = data[-k:]
            for line in data:
                print(line[:-1])
                
    
    @property
    def n_edge(self):
        '''
        the number of active edges
        '''
        depth = len(self.act_fun)
        complexity = 0
        for l in range(depth):
            complexity += torch.sum(self.act_fun[l].mask > 0.)
        return complexity.item()
    
    
    def plot(self, folder="./figures", beta=3, metric='backward', scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0):
        '''
        plot KAN
        
        Args:
        -----
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title
            varscale : float
                the size of input variables
            
        Returns:
        --------
            Figure
        '''
        global Symbol
        
        if not self.save_act:
            print('cannot plot since data are not saved. Set save_act=True first.')
        
        # forward to obtain activations
        if self.acts == None:
            if self.cache_data == None:
                raise Exception('model hasn\'t seen any data yet.')
            self.forward(self.cache_data)
            
        if metric == 'backward':
            self.attribute()
         
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l+1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    numeric_mask = self.act_fun[l].mask[i][j]
                    if numeric_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    elif numeric_mask == 0.:
                        color = "white"
                        alpha_mask = 0

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        
        if metric == 'forward_n':
            scores = self.acts_scale
        elif metric == 'forward_u':
            scores = self.edge_actscale
        elif metric == 'backward':
            scores = self.edge_scores
        else:
            raise Exception(f'metric = \'{metric}\' not recognized')
        
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
            
        # draw skeleton
        width = np.array(self.width)
        width_in = np.array(self.width_in)
        width_out = np.array(self.width_out)
        A = 1
        y0 = 0.2  # height: from input to pre-mult
        z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width_out), 5)

        max_neuron = np.max(width_out)
        max_num_weights = np.max(width_in[:-1] * width_out[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 5) # size (height/width) of 1D function diagrams
        y2 = 0.15 / np.maximum(max_neuron, 5) # size (height/width) of operations (sum and mult)

        fig, ax = plt.subplots(figsize=(10 * scale, 30 * scale * (neuron_depth - 1) * (y0+z0)))

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
        
        # plot scatters and lines
        for l in range(neuron_depth):
            
            n = width_in[l]
            
            # scatters
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * (y0+z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
                
            # plot connections (input to pre-mult)
            for i in range(n):
                if l < neuron_depth - 1:
                    n_next = width_out[l+1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        numeric_mask = self.act_fun[l].mask[i][j]
                        if numeric_mask > 0.:
                            color = "black"
                            alpha_mask = 1
                        elif numeric_mask == 0.:
                            color = "white"
                            alpha_mask = 0
                        
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=min(1., alpha[l][j][i] * alpha_mask + 0.02))
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=min(1., alpha[l][j][i] * alpha_mask + 0.02))
                            
                            
            # plot connections (pre-mult to post-mult, post-mult = next-layer input)
            if l < neuron_depth - 1:
                n_in = width_out[l+1]
                n_out = width_in[l+1]
                mult_id = 0
                for i in range(n_in):
                    if i < width[l+1][0]:
                        j = i
                    else:
                        if i == width[l+1][0]:
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        if current_mult_arity == 0:
                            mult_id += 1
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        j = width[l+1][0] + mult_id
                        current_mult_arity -= 1
                        #j = (i-width[l+1][0])//self.mult_arity + width[l+1][0]
                    plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0+z0) + y0, (l+1) * (y0+z0)], color='black', lw=2 * scale)

            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0+z0), (neuron_depth - 1 + 0.1) * (y0+z0))


        plt.axis('off')

        for l in range(neuron_depth - 1):
            # plot splines
            n = width_in[l]
            for i in range(n):
                n_next = width_out[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0+z0) + y0/2 - y1])[1]
                    up = DC_to_NFC([0, l * (y0+z0) + y0/2 + y1])[1]
                    
                    # print(f'left: {left}, right: {right}, bottom: {bottom}, up: {up}')
                    
                    newax = fig.add_axes([left, bottom, right - left + 0.0005, up - bottom + 0.0005])
                    # newax.imshow(im, alpha=alpha[l][j][i])
                    newax.imshow(im, alpha=1)
                    newax.axis('off')
                    
              
            # plot sum symbols
            N = n = width_out[l+1]
            for j in range(n):
                id_ = j
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/sum_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, l * (y0+z0) + y0 - y2])[1]
                up = DC_to_NFC([0, l * (y0+z0) + y0 + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')
                
            # plot mult symbols
            N = n = width_in[l+1]
            n_sum = width[l+1][0]
            n_mult = width[l+1][1]
            for j in range(n_mult):
                id_ = j + n_sum
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/mult_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, (l+1) * (y0+z0) - y2])[1]
                up = DC_to_NFC([0, (l+1) * (y0+z0) + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')

        if in_vars != None:
            n = self.width_in[0]
            for i in range(n):
                if isinstance(in_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.04, f'${latex(in_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center', rotation='vertical')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.04, in_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center', rotation='vertical')
                
        if out_vars != None:
            n = self.width_in[-1]
            for i in range(n):
                if isinstance(out_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.01, f'${latex(out_vars[i])}$', fontsize=60 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.01, out_vars[i], fontsize=60 * scale * varscale, horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        plt.savefig("plots/model_tree.png")
        plt.show()
        plt.close()
    

KAN = MultKAN