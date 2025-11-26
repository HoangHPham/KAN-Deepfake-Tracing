import torch


def B_batch(x, grid, k=0):
    '''
    evaluate x on B-spline bases (feed x into B-spline basis functions)
    
    Args:
    -----
        x : 2D torch.tensor 
            inputs, shape (number of samples, in_dim) [n_samples, n_l] 
        grid : 2D torch.tensor
            grids, shape (in_dim, total number of grid points) # (n_l, (G+1)+2*k) if extended=True, otherwise (n_l, G+1)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor 
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2) # [n_samples, n_l, 1]
    grid = grid.unsqueeze(dim=0) # [1, n_l, G+1+2*k]
    
    if k == 0:
        # 0-level B-spline
        # if t_i <= x < t_(i+1) => B_0(x) = 1 else B_0(x) = 0
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else: # k = 1, 2, ..., order
        # Recursive function using De Boor-Cox algorithm
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value) # avoid NaN, positive infinity, and negative infinity values 
    return value # [n_samples, in_dim, G+k]


def coef2curve(x_eval, grid, coef, k):
    '''
    converting B-spline coefficients to B-spline curves. 
    Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, (G+1)+2*k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    ''' 
    
    # evaluate x on B-spline bases: number of x is n_samples and total number of B-spline bases is G+k
    b_splines = B_batch(x_eval, grid, k=k) # shape = [n_samples, in_dim, G+k]
    
    # summing up B_batch results over B-spline basis
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device)) # shape = [n_samples, in_dim, out_dim]
    
    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    '''
    using B-spline curves to estimate B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of grid points at middle, in_dim) # [G+1, n_l]
        y_eval : 3D torch.tensor
            shape (number of grid points at middle, in_dim, out_dim) # [G+1, n_l, n_(l+1)]
        grid : 2D torch.tensor
            shape (in_dim, (G+1)+2*k)
        k : int
            spline order
            
    Returns:
    --------
        coef : 3D torch.tensor
            # G+k : total number of splines corresponding to G+1+2*k grid points
            # in_dim*out_dim*(G+k) : total number of coefficients to build (G+k) B-spline bases for each of in_dim*out_dim activation function
            shape (in_dim, out_dim, G+k) 
    '''
    
    batch = x_eval.shape[0] # G + 1: number of grid points or number of splines in the middle (not count the extended grid points)
    in_dim = x_eval.shape[1] # n_l
    out_dim = y_eval.shape[2] # n_(l+1)
    n_coef = grid.shape[1] - k - 1 # G+1+2*k - k - 1 = G + k = total number of splines (count the extended grid points)
    
    # B-spline matrix
    mat = B_batch(x_eval, grid, k) # shape = [G+1, n_l, G+k] or [n_grid_points_no_extension, in_dim, total_n_splines_with_extension]
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef) # shape = [n_l, n_(l+1), G+1, G+k]
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3) # [n_l, n_(l+1), G+1, 1]
    
    # Least square problem: B.c = y (mat x coef = y_eval) ==> Find c so that : ||Bc-y||^2 min
    try:
        # Solve least square: (B^T.B)^(-1).B^T.y
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0] # shape = [n_l, n_(l+1), G+k] 
        # ==> n_l x n_(l+1) = total number of learnable activation functions (B-splines)
        # ==> n_l x n_(l+1) x (G+k) = total number of coefficients for all learnable activation functions (B-splines) (because each function has G+k basis functions)
    except:
        print('lstsq failed')
    
    return coef


def extend_grid(grid, k_extend=0):
    '''
    extend grid: grid points from [t_0,...,t_G] to [t_(-k),...,t_(G+k)]
    Args:
    -----
        grid : 2D torch.tensor
            shape (in_dim, G+1) or (n_l, G+1)
        k_extend : int
            number of points to extend on both ends. Default: 0.
    '''
    
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1) # h = (max - min) / (#grid_intervals) => h is the size of each grid interval

    for _ in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1) # prepend grid point [t_(-k),...t_(-1)]
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1) # append grid point [t_(G+1),...t_(G+k)]

    return grid