import torch


def iterative_svd_filter(data_mat: torch.Tensor, r: int, ratio: float, iter_num: int) -> torch.Tensor:
    """
    Iteratively filter out samples with high reconstruction errors using SVD decomposition.
    In each iteration, equal number of samples with highest reconstruction errors are removed.
    
    Args:
        data_mat (torch.Tensor): Input data matrix of shape (N, D), where N is number of samples
                                and D is feature dimension
        r (int): Number of principal components to keep, must be less than D
        ratio (float): Ratio of samples to keep in final output, must be between 0 and 1
        iter_num (int): Number of iterations for filtering
        
    Returns:
        torch.Tensor: Filtered data matrix of shape (N_kept, D) where N_kept = N * ratio
    """
    
    device = data_mat.device

    # Input validation
    if not (0 < ratio < 1):
        raise ValueError("ratio must be between 0 and 1")
    if r >= data_mat.shape[1]:
        raise ValueError("r must be less than feature dimension")
    if iter_num < 1:
        raise ValueError("iter_num must be at least 1")
    
    curr_data = data_mat.clone().to(device)
    
    for i in range(iter_num):
        # Step 1: Perform SVD
        U, S, Vh = torch.linalg.svd(curr_data, full_matrices=False)  
        
        # Step 2: Construct projection matrix and get reconstruction
        V_r = Vh[:r, :].T.to(device)
        proj_mat = V_r @ V_r.T
        recon_mat = curr_data @ proj_mat
        
        # Step 3: Calculate reconstruction error for each sample
        errors = torch.mean((curr_data - recon_mat)**2, dim=1)
        
        # Step 4: Determine number of samples to keep in this iteration
        n_keep = max(int(curr_data.shape[0] * ratio), 1)
            
        # Step 5: Keep samples with lowest reconstruction errors
        _, indices = torch.topk(errors, n_keep, largest=False)
        curr_data = curr_data[indices]
            
    return curr_data.to(device)


if __name__ == "__main__":

    torch.manual_seed(0)
    
    N, D = 64, 2048
    test_data = torch.randn(N, D)

    # Hyperparameters
    r = 10
    ratio = 0.6
    iter_num = 3 
    
    filtered_data = iterative_svd_filter(test_data, r, ratio, iter_num)
    
    print(f"Original data shape: {test_data.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")