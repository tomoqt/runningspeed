# Old torch Peer and UltraMem mixture code

def topk_approx(s_row, s_col, u, t, m, k):
    """
    Approximated top-m selection using the rank-1 approximation of the Tucker core.
    """
    b, h, t_dim, n_r = s_row.shape  # Get the correct dimensions from s_row
    _, _, _, n_c = s_col.shape      # Get n_c from s_col
    
    # 1. Compute approximated scores using rank-1 tucker
    # Reshape u and t to match expected dimensions
    u_expanded = u.unsqueeze(0).expand(b, -1, -1, -1)  # [B, H, N_r, D_k]
    t_expanded = t.unsqueeze(0).expand(b, -1, -1, -1)  # [B, H, N_c, D_k]
    
    s_row_approx = torch.einsum('bhtr,bhnd->bht', s_row, u_expanded)  # [B, H, T], where n=N_r
    s_col_approx = torch.einsum('bhtc,bhmd->bht', s_col, t_expanded)  # [B, H, T], where m=N_c

    # 2. Find top-m row and column indices.
    topm_row_vals, topm_row_indices = torch.topk(s_row_approx, min(m, n_r), dim=-1)  # [B, H, M]
    topm_col_vals, topm_col_indices = torch.topk(s_col_approx, min(m, n_c), dim=-1)  # [B, H, M]
    
    actual_m = min(m, min(n_r, n_c))  # Ensure m is not larger than n_r or n_c

    # 3. Gather the actual scores (from the original s_row and s_col)
    topm_row_indices_expanded = topm_row_indices[:, :, :actual_m].unsqueeze(-1).expand(-1, -1, -1, s_row.shape[-1])
    topm_col_indices_expanded = topm_col_indices[:, :, :actual_m].unsqueeze(-1).expand(-1, -1, -1, s_col.shape[-1])
    
    s_row_gathered = torch.gather(s_row, 2, topm_row_indices_expanded)  # [B, H, M, N_r]
    s_col_gathered = torch.gather(s_col, 2, topm_col_indices_expanded)  # [B, H, M, N_c]

    # 4. Compute the full S_grid (using gathered scores)
    s_grid = torch.einsum('bhrm,bhcm->bhrc', s_row_gathered, s_col_gathered)  # [B, H, M, M]
    s_grid_flat = s_grid.reshape(b, h, -1)  # [B, H, M*M]

    # 5. Find top-k values and indices within the flattened S_grid
    actual_k = min(k, actual_m * actual_m)  # Ensure k is not larger than m*m
    topk_vals, topk_indices_flat = torch.topk(s_grid_flat, actual_k, dim=-1)  # [B, H, K]

    # 6. Convert flat indices back to row/col indices (within the top-m)
    topk_row_indices = topm_row_indices.gather(2, (topk_indices_flat % actual_m).clamp(0, actual_m-1))   # [B, H, K]
    topk_col_indices = topm_col_indices.gather(2, (topk_indices_flat // actual_m).clamp(0, actual_m-1))   # [B, H, K]

    # 7. Gather original scores
    topk_scores = torch.gather(s_grid_flat, 2, topk_indices_flat) # Grab topk scores

    # Combine indices into a single tensor
    topk_indices = torch.stack([topk_row_indices, topk_col_indices], dim=-1)  # [B, H, K, 2]

    return topk_indices, topk_scores

def ive_projection_index(i, s_grid_out=None, E=None): # deterministic
    return (i % E) + 1

class PEERUltraMem(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.config = config
        self.n_v = config['n_v']
        self.d_v = config['d_v']
        self.d_k = config['d_k']
        self.h = config['h']
        self.r = config['r']
        self.E = config['E']
        self.top_m = config['top_m']
        self.top_k = config['top_k']
        self.tucker_rank_one_approx = config['tucker_rank_one_approx']
        self.aux_loss_weight = config['aux_loss_weight']
        self.aux_loss_margin = config['aux_loss_margin']
        self.use_expert_bias = config['use_expert_bias']
        self.d_model = config['n_embd']  # consistent with rest of code

        # Add factorization parameters
        self.factor_dim = 64  # New hyperparameter
        self.n_factors = 4    # Number of composable factors per slot
        
        # Modified physical memory with factorized storage
        self.V_physical = nn.Parameter(torch.empty(
            config['n_v'], 
            self.n_factors * self.factor_dim
        ))
        
        # Dynamic composition network
        self.composition_net = nn.Sequential(
            nn.Linear(self.factor_dim * self.n_factors, 4*config['d_v']),
            nn.GELU(),
            nn.Linear(4*config['d_v'], config['d_v']**2)
        )

        # Calculate n_r and n_c based on n_v
        
        self.n_r = int(math.sqrt(self.n_v))
        self.n_c = int(math.sqrt(self.n_v))
        # Adjust n_r and n_c if needed
        if self.n_r * self.n_c != self.n_v:
            self.n_r = int(math.floor(math.sqrt(self.n_v)))
            self.n_c = self.n_v // self.n_r
            assert self.n_r * self.n_c == self.n_v

        # 1. Physical Memory
        self.V_physical = nn.Parameter(torch.empty(self.n_v, self.d_v))
        nn.init.normal_(self.V_physical, mean=0.0, std=math.sqrt(2 / (self.n_v * self.d_v)))

        # 2. IVE Projection Matrices
        self.W_p_u = nn.Parameter(torch.empty(self.E, self.d_model, self.d_v))
        nn.init.xavier_uniform_(self.W_p_u, gain=nn.init.calculate_gain('linear')) # Good practice

        self.W_p_v = nn.Parameter(torch.empty(self.E, self.d_v, self.d_v))
        nn.init.xavier_uniform_(self.W_p_v, gain=nn.init.calculate_gain('linear')) # Good practice

        self.W_out = nn.Parameter(torch.empty(self.d_v, self.d_model))
        nn.init.xavier_uniform_(self.W_out, gain=nn.init.calculate_gain('linear')) # Good practice

        if self.use_expert_bias:
            self.W_p_b = nn.Parameter(torch.empty(self.E, self.d_v))
            nn.init.xavier_uniform_(self.W_p_b, gain=nn.init.calculate_gain('relu'))  # Corrected initialization

        # 3. Row and Column Keys
        self.K_row = nn.Parameter(torch.empty(self.n_r, self.d_k)) # Use self.n_r
        nn.init.xavier_uniform_(self.K_row, gain=nn.init.calculate_gain('linear'))

        self.K_col = nn.Parameter(torch.empty(self.n_c, self.d_k)) # Use self.n_c
        nn.init.xavier_uniform_(self.K_col, gain=nn.init.calculate_gain('linear'))

        # 4. Query Networks (Initialized within the Sequential)
        self.q_row = nn.Sequential(
            nn.Linear(self.d_model, self.d_k),
            nn.LayerNorm(self.d_k)  # Add LayerNorm, per UltraMem
        )
        self.q_col = nn.Sequential(
            nn.Linear(self.d_model, self.d_k),
            nn.LayerNorm(self.d_k)  # Add LayerNorm, per UltraMem
        )

        # 5. Tucker Core
        self.tucker_core = nn.Parameter(torch.empty(self.h, self.r, self.d_k, self.d_k))
        nn.init.xavier_uniform_(self.tucker_core, gain=nn.init.calculate_gain('linear'))

        
        if self.tucker_rank_one_approx:
            # Corrected u and t dimensions to match n_r and n_c
            self.u = nn.Parameter(torch.empty(self.h, self.n_r, self.d_k))
            nn.init.xavier_uniform_(self.u, gain=nn.init.calculate_gain('linear'))
            self.t = nn.Parameter(torch.empty(self.h, self.n_c, self.d_k))
            nn.init.xavier_uniform_(self.t, gain=nn.init.calculate_gain('linear'))
        else:
            # Original Tucker core (if needed)
            self.tucker_core = nn.Parameter(torch.empty(self.h, self.r, self.d_k, self.d_k))
            nn.init.xavier_uniform_(self.tucker_core, gain=nn.init.calculate_gain('linear'))

        # Auxiliary loss
        self.aux_loss = None

        self.dropout = nn.Dropout(config["dropout"])  # add dropout

    def calculate_aux_loss(self):
        # Placeholder for auxiliary loss calculation
        # Implement according to your specific requirements
        return torch.tensor(0.0, device=self.W_p_u.device)

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time (sequence length), Channels (embedding dim)

        # Retrieve factorized components
        factors = self.V_physical[memory_indices_flat]  # [B*H*K, n_factors*factor_dim]
        factors = factors.view(-1, self.n_factors, self.factor_dim)
        
        # Dynamic parameter construction
        composed_params = torch.einsum('bfd,dfg->bdg', 
            factors,
            self.composition_net(factors.mean(1)))

        # 1. Query Generation
        q_row = self.q_row(x)  # [B, T, D_k]
        q_col = self.q_col(x)  # [B, T, D_k]

        # 2. Row/Column Score Calculation
        s_row = torch.einsum('btd,rd->btr', q_row, self.K_row)  # [B, T, N_r]
        s_col = torch.einsum('btd,rd->btr', q_col, self.K_col)  # [B, T, N_c]

        # Reshape for heads and rank-1 tucker (if used)
        s_row = s_row.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B, H, T, N_r]
        s_col = s_col.unsqueeze(1).expand(-1, self.h, -1, -1)  # [B, H, T, N_c]

        if self.tucker_rank_one_approx:
            # Use the rank-1 approximation
            # The issue is here - we need to properly handle the ranks and dimensions
            topk_indices, topk_scores = topk_approx(s_row, s_col, self.u, self.t, 
                                                  min(self.top_m, min(self.n_r, self.n_c)), 
                                                  min(self.top_k, min(self.n_r, self.n_c))) # [B, h, k, 2]
        else:
            # Full Tucker decomposition (less efficient, for comparison)
            s_grid = torch.einsum('bhtr,bhtc,hrdc->bhtrc', s_row, s_col, self.tucker_core) # [B, H, T, N_r, N_c]
            s_grid = s_grid.reshape(B, self.h, T, -1)  # [B, H, T, N_r * N_c]
            
            # Ensure top_k is not larger than the available grid size
            actual_k = min(self.top_k, self.n_r * self.n_c)
            topk_vals, topk_indices = torch.topk(s_grid, actual_k, dim=-1)  # [B, H, T, K]

            # Convert flat indices to row/col safely
            topk_row_indices = (topk_indices // self.n_c).clamp(0, self.n_r-1)
            topk_col_indices = (topk_indices % self.n_c).clamp(0, self.n_c-1)
            topk_indices = torch.stack([topk_row_indices, topk_col_indices], dim=-1) #[B,h,t,k,2]
            topk_scores = topk_vals

        # 3. Retrieve Physical Values
        # Get the memory indices.  Shape: [B, H, K]
        topk_row_indices = topk_indices[:, :, :, 0]  # [B, H, K]
        topk_col_indices = topk_indices[:, :, :, 1]  # [B, H, K]
        
        # This is the critical fix - ensure indices are valid before calculating memory position
        topk_row_indices = topk_row_indices.clamp(0, self.n_r-1)
        topk_col_indices = topk_col_indices.clamp(0, self.n_c-1)
        
        memory_indices = (topk_row_indices * self.n_c) + topk_col_indices  # [B, H, K]
        memory_indices = memory_indices.clamp(0, self.n_v-1)  # Ensure we don't go out of bounds

        # Reshape memory_indices for gather - fix the reshape to maintain batch dimensions
        B, H, K = memory_indices.shape
        memory_indices_flat = memory_indices.reshape(-1)  # [B*H*K]
        
        # Fix the gather operation to handle the flattened indices properly
        V_retrieved = self.V_physical[memory_indices_flat]  # [B*H*K, D_v]
        V_retrieved = V_retrieved.view(B, H, K, self.d_v)  # [B, H, K, D_v]

        # 4. Generate u_i, v_i, and b_i (and expert calculation)
        outputs = []
        for h_idx in range(self.h): #Iterate Head
            head_outputs = []
            for k_idx in range(self.top_k): #Iterate topk
                # Skip invalid indices (shouldn't happen with our fixes, but added as safety)
                if k_idx >= K:
                    continue
                    
                # Select projection index (p)
                p = ive_projection_index(k_idx, E=self.E)  # Use helper function
                p_index = p - 1 # adjust to be 0 indexed

                # Generate u_i, v_i, and b_i
                u_i = torch.einsum('cd,bd->bc', self.W_p_u[p_index], V_retrieved[:, h_idx, k_idx, :])  # [B, C]
                v_i_prime = torch.einsum('vd,bd->bv',self.W_p_v[p_index], V_retrieved[:, h_idx, k_idx, :]) # [B, d_v]
                v_i = torch.einsum('bv,vd->bd', v_i_prime, self.W_out)  # [B, d_model]

                if self.use_expert_bias:
                    b_i = torch.einsum('d,bd->b', self.W_p_b[p_index], V_retrieved[:, h_idx, k_idx, :]) # [B]
                    b_i = b_i.unsqueeze(-1) #[B, 1]
                else:
                    b_i = 0  # No bias

                # Expert calculation
                e_i = u_i.unsqueeze(1) * x  # [B, T, D] Element-wise multiply with x
                e_i = e_i + b_i.unsqueeze(-1)  # Add bias and expand. [B, T, D]
                e_i = F.gelu(e_i)  # Apply activation
                e_i = e_i * v_i.unsqueeze(1) #[B, T, D]
                head_outputs.append(e_i)

            # Only use valid outputs
            if head_outputs:
                head_outputs = torch.stack(head_outputs, dim=2)  # [B, T, K, D]
                outputs.append(head_outputs)
            else:
                # Create a dummy tensor of zeros if no valid outputs
                dummy = torch.zeros((B, T, 1, C), device=x.device)
                outputs.append(dummy)
                
        outputs = torch.stack(outputs, dim=1) # [B, H, T, K, D]

        # 5. Weighted Sum Aggregation (no softmax)
        # Ensure scores have the right shape for einsum
        if topk_scores.dim() == 3:  # [B, H, K]
            topk_scores = topk_scores.sigmoid()  # Apply sigmoid, per UltraMem
            topk_scores = topk_scores.unsqueeze(2)  # [B, H, 1, K]
        else:  # For the case of [B, H, T, K]
            topk_scores = topk_scores.sigmoid()  # Apply sigmoid, per UltraMem
            topk_scores = topk_scores.mean(dim=2, keepdim=True)  # Average across T. [B, H, 1, K]        

        # Ensure topk_scores and outputs have compatible shapes for einsum
        K_actual = outputs.size(3)
        if topk_scores.size(3) != K_actual:
            topk_scores = topk_scores[:, :, :, :K_actual]
            
        output = torch.einsum('bhtk,bhtkd->btd', topk_scores.expand(-1, -1, T, -1), outputs) # [B, T, D]

        # Average across heads
        # output = output.mean(dim=1) # Mean across heads [B, T, D]

        output = self.dropout(output) # Apply dropout

        # 6. Auxiliary loss
        if self.training and self.tucker_rank_one_approx:
           self.aux_loss = self.calculate_aux_loss()
        else:
           self.aux_loss = None

        return output

