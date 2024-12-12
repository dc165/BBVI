# Utility functions for variational Bayesian estimation

require(torch)

# ---------------------------------------------------------------------------------------------------- #

# Jaakkola's function

JJ_func = function(xi){
  ans = (torch_sigmoid(xi)-1/2)/(2*xi)
  ans[torch_isnan(ans)] = 1/8
  ans 
}

# ---------------------------------------------------------------------------------------------------- #

# Batchwise outer product
torch_outer_batch = function(M, N = NULL){
  if (is.null(N)){N = M}
  torch_einsum("ab,ac->abc", list(M, N))
}

# ---------------------------------------------------------------------------------------------------- #

# Compute E_Z based on E_Z_K

E_Z_skill_to_all = function(E_Z_K, A_K) {
  N = E_Z_K$shape[1]
  K = E_Z_K$shape[2]
  indT = log(E_Z_K$shape[3], 2)
  E_Z = torch_ones(N, 2^(K*indT))
  
  for (k in 1:K) {
    E_Z = E_Z * E_Z_K[ , k, A_K[,k]$to(dtype = torch_int64())]
  }
  return(E_Z)
}


# ---------------------------------------------------------------------------------------------------- #

# Compute E_Z_T based on E_Z

E_Z_all_to_time = function(E_Z, A_T){
  indT = A_T$shape[2]
  K = log(A_T$shape[1], 2)/indT
  N = E_Z$shape[1]
  E_Z_T = torch_zeros(N, indT, 2^K)
  for (t in 1:indT){
    for (k in 1:(2^K)){
      E_Z_T[,t,k] = E_Z[,A_T[,t] == k]$sum(2)
    }
  }  
  return(E_Z_T)
}

# ---------------------------------------------------------------------------------------------------- #

# Compute E_Z_K based on E_Z

E_Z_all_to_skill = function(E_Z, A_K){
  K = A_K$shape[2]
  indT = log(A_K$shape[1], 2)/K
  N = E_Z$shape[1]
  E_Z_K = torch_zeros(N, K, 2^indT)
  for (k in 1:K){
    for (p in 1:(2^indT)){
      E_Z_K[,k,p] = E_Z[,A_K[,k] == p]$sum(2)
    }
  }  
  return(E_Z_K)
}


# ---------------------------------------------------------------------------------------------------- #





