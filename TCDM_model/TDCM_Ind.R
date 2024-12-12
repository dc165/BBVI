source("utils.R")
# source("utils_MCMC.R")
source("utils_VB.R")
require(cli)

# Variational Bayesian Algorithm for TDCM with independent individual-level random effects
VB_Ind_fit = function(data, beta_interact = T, max_iter = 100){
  print(paste("Torch is using", torch_get_num_threads(), "threads"))
  start = Sys.time()
  Y = torch_tensor(data$Y)
  N = Y$shape[1]
  indT = Y$shape[2]
  J = Y$shape[3]
  K = ncol(data$Q_matrix)
  delta_matrix = lapply(generate_delta_matrix(data$Q_matrix, beta_interact), function(x) torch_tensor(x)$float())
  group = data$group
  group_mat = torch_sparse_coo_tensor(torch_vstack(list(torch_arange(start = 1, end = N), torch_tensor(group)))$to(dtype = torch_int32()), torch_ones(size = N))
  X_group = array_to_tensor(data$X_group)
  X_ind = array_to_tensor(data$X_ind)
  C = X_group[[1]][[1]]$shape[1]
  if (beta_interact){
    beta_dim = 2^rowSums(data$Q_matrix)
  } else {
    beta_dim = rowSums(data$Q_matrix)+1
  }
  A = lapply(0:(2^(K*indT)-1), function(x){torch_tensor(matrix(as.integer(strsplit(intToBin(x, K*indT), "")[[1]]), ncol = indT, byrow = T))})
  A = torch_cat(lapply(A, function(x){x$unsqueeze(1)}), dim = 1)$float()
  A_T = transpose(A$permute(c(3,1,2))$matmul(torch_tensor(2^((K-1):0)))+1)
  A_K = transpose(A$permute(c(2,1,3))$matmul(torch_tensor(2^((indT-1):0)))+1)
  
  # Initialize Beta
  M_beta_init = lapply(beta_dim, function(x) torch_zeros(x))
  M_beta = lapply(beta_dim, function(x) torch_zeros(x))
  V_beta_init = lapply(beta_dim, function(x) torch_eye(x))
  V_beta = lapply(beta_dim, function(x) torch_eye(x))
  F_beta = torch_tensor(t(sapply(1:J, function(j){as_array(torch_matmul(delta_matrix[[j]], M_beta[[j]]))})))
  F2_beta = torch_tensor(t(sapply(1:J, function(j){as_array(torch_sum(torch_matmul(delta_matrix[[j]], torch_cholesky(V_beta[[j]]+torch_outer(M_beta[[j]], M_beta[[j]])))^2, dim = 2))})))
  
  # Initialize Gamma
  # M_gamma = lapply(1:K, function(k) c(torch_zeros(C, X_ind[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(2, C, X_ind[[k]][[t]]$shape[2]))))
  # M_gamma = lapply(1:K, function(k) c(-1*torch_ones(C, X_ind[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(2, C, X_ind[[k]][[t]]$shape[2]))))
  gamma_init_tmp = torch_zeros(2, C, X_ind[[1]][[2]]$shape[2])
  # gamma_init_tmp[1,,] = cbind(rep(-2, C), rep(4, C))
  # gamma_init_tmp[2,,] = cbind(rep(-1, C), rep(3, C))
  M_gamma = lapply(1:K, function(k) c(-0.5*torch_ones(C, X_ind[[k]][[1]]$shape[2]), lapply(2:indT, function(t) gamma_init_tmp)))
  
  V_gamma = lapply(1:K, function(k) c(torch_diag_embed(torch_ones(C,X_ind[[k]][[1]]$shape[2])), lapply(2:indT, function(t) torch_diag_embed(torch_ones(2, C, X_ind[[k]][[t]]$shape[2])))))
  E_gamma_tgamma = list(torch_zeros(N, K), torch_zeros(2, N, indT-1, K))
  for (k in 1:K){
    E_gamma_tgamma[[1]][,k] = group_mat$matmul(M_gamma[[k]][[1]]$squeeze()^2+V_gamma[[k]][[1]][,1,1])
    for (t in 2:indT){
      for (z in 1:2){
        E_gamma_tgamma[[2]][z,,t-1,k] = (torch_sum((X_ind[[k]][[t]]$matmul(torch_cholesky(torch_outer_batch(M_gamma[[k]][[t]][z,,])+V_gamma[[k]][[t]][z,,,])))^2, 3)*transpose(group_mat))$sum(1)$to_dense()
      }
    }
  }
  
  # Initialize Omega
  M_omega_init = lapply(1:K, function(k) c(torch_zeros(X_ind[[k]][[1]]$shape[2], X_group[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(2, X_ind[[k]][[t]]$shape[2], X_group[[k]][[t]]$shape[2]))))
  M_omega = lapply(1:K, function(k) c(torch_zeros(X_ind[[k]][[1]]$shape[2], X_group[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(2, X_ind[[k]][[t]]$shape[2], X_group[[k]][[t]]$shape[2]))))
  
  V_omega_init = lapply(1:K, function(k) c(torch_diag_embed(torch_ones(X_ind[[k]][[1]]$shape[2],X_group[[k]][[t]]$shape[2])), lapply(2:indT, function(t) torch_diag_embed(torch_ones(2, X_ind[[k]][[t]]$shape[2], X_group[[k]][[t]]$shape[2])))))
  V_omega = lapply(1:K, function(k) c(torch_diag_embed(torch_ones(X_ind[[k]][[1]]$shape[2],X_group[[k]][[t]]$shape[2])), lapply(2:indT, function(t) torch_diag_embed(torch_ones(2, X_ind[[k]][[t]]$shape[2], X_group[[k]][[t]]$shape[2])))))
  
  # Initialize Z
  E_Z_K = torch_ones(N, K, 2^indT)/2^indT
  E_Z = E_Z_skill_to_all(E_Z_K = E_Z_K, A_K = A_K)
  E_Z_T = E_Z_all_to_time(E_Z = E_Z, A_T = A_T)
  
  # Initialize Tau
  sigma_tau_init = 1e-6
  a_tau_init = lapply(1:K, function(k){c(torch_ones(X_ind[[k]][[1]]$shape[2])/2*sigma_tau_init, lapply(2:indT, function(t){torch_ones(2, X_ind[[k]][[t]]$shape[2])/2*sigma_tau_init}))})
  a_tau = lapply(1:K, function(k){c(torch_ones(X_ind[[k]][[1]]$shape[2])/2*sigma_tau_init, lapply(2:indT, function(t){torch_ones(2, X_ind[[k]][[t]]$shape[2])/2*sigma_tau_init}))})
  b_tau_init = lapply(1:K, function(k){c(torch_ones(X_ind[[k]][[1]]$shape[2])/2*sigma_tau_init, lapply(2:indT, function(t){torch_ones(2, X_ind[[k]][[t]]$shape[2])/2*sigma_tau_init}))})
  b_tau = lapply(1:K, function(k){c(torch_ones(X_ind[[k]][[1]]$shape[2])/2*sigma_tau_init, lapply(2:indT, function(t){torch_ones(2, X_ind[[k]][[t]]$shape[2])/2*sigma_tau_init}))})
  E_tau_inv = lapply(1:K, function(k){lapply(1:indT, function(t){a_tau[[k]][[t]]/b_tau[[k]][[t]]})})
  
  # Initialize Auxiliary Variables
  xi = torch_zeros(J, 2^K)
  eta = list(torch_zeros(N, K), torch_zeros(2, N, indT-1, K))
  xi = torch_tensor(t(sapply(1:J, function(j) as_array(torch_sqrt(torch_diag(delta_matrix[[j]]$matmul(V_beta[[j]]+M_beta[[j]]$outer(M_beta[[j]]))$matmul(transpose(delta_matrix[[j]]))))))))
  for (k in 1:K){
    eta[[1]][,k] = torch_sqrt(group_mat$matmul(M_gamma[[k]][[1]]$squeeze()^2+V_gamma[[k]][[1]][,1,1]))
    for (t in 2:indT){
      for (z in 1:2){
        eta[[2]][z,,t-1,k] = torch_sqrt((torch_sum((X_ind[[k]][[t]]$matmul(torch_cholesky(torch_outer_batch(M_gamma[[k]][[t]][z,,])+V_gamma[[k]][[t]][z,,,])))^2, 3)*transpose(group_mat))$sum(1)$to_dense())
      }
    }
  }
  
  M_beta_trace = lapply(beta_dim, function(x) torch_zeros(max_iter, x))
  for (j in 1:J){M_beta_trace[[j]][1,] = M_beta[[j]]}
  M_gamma_trace = lapply(1:K, function(k) c(-2*torch_ones(max_iter, C, X_ind[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(max_iter, 2, C, X_ind[[k]][[t]]$shape[2]))))
  M_omega_trace = lapply(1:K, function(k) c(torch_zeros(max_iter, X_ind[[k]][[1]]$shape[2], X_group[[k]][[1]]$shape[2]), lapply(2:indT, function(t) torch_zeros(max_iter, 2, X_ind[[k]][[t]]$shape[2], X_group[[k]][[t]]$shape[2]))))
  E_tau_trace = lapply(1:K, function(k){c(torch_zeros(max_iter, X_ind[[k]][[1]]$shape[2]), lapply(2:indT, function(t){torch_zeros(max_iter, 2, X_ind[[k]][[t]]$shape[2])}))})
  
  cli_progress_bar("Running VB-EM Algorithm: ", total = max_iter)
  for (iter_ in 1:max_iter){
    # Update E_Z_K
    E_Z_K_log = torch_zeros(N, K, 2^indT)
    
    for (k in 1:K){
      for (p in 1:2^indT){
        profile_k = as.integer(strsplit(intToBin(p-1, indT), "")[[1]])
        prob = torch_zeros(N)
        prob = prob + eta[[1]][,k]$sigmoid()$log()-eta[[1]][,k]/2-JJ_func(eta[[1]][,k])*(E_gamma_tgamma[[1]][,k]-eta[[1]][,k]^2)+group_mat$matmul(M_gamma[[k]][[1]]$squeeze())*(profile_k[1]-0.5)
        for (t in 2:indT){
          if (profile_k[t-1]==0){
            prob = prob + eta[[2]][1,,t-1,k]$sigmoid()$log()-eta[[2]][1,,t-1,k]/2-JJ_func(eta[[2]][1,,t-1,k])*(E_gamma_tgamma[[2]][1,,t-1,k]-eta[[2]][1,,t-1,k]^2)+(group_mat$matmul(M_gamma[[k]][[t]][1,,])*X_ind[[k]][[t]])$sum(2)*(profile_k[t]-0.5)
          }
          else if (profile_k[t-1]==1){
            prob = prob + eta[[2]][2,,t-1,k]$sigmoid()$log()-eta[[2]][2,,t-1,k]/2-JJ_func(eta[[2]][2,,t-1,k])*(E_gamma_tgamma[[2]][2,,t-1,k]-eta[[2]][2,,t-1,k]^2)+(group_mat$matmul(M_gamma[[k]][[t]][2,,])*X_ind[[k]][[t]])$sum(2)*(profile_k[t]-0.5)
          }
        }
        E_Z_K_log[,k,p] = prob*1.0
      }
    }
    
    y_tilde = ((Y$unsqueeze(-1)-1/2)*F_beta+(xi$sigmoid()$log()-xi/2-JJ_func(xi)*(F2_beta-xi^2)))$sum(dim = 3)
    
    for(k in 1:K){
      temp_prob_1 = torch_zeros(N, 2^(K*indT))
      for (t in 1:indT){
        temp_prob_1 = temp_prob_1 + y_tilde[,t,as.numeric(A_T[,t])]
      }
      
      
      temp_prob_2 = torch_ones(N, 2^(K*indT))
      for (k1 in 1:K){
        if (k1 != k){
          temp_prob_2 = temp_prob_2*E_Z_K[,k1,as.numeric(A_K[,k1])]
        }
      }
      temp_prob = temp_prob_1*temp_prob_2
      for (p in 1:2^indT){
        E_Z_K_log[,k,p] = E_Z_K_log[,k,p]+temp_prob[,A_K[,k] == p]$sum(2)
      }
      E_Z_K[,k,] = nnf_softmax(E_Z_K_log[,k,], 2)
    }
    E_Z_K = nnf_softmax(E_Z_K_log, 3)
    E_Z = E_Z_skill_to_all(E_Z_K, A_K)
    E_Z_T = E_Z_all_to_time(E_Z, A_T)
    
    # Update Beta
    V_beta = lapply(1:J, function(j){torch_inverse(torch_inverse(V_beta_init[[j]])+2*transpose(delta_matrix[[j]])$matmul(torch_diag(E_Z_T$sum(c(1,2))*JJ_func(xi[j,])))$matmul(delta_matrix[[j]]))})
    M_beta = lapply(1:J, function(j){V_beta[[j]]$matmul(torch_inverse(V_beta_init[[j]])$matmul(M_beta_init[[j]])+transpose(delta_matrix[[j]])$matmul(torch_einsum("ntl,nt->l", list(E_Z_T, Y[,,j]-1/2))))})
    F_beta = torch_tensor(t(sapply(1:J, function(j){as_array(torch_matmul(delta_matrix[[j]], M_beta[[j]]))})))
    F2_beta = torch_tensor(t(sapply(1:J, function(j){as_array(torch_sum(torch_matmul(delta_matrix[[j]], torch_cholesky(V_beta[[j]]+torch_outer(M_beta[[j]], M_beta[[j]])))^2, dim = 2))})))
    for (j in 1:J){M_beta_trace[[j]][iter_] = M_beta[[j]]}
    
    # Update Gamma
    for (k in 1:K){
      tmp = sapply(1:2^indT-1, function(x){as.integer(strsplit(intToBin(x, indT), "")[[1]])})
      V_gamma[[k]][[1]][,1,1] = 1/(a_tau[[k]][[1]]/b_tau[[k]][[1]]+2*transpose(group_mat)$matmul(JJ_func(eta[[1]][,k])))
      M_gamma[[k]][[1]][,1] = V_gamma[[k]][[1]][,1,1]*((a_tau[[k]][[1]]/b_tau[[k]][[1]]*X_group[[k]][[1]]$matmul(transpose(M_omega[[k]][[1]])))$squeeze()+transpose(group_mat)$matmul(E_Z_K[,k,])$matmul(torch_tensor(tmp[1,]-1/2)))
      M_gamma_trace[[k]][[1]][iter_,,1] = M_gamma[[k]][[1]][,1]
    }
    
    for(k in 1:K){
      for (t in 2:indT){
        for (z in 0:1){
          tmp = sapply(1:2^indT-1, function(x){as.integer(strsplit(intToBin(x, indT), "")[[1]])})
          V_gamma[[k]][[t]][z+1,,,] = torch_inverse(torch_diag(a_tau[[k]][[t]][z+1,]/b_tau[[k]][[t]][z+1,])+2*(transpose(group_mat)$matmul(torch_einsum("n,nmc->nmc", list(JJ_func(eta[[2]][z+1,,t-1,k])*E_Z_K[,k,]$matmul(torch_tensor(tmp[t-1,] == z)$float()),torch_outer_batch(X_ind[[k]][[t]])))$permute(c(3,1,2))))$permute(c(2,1,3)))
          M_gamma[[k]][[t]][z+1,,] = torch_einsum("nmc, nc->nm", list(V_gamma[[k]][[t]][z+1,,,], X_group[[k]][[t]]$matmul(transpose(torch_diag(a_tau[[k]][[t]][z+1,]/b_tau[[k]][[t]][z+1,])$matmul(M_omega[[k]][[t]][z+1,])))+transpose(group_mat)$matmul(torch_einsum("n,nm->nm", list(E_Z_K[,k,]$matmul(torch_tensor((tmp[t-1,] == z)*(tmp[t,]-1/2))), X_ind[[k]][[t]])))))
          M_gamma_trace[[k]][[t]][iter_,z+1,] = M_gamma[[k]][[t]][z+1,]
        }
      }
    }
    
    E_gamma_tgamma = list(torch_zeros(N, K), torch_zeros(2, N, indT-1, K))
    for (k in 1:K){
      E_gamma_tgamma[[1]][,k] = group_mat$matmul(M_gamma[[k]][[1]]$squeeze()^2+V_gamma[[k]][[1]][,1,1])
      for (t in 2:indT){
        for (z in 1:2){
          E_gamma_tgamma[[2]][z,,t-1,k] = (torch_sum((X_ind[[k]][[t]]$matmul(torch_cholesky(torch_outer_batch(M_gamma[[k]][[t]][z,,])+V_gamma[[k]][[t]][z,,,])))^2, 3)*transpose(group_mat))$sum(1)$to_dense()
        }
      }
    }
    
    # Update Omega
    for(k in 1:K){
      for (m in 1:X_ind[[k]][[1]]$shape[2]){
        V_omega[[k]][[1]][m,,] = torch_inverse(V_omega_init[[k]][[1]][m,,]$inverse()+a_tau[[k]][[1]][m]/b_tau[[k]][[1]][m]*transpose(X_group[[k]][[1]])$matmul(X_group[[k]][[1]]))
        M_omega[[k]][[1]][m,] = V_omega[[k]][[1]][m,,]$matmul(V_omega_init[[k]][[1]][m,,]$inverse()$matmul(M_omega_init[[k]][[1]][m,])+(a_tau[[k]][[1]][m]/b_tau[[k]][[1]][m])*transpose(X_group[[k]][[1]])$matmul(M_gamma[[k]][[1]][,m]))
        M_omega_trace[[k]][[1]][iter_,m,] = M_omega[[k]][[1]][m,]
      }
      
      for (m in 1:X_ind[[k]][[t]]$shape[2]){
        for(t in 2:indT){
          for(z in 1:2){
            V_omega[[k]][[t]][z,m,,] = torch_inverse(V_omega_init[[k]][[t]][z,m,,]$inverse()+a_tau[[k]][[t]][z,m]/b_tau[[k]][[t]][z,m]*transpose(X_group[[k]][[t]])$matmul(X_group[[k]][[t]]))
            M_omega[[k]][[t]][z,m,] = V_omega[[k]][[t]][z,m,,]$matmul(V_omega_init[[k]][[t]][z,m,,]$inverse()$matmul(M_omega_init[[k]][[t]][z,m,])+(a_tau[[k]][[t]][z,m]/b_tau[[k]][[t]][z,m])*transpose(X_group[[k]][[t]])$matmul(M_gamma[[k]][[t]][z,,m]))
            M_omega_trace[[k]][[t]][iter_,z,m,] = M_omega[[k]][[t]][z,m,]
          }
        }
      }
    }
    
    # Update Tau
    for (k in 1:K){
      for (m in 1: X_ind[[k]][[1]]$shape[2]){
        a_tau[[k]][[1]][m] = a_tau_init[[k]][[1]][m]+C/2
        b_tau[[k]][[1]][m] = b_tau_init[[k]][[1]][m]+(M_gamma[[k]][[1]][,m]^2+V_gamma[[k]][[1]][,m,m]-2*X_group[[k]][[1]]$matmul(M_omega[[k]][[1]][m,])*M_gamma[[k]][[1]][,m]+((X_group[[k]][[1]]$matmul(torch_cholesky(M_omega[[k]][[1]][m,]$outer(M_omega[[k]][[1]][m,])+V_omega[[k]][[1]][m,,])))^2)$sum(2))$sum()/2
      }
      E_tau_trace[[k]][[1]][iter_,] = (b_tau[[k]][[1]]/(a_tau[[k]][[1]]-1))
      
      for (t in 2:indT){
        for (z in 1:2){
          for (m in 1: X_ind[[k]][[t]]$shape[2]){
            a_tau[[k]][[t]][z,m] = a_tau_init[[k]][[t]][z,m]+C/2
            b_tau[[k]][[t]][z,m] = b_tau_init[[k]][[t]][z,m]+(M_gamma[[k]][[t]][z,,m]^2+V_gamma[[k]][[t]][z,,m,m]-2*X_group[[k]][[t]]$matmul(M_omega[[k]][[t]][z,m,])*M_gamma[[k]][[t]][z,,m]+((X_group[[k]][[t]]$matmul(torch_cholesky(M_omega[[k]][[t]][z,m,]$outer(M_omega[[k]][[t]][z,m,])+V_omega[[k]][[t]][z,m,,])))^2)$sum(2))$sum()/2
          }
          # print(b_tau[[k]][[t]]) ########
          E_tau_trace[[k]][[t]][iter_,,] = b_tau[[k]][[t]]/(a_tau[[k]][[t]]-1)
        }
      }
    }
    
    xi = F2_beta$sqrt()
    eta = list(E_gamma_tgamma[[1]]$sqrt(), E_gamma_tgamma[[2]]$sqrt())
    cli_progress_update()
  }
  cli_progress_done()
  
  # Compute the posterior mean and variance of tau
  E_tau = lapply(1:K, function(k){lapply(1:indT, function(t){(b_tau[[k]][[t]]/(a_tau[[k]][[t]]-1))})})
  V_tau = lapply(1:K, function(k){lapply(1:indT, function(t){(b_tau[[k]][[t]]^2/((a_tau[[k]][[t]]-2)*(a_tau[[k]][[t]]-1)^2))})})
  
  end = Sys.time()
  
  result_VB = list( # beta
    'E_beta' = lapply(M_beta, function(beta){as_array(beta)}),
    'V_beta' = lapply(V_beta, function(beta){as_array(beta)}),
    'E_beta_trace' = lapply(M_beta_trace, function(beta){as_array(beta)}),
    # gamma
    'E_gamma' = tensor_to_array(M_gamma), 
    'V_gamma' = tensor_to_array(V_gamma),
    'E_gamma_trace' = tensor_to_array(M_gamma_trace),
    # omega
    'E_omega' = tensor_to_array(M_omega), 
    'V_omega' = tensor_to_array(V_omega),
    'E_omega_trace' = tensor_to_array(M_omega_trace),
    # tau
    'E_tau' = tensor_to_array(E_tau), 
    'V_tau' = tensor_to_array(V_tau),
    'E_tau_trace' = tensor_to_array(E_tau_trace),
    # attribute profiles
    'profiles' = as_array(E_Z$max(2)[[2]]), 
    # run time
    'runtime' = as.numeric(difftime(end, start, units = "secs")))
  return (result_VB)
}


# MCMC Algorithm for TDCM with independent individual-level random effects
MCMC_Ind_fit = function(data, M = 500, priors=list(beta_prior = 1, sigma_omega = 2.5, v_prior = 1)){
  
  
  start = Sys.time()
  ## data preprocessing
  data = reformating_data(data)
  
  Ys = data$Ys
  Xs = data$Xs
  Q = data$Q
  Z = data$Z
  group = data$group
  
  #------------------ set up ------------------#
  indT = dim(Ys)[2] # number of time points
  N = dim(Ys)[1]  # number of respondents
  J = dim(Ys)[3]  # number of questions
  K = dim(Q)[2]  # number of skills
  G = dim(Z)[1]  # number of groups
  p = dim(Z)[2]  # number of group covariates
  m = dim(Xs[[2]])[2]  # number of individual covariates (including intercepts)
  Nq_total = indT*J  # total number of questions
  
  # alpha_mat_true = data$attribute_profiles_class
  # alpha_bin = alpha_int_to_bin(alpha_mat_true, indT, K)
  # beta_true = unlist(data$beta_true)
  # ml_params_true = list(gamma = data$gamma_true, omega = data$omega_true, Sigma = data$Sigma_true)
  
  Xbase = Xs[[2]]
  Nprofile = 2^K # the number of all possible attribute profiles
  Ntransition = 2^indT # the number of all possible transition types
  # delta_matrix = generate_delta_matrix(Q, K) # generate delta matrix
  delta = Q_to_delta(Q)
  
  profiles = map_dfr(1:Nprofile-1,
                     ~data.frame(t(rev(as.integer(intToBits(.))[1:K]))))
  A = model.matrix(data=profiles,
                   as.formula(paste0('~', paste0(names(profiles),collapse='*'))))
  # generate all transition types
  transitions = map_dfr(1:(2^indT)-1,
                        ~data.frame(t(rev(as.integer(intToBits(.))[1:indT]))))
  
  # transitions_each_time = map_dfr(1:(2^2)-1, ~data.frame(t(rev(as.integer(intToBits(.))[1:2]))))
  
  priorsd_beta = matrix(priors$beta_prior, Nq_total, Nprofile)
  
  
  
  ## forming respondent_group_matrix
  group_mat <- diag(1, nrow = G)
  respondents_group_mat <- matrix(NA, nrow = N, ncol = G)
  for (i in 1:N){
    respondents_group_mat[i, ] <- group_mat[group[i], ]
  }
  
  #------------------ initialization ------------------#
  
  
  ## step 1: beta
  beta_mat = matrix(rnorm(Nprofile*J), J, Nprofile)*delta
  beta_mat[ ,1] = -abs(beta_mat[ ,1])
  beta_mat[ ,2] = abs(beta_mat[ ,2])
  beta_mat[ ,3] = abs(beta_mat[ ,3])
  
  beta_vec = c(beta_mat[delta==1])   # vectorize beta by column
  
  ## step 2: transition parameters: gamma, omega, Sigma
  gamma = list()  # random individual-level effects
  omega = list()  # random group-level effects
  Sigma = list()  # random-effect covariance matrix
  
  # prior
  sigma_omega = priors$sigma_omega
  
  ml_priors = list(
    V_prior = diag(m),
    v_prior = 1,
    sigma_omega = sigma_omega,
    shape = 0.01,
    scale = 0.01
  )
  
  ml_priors_init = list(
    V_prior = diag(1),
    v_prior = 1,
    sigma_omega = sigma_omega,
    shape = 0.01,
    scale = 0.01
  )
  
  ###  Sigma
  for (t in 1:indT){
    Sigma[[t]] = list()
    if (t == 1){
      for (k in 1:K){
        # Sigma[[1]][[k]] = array(sapply(1:G, function(x) riwish(3, 1)), dim = c(G,1,1))
        Sigma[[1]][[k]] = riwish(v=3, S=ml_priors_init$V_prior)
      }
    } else {
      for (k in 1:K){
        Sigma[[t]][[k]] = list()
        for (i in 1:2) {
          Sigma[[t]][[k]][[i]] <- riwish(v=m + 1, S=ml_priors$V_prior)
        }
      }
    }
  }
  
  ###  omega and gamma
  for (t in 1:indT){
    gamma[[t]] = list()
    omega[[t]] = list()
    if (t == 1){
      for (k in 1:K){
        omega[[t]][[k]] = matrix(0, p, 1)
        gamma[[t]][[k]] = matrix(-2, G, 1)
      }
    } else {
      for (k in 1:K){
        omega[[t]][[k]] = list()
        gamma[[t]][[k]] = list()
        for (i in 1:2){
          omega[[t]][[k]][[i]] = matrix(0, p, m)
          gamma[[t]][[k]][[i]] = matrix(0, G, m)
        } # end of i
      } # end of k
    } # end of if
  } # end of t
  
  ml_params = list('gamma' = gamma, 'omega' = omega, 'Sigma' = Sigma)
  ml_params_vec = unlist(ml_params)
  gamma_vec = unlist(gamma)
  omega_vec = unlist(omega)
  Sigma_vec = unlist(Sigma)
  
  
  ## step 3: attribute profiles
  trans_probs = list()
  for (t in 1:indT){
    X = Xs[[t]]
    if (t==1){
      init_prob <- ml_params_to_transprobs_init(ml_params, X, respondents_group_mat, K, t)
      trans_probs[[1]] <- init_prob
    }else{
      trans_probs_temp <- ml_params_to_transprobs(ml_params, X, respondents_group_mat, K, t)
      trans_probs[[t]] <- trans_probs_temp
    }
  }
  
  update_transprob <- trans_probs_update(init_prob, trans_probs, transitions, N, indT, K)
  trans_mat <- matrix(NA, N, K)
  for(i in 1:N){
    for(j in 1:K){
      # trans_mat[i, j] = sample(1:(2^indT),1,prob = trans_probs[[j]][i, ])
      trans_mat[i, j] = sample(1:(2^indT),1,prob = update_transprob[[j]][i, ])
    }
  }
  
  alpha_mat <- transitions_to_alpha(trans_mat, transitions)  
  alpha_vec <- c(alpha_mat)
  
  beta_dim = 2^rowSums(Q)
  M_beta_trace = lapply(beta_dim, function(x) matrix(0, nrow = M, ncol = x))
  # M_gamma_trace = rep(list(list(array(0, dim = c(M, G, 1)), array(0, dim = c(M, 2, G, m)))), K)
  # M_omega_trace = rep(list(list(array(0, dim = c(M, 1, p)), array(0, dim = c(M, 2, m, p)))), K)
  # M_Sigma_trace = rep(list(list(array(0, dim = c(M, 1)), array(0, dim = c(M, 2, m)))), K)
  
  temp_gamma = vector("list", indT)
  temp_gamma[[1]] = array(0, dim = c(M, G, 1))
  temp_gamma[2:indT] = list(array(0, dim = c(M, 2, G, m)))
  M_gamma_trace = rep(list(temp_gamma), K)
  
  temp_omega = vector("list", indT)
  temp_omega[[1]] = array(0, dim = c(M, 1, p))
  temp_omega[2:indT] = list(array(0, dim = c(M, 2, m, p)))
  M_omega_trace = rep(list(temp_omega), K)
  
  temp_Sigma = vector("list", indT)
  temp_Sigma[[1]] = array(0, dim = c(M, 1))
  temp_Sigma[2:indT] = list(array(0, dim = c(M, 2, m)))
  M_Sigma_trace = rep(list(temp_Sigma), K)
  
  for (j in 1:J){
    beta_vec_temp = as.numeric(beta_mat[j, ])
    M_beta_trace[[j]][1, ] = beta_vec_temp[beta_vec_temp != 0]
  }
  
  for (k in 1:K){
    for (t in 1:indT){
      if (t==1){
        M_gamma_trace[[k]][[1]][1, , ] = gamma[[1]][[k]]  # time point 1
        M_omega_trace[[k]][[1]][1, , ] = t(omega[[1]][[k]])  # time point 1
        M_Sigma_trace[[k]][[1]][1, ] = diag(Sigma[[1]][[k]])  # time point 1
      }
      else{
        M_gamma_trace[[k]][[t]][1,1, ,] = gamma[[t]][[k]][[1]]  # time point t forward
        M_gamma_trace[[k]][[t]][1,2, ,] = gamma[[t]][[k]][[2]]  # time point t backward
        
        M_omega_trace[[k]][[t]][1,1, ,] = t(omega[[t]][[k]][[1]])  # time point t forward
        M_omega_trace[[k]][[t]][1,2, ,] = t(omega[[t]][[k]][[2]])  # time point t backward
        
        M_Sigma_trace[[k]][[t]][1,1, ] = diag(Sigma[[t]][[k]][[1]])  # time point t forward
        M_Sigma_trace[[k]][[t]][1,2, ] = diag(Sigma[[t]][[k]][[2]])  # time point t backward
      }
    }
  }
  
  
  #------------------ MCMC ------------------#
  ## set up
  varnames = c('beta','ml_params', 'alpha')
  sampnames = varnames%>%paste0('_vec')%>%
    sapply(function(x) paste(x,'[',1:length(get(x)),']',sep=''))%>%
    unlist()
  samples = matrix(NA, M, length(sampnames))%>%
    as.data.frame()%>%
    set_names(sampnames)
  samples[1, ]=c(beta_vec, ml_params_vec, alpha_vec)
  
  #Enforce positivity in all but intercept, as is done in the paper
  Ljp = matrix(0, J, Nprofile)
  Ljp[ ,1] = -5
  Ninteractions = ncol(delta)-ncol(Q) - 1
  Ljp[ ,(Nprofile-Ninteractions+1):Nprofile] = -5
  
  
  theta_list = list()
  
  cli_progress_bar("Running MCMC Algorithm: ", total = M)
  for (ite in 2:M){
    #print(ite)
    ltheta = beta_mat%*%t(A)
    theta = sigmoid(ltheta)
    
    ## step 1: attribute profiles
    
    # init_prob = ml_params_to_transprobs_init(ml_params, X = Xs[[1]], respondents_group_mat, K, t = 1)
    # trans_probs = ml_params_to_transprobs(ml_params, X = Xs[[2]], respondents_group_mat, K, t = 2)
    # theta[theta==1]=.99   # do this to avoid NA's:
    # update_transprob = trans_probs_update(init_prob, trans_probs, N, indT, K)
    # alpha_mat = sample_alpha_mat(alpha_mat, theta, update_transprob, profiles, Ys)
    # alpha_bin <- alpha_int_to_bin(alpha_mat, indT, K)  ## alpha_bin is the attribute profiles of each respondent at each time point
    # trans_mat <- alpha_to_transitions(alpha_bin) ## the transition matrix (K*N*T)
    # alpha_vec = c(alpha_mat)
    
    
    trans_probs = list()
    for (t in 1:indT){
      X = Xs[[t]]
      if (t==1){
        init_prob <- ml_params_to_transprobs_init(ml_params, X, respondents_group_mat, K, t)
        trans_probs[[1]] <- init_prob
      }else{
        trans_probs_temp <- ml_params_to_transprobs(ml_params, X, respondents_group_mat, K, t)
        trans_probs[[t]] <- trans_probs_temp
      }
    }
    theta[theta==1]=.99   # do this to avoid NA's:
    update_transprob <- trans_probs_update(init_prob, trans_probs, transitions, N, indT, K)
    alpha_mat <- sample_alpha_mat(alpha_mat, theta, update_transprob, profiles, Ys)
    alpha_bin <- alpha_int_to_bin(alpha_mat, indT, K)  ## alpha_bin is the attribute profiles of each respondent at each time point
    trans_mat <- alpha_to_transitions(alpha_bin) ## the transition matrix (K*N*T)
    alpha_vec = c(alpha_mat)
    
    ###-----------------------------------###
    
    
    ## step 2: beta
    
    # sample ystar
    nct = matrix(NA, Nprofile, indT)
    for(c in 1:Nprofile){
      for(t in 1:indT){
        nct[c,t] = sum(alpha_mat[,t]==c)
      }
    }
    ystar = sample_ystar_fb(ltheta, nct, indT) #*delta
    
    kappa = map(1:indT, ~matrix(NA, J, Nprofile))
    for(t in 1:indT){
      for(j in 1:J){
        for(c in 1:Nprofile){
          kappa[[t]][j,c] = sum(Ys[alpha_mat[,t]==c, t, j]) - nct[c,t]/2
        }
      }
    }
    z = map2(kappa,ystar,~.x/.y)
    
    beta_post = beta_gibbs_dist_fb(ystar,A,z,beta_mat,priorsd_beta, indT)
    beta_mat = matrix(rtruncnorm(length(beta_post$mean),mean=c(beta_post$mean),
                                 sd=c(beta_post$sd),a=c(Ljp)),
                      dim(beta_post$mean)[1],dim(beta_post$mean)[2])*delta
    # beta_vec = c(beta_mat[delta==1]) # only keep the non-zero elements, expand to a vector by column
    beta_vec = c(t(beta_mat)[t(delta)==1])
    
    for (j in 1:J){
      beta_vec_j = as.numeric(beta_mat[j, ])
      M_beta_trace[[j]][ite, ] = beta_vec_j[beta_vec_j != 0]
    }
    
    ###-----------------------------------###
    
    
    ## step 3: transition parameters: gamma and omega
    gamma_new = list()
    omega_new = list()
    Sigma_new = list()
    for (t in 1:indT){
      gamma_new[[t]] = list()
      omega_new[[t]] = list()
      Sigma_new[[t]] = list()
      X = as.matrix(Xs[[t]])
      if (t==1){
        for (k in 1:K){
          ind = seq(1,N)
          if (length(ind) > 0){
            X_forward = as.matrix(X[ind, ])
            epsilon_1 <- trans_mat[[k]][ ,1][ind]
            group_assignments <- group[ind]
            multi_level_samples_for <- sample_ml_pg(epsilon_1, X_forward, Z,
                                                    gamma = matrix(ml_params[[1]][[t]][[k]], ncol = 1), omega = ml_params[[2]][[t]][[k]], Sigma = as.numeric(ml_params[[3]][[t]][[k]]), group_assignments, priors = ml_priors_init)
            gamma_new[[t]][[k]] = multi_level_samples_for$gamma
            omega_new[[t]][[k]] = multi_level_samples_for$omega
            Sigma_new[[t]][[k]] = multi_level_samples_for$Sigma
          }
        }
      }else{
        ## for t > 1
        for (k in 1:K){
          ## subset for forward transitions
          ind = which(trans_mat[[k]][ ,t-1] == 0) 
          if (length(ind) > 0){
            X_forward = as.matrix(X[ind, ])
            epsilon_1 <- trans_mat[[k]][ ,t][ind] ## epsilon_1 = 1 indicates forward transition type
            group_assignments <- group[ind]
            multi_level_samples_for <- sample_ml_pg(epsilon_1, X_forward, Z,
                                                    gamma = ml_params[[1]][[t]][[k]][[1]], omega = ml_params[[2]][[t]][[k]][[1]], Sigma = ml_params[[3]][[t]][[k]][[1]], group_assignments, priors = ml_priors)
          }
          gamma_for = multi_level_samples_for$gamma
          omega_for = multi_level_samples_for$omega
          Sigma_for = multi_level_samples_for$Sigma
          
          
          ## subset for backward transitions
          ind = which(trans_mat[[k]][ ,t-1] == 1) 
          if (length(ind) > 0){
            X_backward = as.matrix(X[ind, ])
            epsilon_2 <- trans_mat[[k]][ ,t][ind] ## epsilon_2 = 1 indicates backward transition type
            group_assignments <- group[ind]
            multi_level_samples_back<- sample_ml_pg(epsilon_2, X_backward, Z,
                                                    gamma = ml_params[[1]][[t]][[k]][[2]], omega = ml_params[[2]][[t]][[k]][[2]], Sigma = ml_params[[3]][[t]][[k]][[2]], group_assignments, priors = ml_priors)
          }
          gamma_back = multi_level_samples_back$gamma
          omega_back = multi_level_samples_back$omega
          Sigma_back = multi_level_samples_back$Sigma
          
          
          ## combine forward and backward samples
          gamma_new[[t]][[k]] = list()
          omega_new[[t]][[k]] = list()
          Sigma_new[[t]][[k]] = list()
          
          gamma_new[[t]][[k]][[1]] = gamma_for
          gamma_new[[t]][[k]][[2]] = gamma_back
          omega_new[[t]][[k]][[1]] = omega_for
          omega_new[[t]][[k]][[2]] = omega_back
          Sigma_new[[t]][[k]][[1]] = Sigma_for
          Sigma_new[[t]][[k]][[2]] = Sigma_back
          
        } ## end of loop over skill k
        
      } ## end of if-else t>1 statement
      
    } ## end of loop over time t
    
    ml_params = list(gamma = gamma_new, omega = omega_new, Sigma = Sigma_new)
    ml_params_vec = unlist(ml_params)
    gamma_vec = unlist(gamma_new)
    omega_vec = unlist(omega_new)
    Sigma_vec = unlist(Sigma_new)
    
    
    for (k in 1:K){
      for (t in 1:indT){
        if (t==1){
          M_gamma_trace[[k]][[1]][ite, , ] = gamma_new[[1]][[k]]  # time point 1
          M_omega_trace[[k]][[1]][ite, , ] = t(omega_new[[1]][[k]])  # time point 1
          M_Sigma_trace[[k]][[1]][ite, ] = diag(Sigma_new[[1]][[k]])  # time point 1
        }
        else{
          M_gamma_trace[[k]][[t]][ite,1, ,] = gamma_new[[t]][[k]][[1]]  # time point t forward
          M_gamma_trace[[k]][[t]][ite,2, ,] = gamma_new[[t]][[k]][[2]]  # time point t backward
          
          M_omega_trace[[k]][[t]][ite,1, ,] = t(omega_new[[t]][[k]][[1]])  # time point t forward
          M_omega_trace[[k]][[t]][ite,2, ,] = t(omega_new[[t]][[k]][[2]])  # time point t backward
          
          M_Sigma_trace[[k]][[t]][ite,1, ] = diag(Sigma_new[[t]][[k]][[1]])  # time point t forward
          M_Sigma_trace[[k]][[t]][ite,2, ] = diag(Sigma_new[[t]][[k]][[2]])  # time point t backward
        }
      }
    }
    
    samples[ite, ] = c(beta_vec, ml_params_vec, alpha_vec)
    
    cli_progress_update()
  } ## end of MCMC loop 
  cli_progress_done()
  
  end = Sys.time()
  
  
  ## data analysis
  
  burn_in = floor(0.4 * M)
  
  E_beta_trace = lapply(1:J, function(x){M_beta_trace[[x]][burn_in:M, ]})
  E_beta = lapply(E_beta_trace, function(x) {apply(x, 2, mean)})
  V_beta = lapply(E_beta_trace, function(x) {apply(x, 2, var)})
  
  
  E_gamma_trace =  vector("list", K)
  for (k in 1:K){
    gamma_trace_k = vector("list", indT)
    for (t in 1:indT){
      if (t==1){
        gamma_temp = M_gamma_trace[[k]][[1]][burn_in:M, , ]
        gamma_trace_k[[t]] = gamma_temp
      }
      else{
        gamma_temp = M_gamma_trace[[k]][[t]][burn_in:M, , ,]
        gamma_trace_k[[t]] = gamma_temp
      }
    }
    E_gamma_trace[[k]] = gamma_trace_k
  }
  
  E_gamma = lapply(E_gamma_trace, function(x){
    lapply(x, function(M){
      apply(M, MARGIN = c(2:length(dim(M))), FUN = mean)
    })
  })
  
  V_gamma = lapply(E_gamma_trace, function(x){
    lapply(x, function(M){
      apply(M, MARGIN = c(2:length(dim(M))), FUN = var)
    })
  })
  
  
  E_omega_trace =  vector("list", K)
  for (k in 1:K){
    omega_trace_k = vector("list", indT)
    for (t in 1:indT){
      if (t==1){
        omega_temp = M_omega_trace[[k]][[1]][burn_in:M, , ]
        omega_trace_k[[t]] = omega_temp
      }
      else{
        omega_temp = M_omega_trace[[k]][[t]][burn_in:M, , ,]
        omega_trace_k[[t]] = omega_temp
      }
    }
    E_omega_trace[[k]] = omega_trace_k
  }
  
  E_omega = lapply(E_omega_trace, function(x){
    lapply(x, function(M){
      apply(M, MARGIN = c(2:length(dim(M))), FUN = mean)
    })
  })
  
  V_omega = lapply(E_omega_trace, function(x){
    lapply(x, function(M){
      apply(M, MARGIN = c(2:length(dim(M))), FUN = var)
    })
  })
  
  
  E_Sigma_trace =  vector("list", K)
  for (k in 1:K){
    Sigma_trace_k = vector("list", indT)
    for (t in 1:indT){
      if (t==1){
        Sigma_temp = M_Sigma_trace[[k]][[1]][burn_in:M, ]
        Sigma_trace_k[[t]] = Sigma_temp
      }
      else{
        Sigma_temp = M_Sigma_trace[[k]][[t]][burn_in:M, ,]
        Sigma_trace_k[[t]] = Sigma_temp
      }
    }
    E_Sigma_trace[[k]] = Sigma_trace_k
  }
  
  E_Sigma = lapply(E_Sigma_trace, function(x){
    lapply(x, function(M){
      if (is.null(dim(M))){
        mean(M)
      } else {
        apply(M, MARGIN = c(2:length(dim(M))), FUN = mean)
      }
    })
  })
  
  V_Sigma = lapply(E_Sigma_trace, function(x){
    lapply(x, function(M){
      if (is.null(dim(M))){
        mean(M)
      } else {
        apply(M, MARGIN = c(2:length(dim(M))), FUN = var)
      }
    })
  })
  
  
  result_MCMC = list( # beta
    'E_beta' = E_beta,
    'V_beta' = V_beta,
    'E_beta_trace' = E_beta_trace,
    # gamma
    'E_gamma' = E_gamma,
    'V_gamma' = V_gamma,
    'E_gamma_trace' = E_gamma_trace,
    # omega
    'E_omega' = E_omega, 
    'V_omega' = V_omega,
    'E_omega_trace' = E_omega_trace,
    # tau
    'E_tau' = E_Sigma, 
    'V_tau' = V_Sigma,
    'E_tau_trace' = E_Sigma_trace,
    # attribute profiles
    # 'profiles' = colMeans(samples[(burn_in+1):M, (n_Sigma+1):n_alpha]), 
    'profiles' = attribute_bin_to_int_all(alpha_bin),
    # run time
    'runtime' = as.numeric(end-start))
  return (result_MCMC)
}


run_func = function(path, MCMC = F){
  data = readRDS(path)
  if (MCMC){
    result_MCMC = MCMC_Ind_fit(data)
    saveRDS(result_MCMC, file = gsub("simulate_data", "simulate_mcmc_output", path))
  }else{
    result_VB = VB_Ind_fit(data)
    saveRDS(result_VB, file = gsub("simulate_data", "simulate_vb_output", path))
  }
}


if(FALSE){
  # Get command-line arguments
  args = commandArgs(trailingOnly = TRUE)
  
  # Run the function
  run_func(args, MCMC = FALSE)
  # run_func(args, MCMC = TRUE)
}



























