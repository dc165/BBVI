# Data generation for the simulation study

source("utils.R")
library(torch)
# ---------------------------------------------------------------------------------------------------- #

data_generate = function(N_per_group = 100, K = 2, J = 20, C = 20, indT = 2, rand_N = F, rand_cor = 0, N_dataset = 1, beta_interact = TRUE, seed = NULL, Q_matrix = NULL){
  if (!is.null(seed)){torch_manual_seed(seed = seed)}
  
  NC = if(rand_N){torch_randint(low = N_per_group, high = 1.5*N_per_group, size = C)}else{torch_full(size = C, fill = N_per_group)}
  N = as.numeric(NC$sum())
  
  # Generate Q-Matrix
  if (is.null(Q_matrix)){
    Q_matrix = sapply(as_array(torch_randint(low = 1, high = 2^K, size = J)), function(x){intToBin(x, K)})
    Q_matrix = unname(t(sapply(Q_matrix, function(x){sapply(1:K, function(i){as.numeric(substr(x,i,i))})})))
  }
  
  # Generate Delta Matrix
  delta_matrix = generate_delta_matrix(Q_matrix, beta_interact)
  
  # Generate Beta
  beta = lapply(1:J, function(j) torch_tensor(generate_beta(2^sum(Q_matrix[j,]), beta_interact)))
  
  # Generate Group Covariates
  group = unlist(lapply(1:C, function(c){rep(c, as.numeric(NC[c]))}))
  group_mat = torch_sparse_coo_tensor(torch_vstack(list(torch_arange(start = 1, end = N), torch_tensor(group)))$to(dtype = torch_int32()), torch_ones(size = N))
  # X_group = torch_cat(list(torch_ones(size = C)$unsqueeze(2), torch_bernoulli(torch_ones(size = C)*0.5)$unsqueeze(2), (torch_rand(size = C)*4+1)$unsqueeze(2)), dim = 2)
  # X_group = torch_cat(list(torch_ones(size = C)$unsqueeze(2), torch_bernoulli(torch_ones(size = C)*0.5)$unsqueeze(2)), dim = 2)
  X_group = torch_cat(list(torch_ones(size = C)$unsqueeze(2), torch_bernoulli(torch_ones(size = C)*0.5)$unsqueeze(2), (torch_rand(size = C)*4+1)$unsqueeze(2)), dim = 2)
  # X_group = torch_cat(list(torch_ones(size = C)$unsqueeze(2), (torch_rand(size = C)*4+1)$unsqueeze(2)), dim = 2)
  X_group = lapply(1:K, function(k){lapply(1:indT, function(t) X_group)})
  
  # Generate Individual Covariates
  X_ind = torch_cat(list(torch_ones(size = N)$unsqueeze(2), torch_bernoulli(torch_ones(size = N)*0.5)$unsqueeze(2)), dim = 2)
  X_ind = lapply(1:K, function(k){c(torch_ones(N), lapply(2:indT, function(t) X_ind))})
  
  # Generate Omega
  # omega = lapply(1:K, function(k) c(torch_tensor(c(-0.5, 0.8, -0.06)), lapply(1:(indT-1), function(t){(torch_cat(list(torch_tensor(cbind(c(-2, 0.5, -0.1), c(4, 1, -0.2)))$unsqueeze(1), torch_tensor(cbind(c(-1.5, 0.5, -0.2), c(3.5, 0.75, -0.15)))$unsqueeze(1)), dim = 1))$permute(c(1,3,2))})))
  # omega = lapply(1:K, function(k) c(torch_tensor(c(-0.7, 0.8)), lapply(1:(indT-1), function(t){(torch_cat(list(torch_tensor(cbind(c(-2, 0.5), c(4, 1)))$unsqueeze(1), torch_tensor(cbind(c(-1, 0.5), c(3, 1)))$unsqueeze(1)), dim = 1))$permute(c(1,3,2))})))
  # omega = lapply(1:K, function(k) c(torch_tensor(c(-0.375, -0.15)), lapply(1:(indT-1), function(t){(torch_cat(list(torch_tensor(cbind(c(-2, -0.1), c(4.5, -0.1)))$unsqueeze(1), torch_tensor(cbind(c(-0.5, -0.05), c(4, -0.1)))$unsqueeze(1)), dim = 1))})))
  omega = lapply(1:K, function(k) c(torch_tensor(c(-0.5, 0.8, -0.06)), lapply(1:(indT-1), function(t){(torch_cat(list(torch_tensor(cbind(c(-2, 0.5, -0.1), c(4, 1, -0.2)))$unsqueeze(1), torch_tensor(cbind(c(-1.5, 0.5, -0.2), c(3.5, 0.75, -0.15)))$unsqueeze(1)), dim = 1))$permute(c(1,3,2))})))
  
  # Generate Gamma
  sigma = 0.2
  Sigma_gamma = lapply(1:K, function(k) c(torch_tensor(sigma**2), lapply(1:(indT-1), function(t){torch_cat(list(torch_tensor(c(sigma, sigma))$outer(torch_tensor(c(sigma, sigma)))*torch_tensor(matrix(c(1, rand_cor, rand_cor, 1), 2))$unsqueeze(1), torch_tensor(c(sigma, sigma))$outer(torch_tensor(c(sigma, sigma)))*torch_tensor(matrix(c(1, rand_cor, rand_cor, 1), 2))$unsqueeze(1)), dim = 1)})))
  
  generate_gamma = function(k, t){
    if (t == 1){
      return(X_group[[k]][[t]]$matmul(omega[[k]][[t]])+torch_randn(C)*sqrt(Sigma_gamma[[k]][[t]]))
    }
    else{
      return(X_group[[k]][[t]]$matmul(omega[[k]][[t]]$permute(c(1,3,2)))+torch_cat(list(distr_multivariate_normal(torch_zeros(2), Sigma_gamma[[k]][[t]][1,,])$sample(C)$unsqueeze(1), distr_multivariate_normal(torch_zeros(2), Sigma_gamma[[k]][[t]][2,,])$sample(C)$unsqueeze(1)), dim =1))
    }
  }
  
  gamma = lapply(1:K, function(k) lapply(1:indT, function(t) generate_gamma(k = k, t = t)))
  
  # Generate Profiles
  A = lapply(0:(2^(K*indT)-1), function(x){torch_tensor(matrix(as.integer(strsplit(intToBin(x, K*indT), "")[[1]]), ncol = indT, byrow = T))})
  A = torch_cat(lapply(A, function(x){x$unsqueeze(1)}), dim = 1) 
  
  prob_all = torch_zeros(N, 2^(K*indT))
  for (i in 1:A$shape[1]){
    profile = as_array(A[i,,])
    prob = torch_ones(N)
    for (k in 1:K){
      prob = prob*((X_ind[[k]][[1]]*group_mat$matmul(gamma[[k]][[1]]))*(2*profile[k,1]-1))$sigmoid()
      for (t in 2:indT){
        if (profile[k,t-1] == 0){
          gamma_tk = gamma[[k]][[t]][1,,]
        }
        else{
          gamma_tk = gamma[[k]][[t]][2,,]
        }
        prob = prob*(((X_ind[[k]][[t]]*group_mat$matmul(gamma_tk))$sum(2)*(2*profile[k,t]-1))$sigmoid())
      }
    }
    prob_all[,i] = prob
  }
  
  profiles_index = distr_categorical(prob_all)$sample()$squeeze() # index of profiles
  profiles = A[profiles_index,,]$float() # dim = (N, K, T)
  
  # Generate Response Matrix Y
  Y = torch_zeros(N_dataset, N, indT, J)
  for (t in 1:indT){
    for(j in 1:J){
      Y_sampler = distr_bernoulli(torch_tensor(delta_matrix[[j]][as.numeric(profiles[,,t]$matmul(torch_tensor(2^((K-1):0)))+1),])$matmul(beta[[j]])$sigmoid())
      Y[,,t,j] = Y_sampler$sample(N_dataset)
    }
  }
  
  # for (k in 1:K){
  #   for (t in 2:indT){
  #     omega[[k]][[t]] = omega[[k]][[t]]$permute(c(1,3,2))
  #   }
  # }
  
  if (N_dataset == 1){Y = Y[1,,,]}
  list('Y' = as_array(Y), 
       'X_group' = tensor_to_array(X_group), 
       'X_ind' = tensor_to_array(X_ind), 
       'group' = group, 
       'profiles' = as_array(profiles), 
       'profiles_index' = as.numeric(profiles_index), 
       'beta' = lapply(beta, function(x) as_array(x)), 
       'Q_matrix' = Q_matrix, 
       'omega' = tensor_to_array(omega), 
       'gamma' = tensor_to_array(gamma), 
       'Sigma_gamma' = tensor_to_array(Sigma_gamma),
       'K' = K,
       'indT' = indT,
       'rand_cor' = rand_cor,
       'folder_name' = paste("N", N_per_group, "K", K, "J", J, "C", C, "T", indT, "cor", rand_cor, sep = "_")
       )
}

# ---------------------------------------------------------------------------------------------------- #

write_data = function(data){
  # Define the main folder name
  data_folder = "simulate_data"
  output_folder_vb = "simulate_vb_output"
  output_folder_mcmc = "simulate_mcmc_output"
  
  for (folder in c(data_folder, output_folder_vb, output_folder_mcmc)){
    dir.create(file.path(folder, paste('K', data$K, sep = "_"), paste('T', data$indT, sep = "_"), paste("cor", data$rand_cor, sep = "_"), data$folder_name), recursive = T)
  }

  # Separate the data, and write them as .rds files
  if (length(dim(data$Y)) == 3){
    stop("Generate more than one dataset for the simulation study.")
  }
  else{
    for (i in 1:dim(data$Y)[1]){
      data_i = list('Y' = data$Y[i,,,], 
                    'X_group' = data$X_group, 
                    'X_ind' = data$X_ind, 
                    'group' = data$group, 
                    'profiles' = data$profiles, 
                    'profiles_index' = data$profiles_index, 
                    'beta' = data$beta, 
                    'Q_matrix' = data$Q_matrix, 
                    'omega' = data$omega, 
                    'gamma' = data$gamma, 
                    'Sigma_gamma' = data$Sigma_gamma)
      saveRDS(data_i, file = file.path(data_folder, paste('K', data$K, sep = "_"), paste('T', data$indT, sep = "_"), paste("cor", data$rand_cor, sep = "_"), data$folder_name, paste0("dataset_", i, ".rds")))
    }
  }
}


# Generate and write out data
if(FALSE){
  Ns = c(50, 100, 200, 500)
  Cs = c(25, 50, 100)
  Ks = c(3, 4)
  indTs = c(2, 3)
  cors = c(0, 0.2, 0.8)
  
  for (rand_cor in cors){
    for (N in Ns){
      for (C in Cs){
        for (K in Ks){
          
          if (K == 3){Q_matrix = as.matrix(read.table("Q_3.txt"))}
          else{Q_matrix = as.matrix(read.table("Q_4.txt"))}
          
          for (indT in indTs){
            J = ifelse(K == 3, 20, 25)
            data = data_generate(N_per_group = N, K = K, J = J, C = C, indT = indT, rand_N = F, rand_cor = rand_cor, N_dataset = 50, beta_interact = T, seed = 2025, Q_matrix = Q_matrix)
            write_data(data)
          }
        }
      }
    }
  }
}


# data = data_generate(N_per_group = 100, K = 3, J = 21, C = 50, indT = 2, Q_matrix = as.matrix(read.table("Q_3.txt")), rand_N = F, rand_cor = 0, N_dataset = 1, beta_interact = T, seed = 2024)

data = data_generate(N_per_group = 100, K = 2, J = 25, C = 50, indT = 2, rand_N = F, rand_cor = 0, N_dataset = 1, beta_interact = T, seed = 2024)

res = VB_Ind_fit(data, beta_interact = T, max_iter = 300)

plot(unlist(data$beta), unlist(res$E_beta))
abline(0,1)

# plot(unlist(data$beta), unlist(lapply(res$beta, function(x) as_array(x))))
# abline(0,1)

mean(data$profiles_index == res$profiles)

plot(unlist(data$gamma), unlist(res$E_gamma))
abline(0,1)

plot(unlist(data$gamma[[3]]), unlist(res$E_gamma[[3]]))
abline(0,1)

# plot(unlist(data$gamma[[1]][[2]][2,,1]), unlist(res$E_gamma[[1]][[2]][2,,1]))
# abline(0,1)

# plot(unlist(data$gamma), unlist(lapply(res$gamma, function(x){lapply(x, function(y)as_array(y))})))
# abline(0,1)

plot(unlist(data$omega), unlist(res$E_omega))
abline(0,1)

res$E_tau


summary(lm(data$gamma[[1]][[2]][1,,1] ~ data$X_group[[1]][[2]]+0))

summary(lm(res$E_gamma[[1]][[2]][1,,1] ~ data$X_group[[1]][[2]]+0))

summary(lm(data$gamma[[1]][[2]][1,,2] ~ data$X_group[[1]][[2]]+0))

summary(lm(res$E_gamma[[1]][[2]][2,,2] ~ data$X_group[[2]][[2]]+0))







