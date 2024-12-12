# Utility functions in common use

require(torch)

# ---------------------------------------------------------------------------------------------------- #

# Convert any integer to binary

intToBin = function(x, d){
  if (x == 0){res = c(0)}
  if (x >= 2^d){stop("x should be in [0, 2^d-1]")}
  res = c()
  while (x > 1){
    res = c(x %% 2, res)
    x = x %/% 2
  }
  if (x == 1){res = c(x, res)}
  if (length(res) < d){res = c(rep(0, d-length(res)), res)}
  return(paste(res, collapse=""))
}

# ---------------------------------------------------------------------------------------------------- #

# Transpose of a matrix

transpose = function(M){return(M$permute(c(2,1)))}

# ---------------------------------------------------------------------------------------------------- #

# Convert a nested list of tensors to a nested list of arrays

tensor_to_array = function(list_of_tensors){
  K = length(list_of_tensors)
  indT = length(list_of_tensors[[1]])
  for (k in 1:K){
    for (t in 1:indT){
      list_of_tensors[[k]][[t]] = as_array(list_of_tensors[[k]][[t]])
    }
  }
  list_of_tensors
}

# ---------------------------------------------------------------------------------------------------- #

# Convert a nested list of arrays to a nested list of tensors

array_to_tensor = function(list_of_arrays){
  K = length(list_of_arrays)
  indT = length(list_of_arrays[[1]])
  for (k in 1:K){
    for (t in 1:indT){
      list_of_arrays[[k]][[t]] = torch_tensor(list_of_arrays[[k]][[t]])
      if (length((dim(list_of_arrays[[k]][[t]]))) == 1){
        list_of_arrays[[k]][[t]] = list_of_arrays[[k]][[t]]$unsqueeze(-1)
      }
    }
  }
  list_of_arrays
}

# ---------------------------------------------------------------------------------------------------- #

# Generate the delta matrix (design matrix for item parameters)

generate_delta_matrix = function(Q_matrix, beta_interact = TRUE){
  K = ncol(Q_matrix)
  calculate_q = function(t, q){prod(sapply(1:length(q), function(i){ifelse(substr(t,i,i) == '1', q[i], 1)}))}
  calculate_delta = function(q){
    t = sapply(0:(2^K-1), function(i){intToBin(i, d = K)})
    Q = matrix(rep(sapply(t, function(x){calculate_q(x, q)}), 2^K), nrow = 2^K, byrow = T)
    A = matrix(unlist(lapply(t, function(y){sapply(t, function(x){calculate_q(x, sapply(1:K, function(i){as.numeric(substr(y,i,i))}))})})), ncol = 2^K, byrow = T, dimnames = list(t,t))
    res = A*Q
    if (beta_interact){
      return(res)
    }else{
      return(res[,sapply(strsplit(t, ""), function(x){sum(as.numeric(x))}) <= 1])
    }
  }
  
  J = nrow(Q_matrix)
  delta_matrix = lapply(1:J, function(j){calculate_delta(Q_matrix[j,])})
  valid_cols = lapply(1:J, function(j) names(which(colSums(delta_matrix[[j]])>0)))
  
  for (j in 1:J){
    delta_matrix[[j]] = matrix((delta_matrix[[j]][,valid_cols[[j]]]), nrow = 2^K, dimnames = c(list(rownames(delta_matrix[[1]])), list(valid_cols[[j]])))
  }
  return(delta_matrix)
}

# ---------------------------------------------------------------------------------------------------- #

# Generate the beta vector (item parameters), up to 4 attributes (K <= 4)

generate_beta = function(d, beta_interact = TRUE){
  if (d == 2){
    beta = c(-2.5, 6)
  }else if (d == 4){
    if(beta_interact){
      beta = c(-2.5, 2.5, 2.5, 1)
    }else{
      beta = c(-2.5, 3, 3)
    }
  }else if (d == 8){
    if(beta_interact){
      beta = c(-2.5, 1.2, 1.2, 0.7, 1.2, 0.7, 0.7, 0.3)
    }else{
      beta = c(-2.5, 2, 2, 2)
    }
  }else if (d == 16){
    if (beta_interact){
      beta = c(-3, 0.8, 0.8, 0.4, 0.8, 0.4, 0.4, 0.2, 0.8, 0.4, 0.4, 0.2, 0.4, 0.2, 0.2, 0.1)
    }else{
      beta = c(-2.5, 1.5, 1.5, 1.5, 1.5)
    }
  }
  else{
    stop("Invalid d")
  }
  beta
}

# ---------------------------------------------------------------------------------------------------- #

# Label the beta vector

label_beta = function(d, k){
  res = c()
  for (i in 0:(d-1)){
    res = c(res, paste0(sum(as.integer(unlist(strsplit(intToBin(i,k),"")))), "-way interaction"))
  }
  res[res == "1-way interaction"] = "Main effect"
  res[res == "0-way interaction"] = "Intercept"
  return(res)                       
}

# ---------------------------------------------------------------------------------------------------- #






















