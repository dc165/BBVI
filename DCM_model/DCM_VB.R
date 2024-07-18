
# Helper Functions
sigmoid = function(x){1/(1+exp(-x))}

bernoulli_sampler = function(n, p){as.integer(runif(n) < p)}

intToBin = function(t, d){
  if (t == 0){res = c(0)}
  res = c()
  while (t > 1){
    res = c(t %% 2, res)
    t = t %/% 2
  }
  if (t == 1){res = c(t, res)}
  if (length(res) < d){res = c(rep(0, d-length(res)), res)}
  return(paste(res, collapse=""))
}

my_lambda = function(xi){
  ans = rep(1/8, length(xi))
  ans[xi != 0] = (sigmoid(xi[xi != 0])-1/2)/(2*xi[xi != 0])
  ans
}

trace = function(M){sum(diag(M))}

my_lbeta = function(x){sum(lgamma(x))-lgamma(sum(x))}

generate_beta = function(d){
  if (d == 1){
    beta = c(runif(1,-1.15,-1.05))
  }else if (d == 2){
    beta = c(runif(1,-1.15,-1.05), runif(1, 2.9, 3.1))
  }else if (d == 4){
    beta = c(runif(1,-1.15,-1.05), runif(2, 1.45, 1.55), runif(1, 0.45, 0.55))
  }else if (d == 8){
    beta = c(runif(1,-1.15,-1.05), runif(2, 0.65, 0.75), runif(1, 0.55, 0.65), runif(1, 0.65, 0.75), runif(2, 0.55, 0.65), runif(1, 0.35, 0.45))
  }else if (d == 16){
    beta = c(runif(1, -2.02, -1.99), runif(2, 0.68, 0.72), runif(1, 0.23, 0.27), runif(1, 0.68, 0.72), runif(2, 0.23, 0.27), runif(1, 0.18, 0.22), runif(1, 0.68, 0.72), 
             runif(2, 0.23, 0.27), runif(1, 0.23, 0.27), runif(1, 0.68, 0.72), runif(2, 0.23, 0.27), runif(1, 0.13, 0.17))    
  }
  beta
}

generate_delta_matrix = function(Q, k){
  calculate_q = function(t, q){prod(sapply(1:length(q), function(i){ifelse(substr(t,i,i) == '1', q[i], 1)}))}
  calculate_delta = function(q){
    t = sapply(0:(2^k-1), function(i){intToBin(i, d = k)})
    Q_res = matrix(rep(sapply(t, function(x){calculate_q(x, q)}), 2^k), ncol = 2^k, byrow = T, dimnames = list(t,t))
    A_res = matrix(unlist(lapply(t, function(y){sapply(t, function(x){calculate_q(x, sapply(1:k, function(i){as.numeric(substr(y,i,i))}))})})), ncol = 2^k, byrow = T, dimnames = list(t,t))
    A_res*Q_res 
  }
  
  lapply(1:nrow(Q), function(i){calculate_delta(Q[i,])})
}

plot_ccr = function(res, data){
  k = dim(data$Q)[2]
  N = length(data$skill)
  post_pred = sapply(1:N, function(i)which.max(res$p_matrix[i,]))
  post_pred = sapply(post_pred, function(i){intToBin(i-1, d = k)})
  res = c()
  cnames = c()
  for (i in 1:k){
    res = c(res, sum(sapply(1:N, function(n) substr(data$skill[n],i,i) == substr(post_pred[n],i,i)))/N)
    cnames = c(cnames, paste("CCR", as.character(i)))
  }
  df = data.frame(c(res, sum(data$skill == post_pred)/N), c(cnames, 'CCRAll'))
  colnames(df) = c("Ratio", "names")
  ggplot(df, aes(x = names,  y =Ratio))+
    geom_point()+
    theme(axis.title.x=element_blank(),
          axis.ticks.x=element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank())
}

plot_ELBO = function(res, data){
  temp = data.frame(1:length(res$ELBO), res$ELBO)
  colnames(temp) = c("Iteration", "ELBO")
  ggplot(temp, aes(x = Iteration, y = ELBO))+
    geom_point()+
    geom_line()+
    theme(axis.ticks.x=element_blank(),
          axis.ticks.y=element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank()
    )
}

label_beta = function(d, k){
  res = c()
  for (i in 1:d){
    res = c(res, paste0(sum(as.integer(unlist(strsplit(intToBin(i-1,k),"")))), "-way interaction"))
  }
  res[res == "0-way interaction"] = "Intercept"
  res[res == "1-way interaction"] = "Main effect"
  res                       
}

# Data Simulation
data_generate = function(N, k, J, seed = 1){
  set.seed(seed)
  col_name = sapply(0:(2^k-1), function(i){intToBin(i, d = k)})
  prob = rep(1/2^k, 2^k) #1:2^k/sum(2^k)
  skills = sapply(sample(0:(2^k-1), N, replace = T, prob = prob), function(x){intToBin(x, k)})
  
  Q_matrix = sapply(sample(0:(2^k-1), J, replace = T), function(x){intToBin(x, k)})
  Q_matrix = t(sapply(Q_matrix, function(x){sapply(1:k, function(i){as.numeric(substr(x,i,i))})}))
  rownames(Q_matrix) = 1:J
  beta = lapply(1:J, function(j) generate_beta(2^sum(Q_matrix[j,])))
  delta_matrix = generate_delta_matrix(Q_matrix, k)
  
  valid_cols = lapply(1:J, function(j) names(which(colSums(delta_matrix[[j]])>0)))
  
  for (j in 1:J){
    delta_matrix[[j]] = matrix(delta_matrix[[j]][,valid_cols[[j]]], nrow = 2^k, dimnames = c(list(rownames(delta_matrix[[1]])), list(valid_cols[[j]])))
  }
  
  simulate_once = function(j, beta, delta_matrix){
    samples = bernoulli_sampler(length(skills), as.vector(sigmoid(delta_matrix[[j]][skills,]%*%beta[[j]])))
    return(samples)
  }
  
  Y = sapply(1:J, function(j) bernoulli_sampler(length(skills), as.vector(sigmoid(matrix(delta_matrix[[j]][skills,], nrow = N)%*%beta[[j]]))))
  return (list('Y' = Y, 'Q' = Q_matrix, 'skill' = skills, 'beta' = beta, 'delta_matrix' = delta_matrix))
}


# Main Algorithm
JJVI_fit = function(data, max_iter = 500, tolerance = 1e-10){
  N = dim(data$Y)[1]
  J = dim(data$Y)[2]
  k = dim(data$Q)[2]
  L = 2^k
  dim_beta = sapply(1:J, function(j) length(data$beta[[j]]))
  
  a0 = rep(1e-2, J)
  b0 = rep(1e-4, J)
  d0 = rep(1, L)
  
  xi = matrix(rep(0, J*L), ncol = L, byrow = T)
  
  max_iter = 500
  E_Z = matrix(rep(1/L, L*N), ncol = L)
  
  # First Iteration
  inv_V = lapply(1:J, function(j) 2*t(data$delta_matrix[[j]])%*%diag(my_lambda(xi[j,])*colSums(E_Z))%*%data$delta_matrix[[j]]+a0[j]/b0[j]*diag(dim_beta[j]))
  V = lapply(inv_V, solve)
  mu = lapply(1:J, function(j)V[[j]]%*%t(data$delta_matrix[[j]])%*%t(E_Z)%*%(data$Y[,j]-1/2))
  E_tbeta_beta = sapply(1:J, function(j) t(mu[[j]])%*%mu[[j]]+trace(V[[j]]))
  a = a0 + dim_beta/2
  b = b0 + E_tbeta_beta/2
  d = d0 + colSums(E_Z)
  
  calculate_ELBO = function(a,b,d,E_Z,mu,V){
    term1 = sum(colSums(E_Z)*(rowSums(sapply(1:J, function(j) -diag(data$delta_matrix[[j]]%*%(mu[[j]]%*%t(mu[[j]])+V[[j]])%*%t(data$delta_matrix[[j]]))*my_lambda(xi[j,])))+colSums(apply(xi, c(1,2), function(x){log(sigmoid(x))-x/2+my_lambda(x)*x^2}))))+sum(sapply(1:J, function(j) t(mu[[j]])%*%t(data$delta_matrix[[j]])%*%t(E_Z)%*%(data$Y[,j]-1/2)))
    term2 = sum(sapply(1:N, function(i) sum(sapply(1:L, function(l) E_Z[i,l]*(digamma(d[l])-digamma(sum(d))-log(E_Z[i,l]))))))
    term3 = sum((d0-d)*(digamma(d)-digamma(sum(d))))+my_lbeta(d)-my_lbeta(d0)
    term4 = -1/2*sum(sapply(1:J, function(j){(t(mu[[j]])%*%mu[[j]]+trace(V[[j]]))*a[j]/b[j]}))
    term5 = 1/2*sum(sapply(V, function(M){log(abs(det(M)))}))
    return (term1+term2+term3+term4+term5+sum(lgamma(a))+sum(a*(1-b0/b))-sum(a*log(b)))
  }
  
  ELBO_arr = c(calculate_ELBO(a,b,d,E_Z,mu,V))
  
  for (iter_ in 1:max_iter){
    E_beta_tbeta = lapply(1:J, function(j){V[[j]]+mu[[j]]%*%t(mu[[j]])})  
    xi = t(sapply(1:J, function(j) sqrt(diag(data$delta_matrix[[j]]%*%E_beta_tbeta[[j]]%*%t(data$delta_matrix[[j]])))))
    
    # Update Q_1
    E_log_pi = digamma(d)-digamma(sum(d))
    E_Z = matrix(rep(colSums(apply(xi, c(1,2), function(x){log(sigmoid(x))-x/2+my_lambda(x)*x^2}))+ E_log_pi + rowSums(sapply(1:J, function(j) -diag(data$delta_matrix[[j]]%*%(mu[[j]]%*%t(mu[[j]])+V[[j]])%*%t(data$delta_matrix[[j]]))*my_lambda(xi[j,]))) , N), nrow = N, byrow=T)+Reduce("+", lapply(1:J, function(j) (data$Y[,j]-1/2)%*%t(mu[[j]])%*%t(data$delta_matrix[[j]])))
    
    E_Z = t(apply(E_Z, 1, function(x)exp(x-max(x))/sum(exp(x-max(x)))))
    
    # Update Q_2
    inv_V = lapply(1:J, function(j) 2*t(data$delta_matrix[[j]])%*%diag(my_lambda(xi[j,])*colSums(E_Z))%*%data$delta_matrix[[j]]+a[j]/b[j]*diag(dim_beta[j]))
    V = lapply(inv_V, solve)
    mu = lapply(1:J, function(j) V[[j]]%*%t(data$delta_matrix[[j]])%*%t(E_Z)%*%(data$Y[,j]-1/2))
    
    # Update Q_3
    d = d0 + colSums(E_Z)
    
    # Q4
    E_tbeta_beta = sapply(1:J, function(j){(t(mu[[j]])%*%mu[[j]])+trace(V[[j]])})
    b = b0 + E_tbeta_beta/2
    
    
    ELBO = calculate_ELBO(a,b,d,E_Z,mu,V)
    if ((ELBO > ELBO_arr[length(ELBO)]) &abs(ELBO-ELBO_arr[length(ELBO_arr)])<abs(tolerance*ELBO_arr[length(ELBO)])){
      break
    }else{ELBO_arr = c(ELBO_arr, ELBO)}
  }
  beta_sd = lapply(1:J, function(j)sqrt(diag(V[[j]])))
  
  res = list('beta_mu_post' = mu, 'beta_sd_post' = beta_sd, 'delta_matrix' = data$delta_matrix, 'ELBO' = ELBO_arr, 'p_matrix' = E_Z, 'xi' = xi, 'd' = d)
  
  p1 = plot_ELBO(res, data)
  df = data.frame(cbind(unlist(res$beta_mu_post), unlist(res$beta_sd_post), unlist(data$beta)), unlist(lapply(1:length(res$beta_mu_post), function(j) label_beta(length(res$beta_mu_post[[j]]), dim(data$Q)[2]))),
                  row.names = 1: length(unlist(data$beta)))
  colnames(df) = c("posterior mean", "posterior sd", "true value", "Type")
  df['lower'] = df$`posterior mean`- qnorm(0.975)*df$`posterior sd`
  df['upper'] = df$`posterior mean`+ qnorm(0.975)*df$`posterior sd`
  df['index'] = 1:nrow(df)
  p2 = ggplot(df, aes(x = index, y = `posterior mean`))+geom_errorbar(aes(ymin = lower, ymax = upper))+geom_point(aes(y = `true value`, col = Type))
  p3 = ggplot(df, aes(x =`true value`, y=`posterior mean`, col = Type))+geom_point()+geom_abline(slope = 1)
  p4 = plot_ccr(res, data)
  theme_set(theme_pubr())
  p_final = ggarrange(p1, p2, p3, p4 , ncol = 2, nrow = 2)
  ggsave("fig", plot = p_final, device = "eps", width = 20, height = 12, dpi = 320)
  res$`fig` = p_final
  res$`recovery rate` = sum((df$lower-df$`true value`)*(df$upper-df$`true value`) <= 0)/nrow(df)
  return(res)
}

if(1){
  library(ggplot2)
  library(ggpubr)
  data = data_generate(N = 1000, k = 2 , J = 30, seed = 1)
  res = JJVI_fit(data)
  # save(data$Q, file = "Q.csv")
  # save(data$Y, file = "Y.csv")
  save(data, file = "data.RData")
  save(res, file = "res.RData")
  res
}

































