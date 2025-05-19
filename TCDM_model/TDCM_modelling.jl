using LinearAlgebra, Distributions, Combinatorics, Random, Kronecker, SpecialFunctions
using ResumableFunctions
include("../DCM_model/BBVI_utils.jl")

# Object for observed data
struct TDCMObs{T <: AbstractFloat}
    # data
    Y           :: Array{Int, 3}
    Q           :: Matrix{Int}
    D           :: Vector{Matrix{Int}}
    U           :: Vector{Vector{Matrix{T}}}
    X           :: Vector{Vector{Matrix{T}}}
    group       :: Vector{Int}
    skill_dict  :: Dict{Int, Vector{Int}}
end

function TDCMObs(
    Y       :: Array{Int, 3}, 
    Q       :: Matrix{Int},
    U       :: Vector{Vector{Matrix{T}}},
    X       :: Vector{Vector{Matrix{T}}},
    group   :: Vector{Int}) where T <: AbstractFloat
    D = generate_delta(Q)
    K, L = size(Q, 2), size(D[1], 1)
    skill_dict = Dict{Int, Vector{Int}}()
    for l in 0:(L - 1)
        skill_dict[l + 1] = reverse(digits(l, base=2, pad=K))
    end
    TDCMObs(Y, Q, D, U, X, group, skill_dict)
end

# Object including latent variables and model parameters
struct TDCModel{T <: AbstractFloat}
    # data
    obs         :: TDCMObs
    # prior distribution parameters
    mu_beta_prior       :: Vector{Vector{T}}
    L_beta_prior        :: Vector{Matrix{T}} # Upper triangular Cholesky factor of prior inverse covariance matrix of betas
    mu_omega_prior      :: Vector{Vector{Vector{Vector{Vector{T}}}}}
    L_omega_prior       :: Vector{Vector{Vector{Vector{Matrix{T}}}}} # Upper triangular Cholesky factor of prior inverse covariance matrix of omegas
    a_tau_prior         :: Vector{Vector{Vector{Vector{T}}}}
    b_tau_prior         :: Vector{Vector{Vector{Vector{T}}}}
    # This option allocates extra memory based on number of threads availible in the environment
    enable_parallel     :: Bool
    # Variational distribution parameters
    pi_star             :: Vector{Vector{Vector{T}}}
    mu_beta_star        :: Vector{Vector{T}}
    V_beta_star         :: Vector{Matrix{T}}
    mu_gamma_star       :: Vector{Vector{Vector{Vector{Vector{T}}}}}
    V_gamma_star        :: Vector{Vector{Vector{Vector{Matrix{T}}}}}
    mu_omega_star       :: Vector{Vector{Vector{Vector{Vector{T}}}}}
    V_omega_star        :: Vector{Vector{Vector{Vector{Matrix{T}}}}}
    a_tau_star          :: Vector{Vector{Vector{Vector{T}}}}
    b_tau_star          :: Vector{Vector{Vector{Vector{T}}}}
    # Number of samples for noisy gradient
    M                   :: Int
    # Preallocated storage for samples from variational distribution
    Z_sample            :: Vector{Vector{Vector{Vector{Int}}}}
    beta_sample         :: Vector{Vector{Vector{T}}}
    gamma_sample        :: Vector{Vector{Vector{Vector{Vector{Vector{T}}}}}}
    omega_sample        :: Vector{Vector{Vector{Vector{Vector{Vector{T}}}}}}
    tau_sample          :: Vector{Vector{Vector{Vector{Vector{T}}}}}
    # Preallocated storage for noisy gradient descent calculations
    storage_L           :: Vector{T}
    storage_L2          :: Vector{T}
    storage_L3          :: Vector{T}
    storage_L4          :: Vector{T}
    storage_LL          :: Matrix{T}
    storage_LL2         :: Matrix{T}
    storage_LL3         :: Matrix{T}
    storage_LL4         :: Matrix{T}
    # Preallocated storage for matrix vectorization operations
    storage_comm        :: Matrix{T}
    storage_dup         :: Matrix{T}
    storage_Lsqr        :: Vector{T}
    storage_Lsqr2       :: Vector{T}
    storage_L2L2        :: Matrix{T}
    storage_C           :: Matrix{T}
    storage_gradC       :: Vector{T}
    # Preallocated Identity matrix
    I_LL                :: Matrix{T}
    # Preallocated storage for parallel noisy gradient descent calculations
    storage_L_par       :: Vector{Vector{T}}
    storage_L2_par      :: Vector{Vector{T}}
    storage_L3_par      :: Vector{Vector{T}}
    storage_L4_par      :: Vector{Vector{T}}
    storage_LL_par      :: Vector{Matrix{T}}
    storage_LL2_par     :: Vector{Matrix{T}}
    storage_LL3_par     :: Vector{Matrix{T}}
    storage_LL4_par     :: Vector{Matrix{T}}
    # Preallocated storage for parallel matrix vectorization operations
    storage_comm_par    :: Vector{Matrix{T}}
    storage_dup_par     :: Vector{Matrix{T}}
    storage_Lsqr_par    :: Vector{Vector{T}}
    storage_Lsqr2_par   :: Vector{Vector{T}}
    storage_L2L2_par    :: Vector{Matrix{T}}
    storage_C_par       :: Vector{Matrix{T}}
    storage_gradC_par   :: Vector{Vector{T}}
end

function TDCModel(
    obs                 :: TDCMObs,
    mu_beta_prior       :: Vector{Vector{T}},
    L_beta_prior        :: Vector{Matrix{T}},
    mu_omega_prior      :: Vector{Vector{Vector{Vector{Vector{T}}}}},
    L_omega_prior       :: Vector{Vector{Vector{Vector{Matrix{T}}}}},
    a_tau_prior         :: Vector{Vector{Vector{Vector{T}}}},
    b_tau_prior         :: Vector{Vector{Vector{Vector{T}}}},
    M                   :: Int;
    # This option allocates extra memory based on number of threads availible in the environment
    enable_parallel     :: Bool=false
) where T <: AbstractFloat
    # Number of students, time points, questions, skills, attribute profiles, groups
    N, O, J, K, L, S = size(obs.Y, 1), size(obs.Y, 2), size(obs.Y, 3),  size(obs.Q, 2), size(obs.D[1], 1), size(obs.U[1][1], 1)
    # Initialize variational distribution parameters
    pi_star = Vector{Vector{Vector{T}}}(undef, N)
    for i in 1:N
        pi_star[i] = Vector{Vector{T}}(undef, O)
        for t in 1:O
            # probability vector of possible single skill mastery over all time points
            pi_star[i][t] = ones(2^K) ./ 2^K
        end
    end
    mu_beta_star = Vector{Vector{T}}(undef, J)
    V_beta_star = Vector{Matrix{T}}(undef, J)
    for j in 1:J
        num_features = size(obs.D[j], 2)
        mu_beta_star[j] = zeros(num_features)
        V_beta_star[j] = Matrix(1.0I, num_features, num_features)
    end
    mu_gamma_star = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    V_gamma_star = Vector{Vector{Vector{Vector{Matrix{T}}}}}(undef, K)
    for k in 1:K
        mu_gamma_star[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        V_gamma_star[k] = Vector{Vector{Vector{Matrix{T}}}}(undef, O)
        for t in 1:O
            if t == 1
                mu_gamma_star[k][t] = Vector{Vector{Vector{T}}}(undef, 1)
                V_gamma_star[k][t] = Vector{Vector{Matrix{T}}}(undef, 1)

                mu_gamma_star[k][t][1] = Vector{Vector{T}}(undef, S)
                V_gamma_star[k][t][1] = Vector{Matrix{T}}(undef, S)
                for s in 1:S
                    mu_gamma_star[k][t][1][s] = zeros(1)
                    V_gamma_star[k][t][1][s] = ones(1, 1)
                end
            else
                mu_gamma_star[k][t] = Vector{Vector{Vector{T}}}(undef, 2)
                V_gamma_star[k][t] = Vector{Vector{Matrix{T}}}(undef, 2)
                num_features = size(obs.X[k][t], 2)
                for z in 1:2
                    mu_gamma_star[k][t][z] = Vector{Vector{T}}(undef, S)
                    V_gamma_star[k][t][z] = Vector{Matrix{T}}(undef, S)
                    for s in 1:S
                        mu_gamma_star[k][t][z][s] = zeros(num_features)
                        V_gamma_star[k][t][z][s] = Matrix(1.0I, num_features, num_features)
                    end
                end
            end
        end
    end
    mu_omega_star = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    V_omega_star = Vector{Vector{Vector{Vector{Matrix{T}}}}}(undef, K)
    a_tau_star = Vector{Vector{Vector{Vector{T}}}}(undef, K)
    b_tau_star = Vector{Vector{Vector{Vector{T}}}}(undef, K)
    for k in 1:K
        mu_omega_star[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        V_omega_star[k] = Vector{Vector{Vector{Matrix{T}}}}(undef, O)
        a_tau_star[k] = Vector{Vector{Vector{T}}}(undef, O)
        b_tau_star[k] = Vector{Vector{Vector{T}}}(undef, O)
        for t in 1:O
            num_features_gamma = size(obs.X[k][t], 2)
            num_features_omega = size(obs.U[k][t], 2)
            if t == 1
                mu_omega_star[k][t] = Vector{Vector{Vector{T}}}(undef, 1)
                mu_omega_star[k][t][1] = Vector{Vector{T}}(undef, 1)
                mu_omega_star[k][t][1][1] = zeros(num_features_omega)

                V_omega_star[k][t] = Vector{Vector{Matrix{T}}}(undef, 1)
                V_omega_star[k][t][1] = Vector{Matrix{T}}(undef, 1)
                V_omega_star[k][t][1][1] = Matrix{T}(1.0I, num_features_omega, num_features_omega)

                a_tau_star[k][t] = Vector{Vector{T}}(undef, 1)
                a_tau_star[k][t][1] = ones(1) .* S / 2

                b_tau_star[k][t] = Vector{Vector{T}}(undef, 1)
                b_tau_star[k][t][1] = ones(1)
            else
                mu_omega_star[k][t] = Vector{Vector{Vector{T}}}(undef, 2)
                V_omega_star[k][t] = Vector{Vector{Matrix{T}}}(undef, 2)
                a_tau_star[k][t] = Vector{Vector{T}}(undef, 2)
                b_tau_star[k][t] = Vector{Vector{T}}(undef, 2)
                for z in 1:2
                    mu_omega_star[k][t][z] = Vector{Vector{T}}(undef, num_features_gamma)
                    V_omega_star[k][t][z] = Vector{Matrix{T}}(undef, num_features_gamma)
                    a_tau_star[k][t][z] = ones(num_features_gamma) .* S / 2
                    b_tau_star[k][t][z] = ones(num_features_gamma) .* 1
                    for m in 1:num_features_gamma
                        mu_omega_star[k][t][z][m] = zeros(num_features_omega)
                        V_omega_star[k][t][z][m] = Matrix(1.0I, num_features_omega, num_features_omega)
                    end
                end
            end
        end
    end
    # Preallocate space for samples from variational distribution
    Z_sample = Vector{Vector{Vector{Vector{Int}}}}(undef, N)
    for i in 1:N
        Z_sample[i] = Vector{Vector{Vector{Int}}}(undef, O)
        for t in 1:O
            Z_sample[i][t] = Vector{Vector{Int}}(undef, M)
            for m in 1:M
                Z_sample[i][t][m] = Vector{Int}(undef, 2^(K))
            end
        end
    end
    beta_sample = Vector{Vector{Vector{T}}}(undef, J)
    for j in 1:J
        beta_sample[j] = Vector{Vector{T}}(undef, M)
        num_features = size(obs.D[j], 2)
        for m in 1:M
            beta_sample[j][m] = Vector{T}(undef, num_features)
        end
    end
    gamma_sample = Vector{Vector{Vector{Vector{Vector{Vector{T}}}}}}(undef, K)
    for k in 1:K
        gamma_sample[k] = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, O)
        for t in 1:O
            if t == 1
                gamma_sample[k][t] = Vector{Vector{Vector{Vector{T}}}}(undef, 1)
                gamma_sample[k][t][1] = Vector{Vector{Vector{T}}}(undef, S)
                for s in 1:S
                    gamma_sample[k][t][1][s] = Vector{Vector{T}}(undef, M)
                    for m in 1:M
                        gamma_sample[k][t][1][s][m] = Vector{T}(undef, 1)
                    end
                end
            else
                num_features = size(obs.X[k][t], 2)
                gamma_sample[k][t] = Vector{Vector{Vector{Vector{T}}}}(undef, 2)
                for z in 1:2
                    gamma_sample[k][t][z] = Vector{Vector{Vector{T}}}(undef, S)
                    for s in 1:S
                        gamma_sample[k][t][z][s] = Vector{Vector{T}}(undef, M)
                        for m in 1:M
                            gamma_sample[k][t][z][s][m] = Vector{T}(undef, num_features)
                        end
                    end
                end
            end
        end
    end
    omega_sample = Vector{Vector{Vector{Vector{Vector{Vector{T}}}}}}(undef, K)
    tau_sample = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    for k in 1:K
        omega_sample[k] = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, O)
        tau_sample[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        for t in 1:O
            num_features_gamma = size(obs.X[k][t], 2)
            num_features_omega = size(obs.U[k][t], 2)
            if t == 1
                omega_sample[k][t] = Vector{Vector{Vector{Vector{T}}}}(undef, 1)
                omega_sample[k][t][1] = Vector{Vector{Vector{T}}}(undef, 1)
                omega_sample[k][t][1][1] = Vector{Vector{T}}(undef, M)

                tau_sample[k][t] = Vector{Vector{Vector{T}}}(undef, 1)
                tau_sample[k][t][1] = Vector{Vector{T}}(undef, 1)
                tau_sample[k][t][1][1] = Vector{T}(undef, M)
                for m in 1:M
                    omega_sample[k][t][1][1][m] = Vector{T}(undef, num_features_omega)
                end
            else
                omega_sample[k][t] = Vector{Vector{Vector{Vector{T}}}}(undef, 2)
                tau_sample[k][t] = Vector{Vector{Vector{T}}}(undef, 2)
                for z in 1:2
                    omega_sample[k][t][z] = Vector{Vector{Vector{T}}}(undef, num_features_gamma)
                    tau_sample[k][t][z] = Vector{Vector{T}}(undef, num_features_gamma)
                    for g in 1:num_features_gamma
                        omega_sample[k][t][z][g] = Vector{Vector{T}}(undef, M)
                        tau_sample[k][t][z][g] = Vector{T}(undef, M)
                        for m in 1:M
                            omega_sample[k][t][z][g][m] = Vector{T}(undef, num_features_omega)
                        end
                    end
                end
            end
        end
    end
    # Determine maximum size needed for vector and matrix allocations
    for k in 1:K
        for t in 1:O
            L = max(L, size(obs.X[k][t], 2), size(obs.U[k][t], 2))
        end
    end
    # Preallocate storage for noisy gradient descent calculations
    storage_L = Vector{T}(undef, L)
    storage_L2 = similar(storage_L)
    storage_L3 = similar(storage_L)
    storage_L4 = similar(storage_L)
    storage_LL = Matrix{T}(undef, L, L)
    storage_LL2 = similar(storage_LL)
    storage_LL3 = similar(storage_LL)
    storage_LL4 = similar(storage_LL)
    # Preallocate storage for matrix vectorization operations
    storage_comm = Matrix{T}(undef, L^2, L^2)
    storage_dup = Matrix{T}(undef, L^2, Int(L*(L+1)/2))
    storage_Lsqr = Vector{T}(undef, L^2)
    storage_Lsqr2 = Vector{T}(undef, L^2)
    storage_L2L2 = Matrix{T}(undef, L^2, L^2)
    storage_C = Matrix{T}(undef, L, L)
    storage_gradC = Vector{T}(undef, Int(L*(L+1)/2))
    # Preallocate Identity matrix
    I_LL = Matrix{T}(I, L, L)
    # Allocate optional space for parallel computing
    nthreads = Threads.nthreads()
    storage_L_par = Vector{Vector{T}}(undef, nthreads)
    storage_L2_par = similar(storage_L_par)
    storage_L3_par = similar(storage_L_par)
    storage_L4_par = similar(storage_L_par)
    storage_LL_par = Vector{Matrix{T}}(undef, nthreads)
    storage_LL2_par = similar(storage_LL_par)
    storage_LL3_par = similar(storage_LL_par)
    storage_LL4_par = similar(storage_LL_par)
    storage_comm_par = Vector{Matrix{T}}(undef, nthreads)
    storage_dup_par = Vector{Matrix{T}}(undef, nthreads)
    storage_Lsqr_par = Vector{Vector{T}}(undef, nthreads)
    storage_Lsqr2_par = Vector{Vector{T}}(undef, nthreads)
    storage_L2L2_par = Vector{Matrix{T}}(undef, nthreads)
    storage_C_par = Vector{Matrix{T}}(undef, nthreads)
    storage_gradC_par = Vector{Vector{T}}(undef, nthreads)
    if enable_parallel
        storage_L_par[1] = storage_L
        storage_L2_par[1] = storage_L2
        storage_L3_par[1] = storage_L3
        storage_L4_par[1] = storage_L4
        storage_LL_par[1] = storage_LL
        storage_LL2_par[1] = storage_LL2
        storage_LL3_par[1] = storage_LL3
        storage_LL4_par[1] = storage_LL4
        storage_comm_par[1] = storage_comm
        storage_dup_par[1] = storage_dup
        storage_Lsqr_par[1] = storage_Lsqr
        storage_Lsqr2_par[1] = storage_Lsqr2
        storage_L2L2_par[1] = storage_L2L2
        storage_C_par[1] = storage_C
        storage_gradC_par[1] = storage_gradC
        for thread in 2:nthreads
            storage_L_par[thread] = Vector{T}(undef, L)
            storage_L2_par[thread] = similar(storage_L)
            storage_L3_par[thread] = similar(storage_L)
            storage_L4_par[thread] = similar(storage_L)
            storage_LL_par[thread] = Matrix{T}(undef, L, L)
            storage_LL2_par[thread] = similar(storage_LL)
            storage_LL3_par[thread] = similar(storage_LL)
            storage_LL4_par[thread] = similar(storage_LL)
            storage_comm_par[thread] = Matrix{T}(undef, L^2, L^2)
            storage_dup_par[thread] = Matrix{T}(undef, L^2, Int(L*(L+1)/2))
            storage_Lsqr_par[thread] = Vector{T}(undef, L^2)
            storage_Lsqr2_par[thread] = Vector{T}(undef, L^2)
            storage_L2L2_par[thread] = Matrix{T}(undef, L^2, L^2)
            storage_C_par[thread] = Matrix{T}(undef, L, L)
            storage_gradC_par[thread] = Vector{T}(undef, Int(L*(L+1)/2))
        end
        println("TDCModel constructed for computation on $nthreads threads")
    end
    # Initialize DCModel object
    TDCModel(obs, mu_beta_prior, L_beta_prior, mu_omega_prior, L_omega_prior, a_tau_prior, b_tau_prior, enable_parallel,
    pi_star, mu_beta_star, V_beta_star, mu_gamma_star, V_gamma_star, mu_omega_star, V_omega_star, a_tau_star, b_tau_star, M,
    Z_sample, beta_sample, gamma_sample, omega_sample, tau_sample,
    storage_L, storage_L2, storage_L3, storage_L4, storage_LL, storage_LL2, storage_LL3, storage_LL4,
    storage_comm, storage_dup, storage_Lsqr, storage_Lsqr2, storage_L2L2, storage_C, storage_gradC,
    I_LL, storage_L_par, storage_L2_par, storage_L3_par, storage_L4_par, storage_LL_par, storage_LL2_par, storage_LL3_par, storage_LL4_par,
    storage_comm_par, storage_dup_par, storage_Lsqr_par, storage_Lsqr2_par, storage_L2L2_par, storage_C_par, storage_gradC_par)
end

function sample_Z(
    model           :: TDCModel,
    idx_student     :: Int,
    idx_time        :: Int
)
    # Create variational distribution from model parameters for specific Z
    Z_it_variational_distribution = Multinomial(1, model.pi_star[idx_student][idx_time])
    # Populate preallocated arrays with samples from variational distribution
    rand!(Z_it_variational_distribution, model.Z_sample[idx_student][idx_time])
end

function sample_β(
    model           :: TDCModel;
    idx_question    :: Int = -1
)
    obs = model.obs
    J = size(obs.Y, 3)
    if idx_question == -1
        for j in 1:J
            # Create variational distribution from model parameters for each β_j
            beta_j_variational_distribution = MvNormal(model.mu_beta_star[j], model.V_beta_star[j])
            # Populate preallocated arrays with samples from variational distribution
            rand!(beta_j_variational_distribution, model.beta_sample[j])
        end
    else
        # Create variational distribution from model parameters for specific β_j
        beta_j_variational_distribution = MvNormal(model.mu_beta_star[idx_question], model.V_beta_star[idx_question])
        # Populate preallocated arrays with samples from variational distribution
        rand!(beta_j_variational_distribution, model.beta_sample[idx_question])
    end
end

function sample_γ(
    model           :: TDCModel,
    idx_group       :: Int,
    idx_time        :: Int,
    idx_skill       :: Int,
    indicator_skill :: Int
)
    # Create variational distribution from model parameters for γ
    gamma_stkz_variational_distribution = MvNormal(model.mu_gamma_star[idx_skill][idx_time][indicator_skill + 1][idx_group],
                                            model.V_gamma_star[idx_skill][idx_time][indicator_skill + 1][idx_group])
    # Populate preallocated arrays with samples from variational distribution
    rand!(gamma_stkz_variational_distribution, model.gamma_sample[idx_skill][idx_time][indicator_skill + 1][idx_group])
end

function sample_ω(
    model           :: TDCModel,
    idx_skill       :: Int,
    idx_time        :: Int,
    indicator_skill :: Int,
    idx_feature     :: Int
)
    # Create variational distribution from model parameters for ω
    omega_ktzm_variational_distribution = MvNormal(model.mu_omega_star[idx_skill][idx_time][indicator_skill + 1][idx_feature],
                                            model.V_omega_star[idx_skill][idx_time][indicator_skill + 1][idx_feature])
    # Populate preallocated arrays with samples from variational distribution
    rand!(omega_ktzm_variational_distribution, model.omega_sample[idx_skill][idx_time][indicator_skill + 1][idx_feature])
end

function sample_τ(
    model           :: TDCModel,
    idx_skill       :: Int,
    idx_time        :: Int,
    indicator_skill :: Int,
    idx_feature     :: Int
)
    # Create variational distribution from model parameters of τ
    tau_ktzm_variational_distribution = InverseGamma(model.a_tau_star[idx_skill][idx_time][indicator_skill + 1][idx_feature],
                                            model.b_tau_star[idx_skill][idx_time][indicator_skill + 1][idx_feature])
    # Populate preallocated arrays with samples from variational distribution
    rand!(tau_ktzm_variational_distribution, model.tau_sample[idx_skill][idx_time][indicator_skill + 1][idx_feature])
end

function update_categorical_variational_distribution(
    model               :: TDCModel;
    step                :: T = 1e-2,
    tol                 :: T = 1e-6,
    maxiter             :: Int = 100000,
    verbose             :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    Y, D, X = obs.Y, obs.D, obs.X
    Z_sample, beta_sample, gamma_sample = model.Z_sample, model.beta_sample, model.gamma_sample
    pi_star_old = model.pi_star
    # Number of students, time points, questions, skills, attribute profiles, groups
    N, O, J, K, L, S = size(obs.Y, 1), size(obs.Y, 2), size(obs.Y, 3),  size(obs.Q, 2), size(obs.D[1], 1), size(obs.U[1][1], 1)
    M = model.M
    # Fully update parameters of each Z_i using noisy gradients before moving to update parameters of next Z_i
    if !model.enable_parallel
        @inbounds for i in 1:N
            # Storage for gradient terms
            grad_log_q = model.storage_L2
            grad_L = model.storage_L3
            # Storage for intermediate term in gradient calculations
            D_beta = model.storage_L
            rho_star_old_i = view(model.storage_LL3, 1:L)
            # Get parameters for variational distribution of skill of i-th student
            pi_star_old_i = pi_star_old[i][1]
            # Get group number of student i
            group_i = obs.group[i]
            # Perform gradient descent update of i-th π*    
            @inbounds for iter in 1:maxiter
                # Rho is unique up to a constant addative term
                rho_star_old_i = log.(pi_star_old_i)
                # Sample Z with updated π*
                sample_Z(model, i, 1)
                # Set gradient of ELBO to 0
                fill!(grad_L, 0)
                # Rao Blackwellized ELBO
                ELBO = 0
                # Calculate the gradient estimate of the m-th sample
                @inbounds for m in 1:M
                    z_im = Z_sample[i][1][m]
                    # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                    grad_log_q .= z_im .- pi_star_old_i
                    # Calculate log(p(Y, Z_(i)))
                    log_prob_YZ = 0
                    @inbounds for j in 1:J
                        mul!(D_beta, D[j], beta_sample[j][m])
                        log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i, 1, j] - 1) .* D_beta)))
                    end
                    skill_profile = obs.skill_dict[argmax(z_im)]
                    @inbounds for k in 1:K
                        log_prob_YZ += log(sigmoid((2*skill_profile[k] - 1) * dot(gamma_sample[k][1][1][group_i][m], obs.X[k][1][i])))
                    end
                    # Calculate log(q_1i(Z_i))
                    log_q = dot(z_im, log.(pi_star_old_i))
                    # Update average gradient
                    grad_L .= (m - 1)/m .* grad_L + 1/m .* grad_log_q .* (log_prob_YZ - log_q)
                    # Update ELBO estimator
                    ELBO = (m-1)/m * ELBO + 1/m * (log_prob_YZ - log_q)
                end
                # Print ELBO, parameter and gradient if verbose
                if verbose
                    println("ELBO: $ELBO")
                    println("π*_$i: $pi_star_old_i")
                    println("gradient: $grad_L")
                end
                # Update with one step
                rho_star_old_i .+= step * grad_L
                # Convert logits into probabilities
                pi_star_old_i .= exp.(rho_star_old_i) ./ sum(exp.(rho_star_old_i))
                # Stop condition
                if abs2(norm(grad_L)) <= tol
                    break
                end
            end
        end
    else
        Threads.@threads for i in 1:N
            # Get thread id
            tid = Threads.threadid()
            # Storage for gradient terms
            grad_log_q = model.storage_L2_par[tid]
            grad_L = model.storage_L3_par[tid]
            # Storage for intermediate term in gradient calculations
            D_beta = model.storage_L_par[tid]
            rho_star_old_i = view(model.storage_LL3_par[tid], 1:L)
            # Get parameters for variational distribution of skill of i-th student
            pi_star_old_i = pi_star_old[i][1]
            # Get group number of student i
            group_i = obs.group[i]
            # Perform gradient descent update of i-th π*    
            @inbounds for iter in 1:maxiter
                # Rho is unique up to a constant addative term
                rho_star_old_i = log.(pi_star_old_i)
                # Sample Z with updated π*
                sample_Z(model, i, 1)
                # Set gradient of ELBO to 0
                fill!(grad_L, 0)
                # Rao Blackwellized ELBO
                ELBO = 0
                # Calculate the gradient estimate of the m-th sample
                @inbounds for m in 1:M
                    z_im = Z_sample[i][1][m]
                    # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                    grad_log_q .= z_im .- pi_star_old_i
                    # Calculate log(p(Y, Z_(i)))
                    log_prob_YZ = 0
                    for j in 1:J
                        mul!(D_beta, D[j], beta_sample[j][m])
                        log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i, 1, j] - 1) .* D_beta)))
                    end
                    skill_profile = obs.skill_dict[argmax(z_im)]
                    for k in 1:K
                        log_prob_YZ += log(sigmoid((2*skill_profile[k] - 1) * dot(gamma_sample[k][1][1][group_i][m], obs.X[k][1][i])))
                    end
                    # Calculate log(q_1i(Z_i))
                    log_q = dot(z_im, log.(pi_star_old_i))
                    # Update average gradient
                    grad_L .= (m - 1)/m .* grad_L + 1/m .* grad_log_q .* (log_prob_YZ - log_q)
                    # Update ELBO estimator
                    ELBO = (m-1)/m * ELBO + 1/m * (log_prob_YZ - log_q)
                end
                # Print ELBO, parameter and gradient if verbose
                if verbose
                    println("ELBO: $ELBO")
                    println("π*_$i: $pi_star_old_i")
                    println("gradient: $grad_L")
                end
                # Update with one step
                rho_star_old_i .+= step * grad_L
                # Convert logits into probabilities
                pi_star_old_i .= exp.(rho_star_old_i) ./ sum(exp.(rho_star_old_i))
                # Stop condition
                if abs2(norm(grad_L)) <= tol
                    break
                end
            end
        end
    end
end

function update_categorical_variational_distribution2(
    model               :: TDCModel,
    time                :: Int;
    step                :: T = 1e-2,
    tol                 :: T = 1e-6,
    maxiter             :: Int = 100000,
    verbose             :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    Y, D, X = obs.Y, obs.D, obs.X
    Z_sample, beta_sample, gamma_sample = model.Z_sample, model.beta_sample, model.gamma_sample
    pi_star_old = model.pi_star
    # Number of students, time points, questions, skills, attribute profiles, groups
    N, O, J, K, L, S = size(obs.Y, 1), size(obs.Y, 2), size(obs.Y, 3),  size(obs.Q, 2), size(obs.D[1], 1), size(obs.U[1][1], 1)
    M = model.M
    # Fully update parameters of each Z_i using noisy gradients before moving to update parameters of next Z_i
    if !model.enable_parallel
        @inbounds for i in 1:N
            # Storage for gradient terms
            grad_log_q = model.storage_L2
            grad_L = model.storage_L3
            # Storage for intermediate term in gradient calculations
            D_beta = model.storage_L
            rho_star_old_i = view(model.storage_LL3, 1:L)
            # Get parameters for variational distribution of skill of i-th student
            pi_star_old_i = pi_star_old[i][time]
            # Get group number of student i
            group_i = obs.group[i]
            # Perform gradient descent update of i-th π*    
            for iter in 1:maxiter
                # Rho is unique up to a constant addative term
                rho_star_old_i = log.(pi_star_old_i)
                # Sample Z with updated π*
                sample_Z(model, i, time)
                # Set gradient of ELBO to 0
                fill!(grad_L, 0)
                # Rao Blackwellized ELBO
                ELBO = 0
                # Calculate the gradient estimate of the m-th sample
                for m in 1:M
                    z_im = Z_sample[i][time][m]
                    # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                    grad_log_q .= z_im .- pi_star_old_i
                    # Calculate log(p(Y, Z_(i)))
                    log_prob_YZ = 0
                    for j in 1:J
                        mul!(D_beta, D[j], beta_sample[j][m])
                        log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i, time, j] - 1) .* D_beta)))
                    end
                    skill_profile = obs.skill_dict[argmax(z_im)]
                    prev_skill_profile = obs.skill_dict[argmax(Z_sample[i][time - 1][m])]
                    for k in 1:K
                        log_prob_YZ += log(sigmoid((2*skill_profile[k] - 1) * 
                                    dot(gamma_sample[k][time][prev_skill_profile[k] + 1][group_i][m], obs.X[k][time][i, :])))
                    end
                    # Calculate log(q_1i(Z_i))
                    log_q = dot(z_im, log.(pi_star_old_i))
                    # Update average gradient
                    grad_L .= (m - 1)/m .* grad_L + 1/m .* grad_log_q .* (log_prob_YZ - log_q)
                    # Update ELBO estimator
                    ELBO = (m-1)/m * ELBO + 1/m * (log_prob_YZ - log_q)
                end
                # Print ELBO, parameter and gradient if verbose
                if verbose
                    println("ELBO: $ELBO")
                    println("π*_$i: $pi_star_old_i")
                    println("gradient: $grad_L")
                end
                # Update with one step
                rho_star_old_i .+= step * grad_L
                # Convert logits into probabilities
                pi_star_old_i .= exp.(rho_star_old_i) ./ sum(exp.(rho_star_old_i))
            end
        end
    else
        Threads.@threads for i in 1:N
            # Get thread id
            tid = Threads.threadid()
            # Storage for gradient terms
            grad_log_q = model.storage_L2_par[tid]
            grad_L = model.storage_L3_par[tid]
            # Storage for intermediate term in gradient calculations
            D_beta = model.storage_L_par[tid]
            rho_star_old_i = view(model.storage_LL3_par[tid], 1:L)
            # Get parameters for variational distribution of skill of i-th student
            pi_star_old_i = pi_star_old[i][time]
            # Get group number of student i
            group_i = obs.group[i]
            # Perform gradient descent update of i-th π*    
            @inbounds for iter in 1:maxiter
                # Rho is unique up to a constant addative term
                rho_star_old_i = log.(pi_star_old_i)
                # Sample Z with updated π*
                sample_Z(model, i, time)
                # Set gradient of ELBO to 0
                fill!(grad_L, 0)
                # Rao Blackwellized ELBO
                ELBO = 0
                # Calculate the gradient estimate of the m-th sample
                for m in 1:M
                    z_im = Z_sample[i][time][m]
                    # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                    grad_log_q .= z_im .- pi_star_old_i
                    # Calculate log(p(Y, Z_(i)))
                    log_prob_YZ = 0
                    for j in 1:J
                        mul!(D_beta, D[j], beta_sample[j][m])
                        log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i, time, j] - 1) .* D_beta)))
                    end
                    skill_profile = obs.skill_dict[argmax(z_im)]
                    prev_skill_profile = obs.skill_dict[argmax(Z_sample[i][time - 1][m])]
                    for k in 1:K
                        log_prob_YZ += log(sigmoid((2*skill_profile[k] - 1) * 
                                    dot(gamma_sample[k][time][prev_skill_profile[k] + 1][group_i][m], obs.X[k][time][i, :])))
                    end
                    # Calculate log(q_1i(Z_i))
                    log_q = dot(z_im, log.(pi_star_old_i))
                    # Update average gradient
                    grad_L .= (m - 1)/m .* grad_L + 1/m .* grad_log_q .* (log_prob_YZ - log_q)
                    # Update ELBO estimator
                    ELBO = (m-1)/m * ELBO + 1/m * (log_prob_YZ - log_q)
                end
                # Print ELBO, parameter and gradient if verbose
                if verbose
                    println("ELBO: $ELBO")
                    println("π*_$i: $pi_star_old_i")
                    println("gradient: $grad_L")
                end
                # Update with one step
                rho_star_old_i .+= step * grad_L
                # Convert logits into probabilities
                pi_star_old_i .= exp.(rho_star_old_i) ./ sum(exp.(rho_star_old_i))
            end
        end
    end
end

function update_normal_variational_distribution(
    model       :: TDCModel;
    init_step   :: T=1e-3,
    step_iterator=get_robbins_monroe_iterator(init_step, 20),
    # step_iterator_factory=get_robbins_monroe_iterator,
    use_iter    :: Bool=false,
    tol         :: T=1e-6,
    maxiter     :: Int=100000,
    verbose     :: Bool=true
) where T <: AbstractFloat

    ELBO_tracker = Vector{T}(undef, maxiter)
    log_prob_tracker = Vector{T}(undef, maxiter)
    log_q_tracker = Vector{T}(undef, maxiter)
    prob_comp1_tracker = Vector{T}(undef, maxiter)
    prob_comp2_tracker = Vector{T}(undef, maxiter)

    obs = model.obs
    Y, D = Array{T, 3}(obs.Y), Vector{Matrix{T}}(obs.D)
    Z_sample, beta_sample = model.Z_sample, model.beta_sample
    mu_star_old, V_star_old = model.mu_beta_star, model.V_beta_star
    N, J, L, O = size(Y, 1), size(Y, 3), size(D[1], 1), size(obs.Y, 2)
    M = model.M
    # Fully update parameters of each β_j using noisy gradients before moving to update parameters of next β_j
    if !model.enable_parallel
        @inbounds for j in 1:J
            mu_star_old_j = mu_star_old[j]
            V_star_old_j = V_star_old[j]
            # Perform gradient descent update of mu_j and V_j
            len_beta = length(beta_sample[j][1])
            # Assign storage for gradient terms
            # Memory assigned from preallocated storage
            # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
            # (important for vectorized matricies to be strided if we want to use them for linear algebra)
            # Matricies are stored column major in Julia, so memory is assigned by column left to right
            grad_mu_L = view(model.storage_L, 1:len_beta)
            grad_C_L = view(model.storage_LL2, 1:len_beta, 1:len_beta)
            vech_grad_C_L = view(grad_C_L, [len_beta * (j - 1) + i for j in 1:len_beta for i in j:len_beta]) # Uses same memory as grad_C_L
            grad_mu_log_q = view(model.storage_L2, 1:len_beta)
            vec_grad_V_log_q = view(model.storage_LL3, 1:len_beta^2)
            grad_V_log_q = reshape(vec_grad_V_log_q, len_beta, len_beta) # Uses same memory as vec_grad_V_log_q
            # Assign storage for calculating intermediate terms for gradient
            Vinv_star_old_j = view(model.storage_LL, 1:len_beta, 1:len_beta)
            beta_minus_mu = view(model.storage_L3, 1:len_beta)
            L_beta_minus_mu = view(model.storage_L4, 1:len_beta)
            C_star_old_j = view(model.storage_C, 1:len_beta, 1:len_beta)
            vech_C_star_old_j = view(C_star_old_j, [len_beta * (j - 1) + i for j in 1:len_beta for i in j:len_beta]) # Uses same memory as C_star_old_j
            fill!(C_star_old_j, 0)
            storage_kron_prod = view(model.storage_L2L2, 1:len_beta^2, 1:len_beta^2)
            storage_len_beta_sqr = view(model.storage_Lsqr, 1:len_beta^2)
            storage_len_beta_sqr2 = view(model.storage_Lsqr2, 1:len_beta^2)
            storage_gradC = view(model.storage_gradC, 1:Int(len_beta * (len_beta + 1) / 2))
            # Generate commutation and duplication matrix
            comm_j = view(model.storage_comm, 1:len_beta^2, 1:len_beta^2)
            dup_j = view(model.storage_dup, 1:len_beta^2, 1:Int(len_beta * (len_beta + 1) / 2))
            get_comm!(comm_j, len_beta)
            get_dup!(dup_j, len_beta)
            # Assign len_beta by len_beta identity matrix
            I_j = view(model.I_LL, 1:len_beta, 1:len_beta)
            # # Get step size iterator
            # step_iterator = step_iterator_factory(init_step)
            for iter in 1:maxiter
                # Sample β from variational distribution
                sample_β(model, idx_question=j)
                fill!(grad_mu_L, 0)
                fill!(grad_C_L, 0)
                # Copy V* into storage
                copy!(Vinv_star_old_j, V_star_old_j)
                # Perform cholesky decomposition on V*
                # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                LAPACK.potrf!('L', Vinv_star_old_j)
                # Calculate log|V_j| from diagonal of cholesky decomposition
                logdet_V_j = 0
                for b in 1:len_beta
                    logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                end
                # Copy lower triangular cholesky factor into preallocated storage
                for k in 1:len_beta
                    for l in 1:k
                        C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                    end
                end
                # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                LAPACK.potri!('L', Vinv_star_old_j)
                LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                ELBO = 0

                # Calculate the gradient estimate of the m-th sample
                for m in 1:M
                    beta_jm = beta_sample[j][m]
                    fill!(grad_mu_log_q, 0)
                    # grad_mu_log_q = Vinv_star * β_jm
                    BLAS.gemv!('N', T(1), Vinv_star_old_j, beta_jm, T(1), grad_mu_log_q)
                    # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                    BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                    # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                    copy!(grad_V_log_q, Vinv_star_old_j)
                    BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                    # storage_kron_prod = I ⊗ C_j
                    collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                    # storage_len_beta_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_beta_sqr, 0))
                    # storage_kron_prod = C_j ⊗ I
                    collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                    # storage_len_beta_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_beta_sqr2, 0))
                    # storage_len_beta_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), comm_j, storage_len_beta_sqr, T(1), storage_len_beta_sqr2)
                    # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), dup_j, storage_len_beta_sqr2, T(1), fill!(storage_gradC, 0))
                    # Calculate log(p(Y, β_(j)))
                    log_prob_Ybeta = 0
                    for i in 1:N
                        for t in 1:O
                            # fill!(model.storage_L3, 0)
                            # BLAS.gemv!('N', (2 * Y[i, t, j] - 1), D[j], beta_jm, T(1), model.storage_L3)
                            # log_prob_Ybeta += dot(Z_sample[i][t][m], log.(sigmoid.(model.storage_L3)))

                            log_prob_Ybeta += log(sigmoid((2 * Y[i, t, j] - 1) * dot(D[j][argmax(Z_sample[i][t][m]), :], beta_jm)))
                        end
                    end
                    beta_minus_mu .= beta_jm
                    beta_minus_mu .-= model.mu_beta_prior[j]
                    mul!(L_beta_minus_mu, model.L_beta_prior[j], beta_minus_mu)
                    log_prob_Ybeta -= 1/2 * dot(L_beta_minus_mu, L_beta_minus_mu)
                    beta_minus_mu .= beta_jm
                    beta_minus_mu .-= mu_star_old_j
                    log_q = -len_beta / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(beta_minus_mu, grad_mu_log_q)
                    # Update average gradient
                    grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Ybeta - log_q)
                    vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Ybeta - log_q)
                    # Update ELBO estimator
                    ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Ybeta - log_q)
                    log_prob = (m - 1) / m * log_prob + 1 / m * (log_prob_Ybeta)
                    log_q_avg = (m - 1) / m * log_q_avg + 1 / m * (log_q)
                end
                # Print ELBO, parameter and gradient if verbose
                # if verbose
                #     println("ELBO: $ELBO")
                #     println("mu*_$j: mu_star_old_j")
                #     println("gradient mu: $grad_mu_L")
                #     println("C*_$j: C_star_old_j")
                #     println("gradient C: $grad_C_L")
                # end
                # Update mu and C with one step
                step = init_step
                if use_iter
                    step = step_iterator()
                end
                mu_star_old_j .+= sqrt(len_beta) .* step .* grad_mu_L ./ norm(grad_mu_L)
                vech_C_star_old_j .+= len_beta .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                # Set V_star_old_j = C * C'
                BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
            end
        end
    else
        Threads.@threads for j in 1:J
            # Get thread id
            tid = Threads.threadid()
            # Perform gradient descent update of mu_j and V_j
            len_beta = length(beta_sample[j][1])
            # Assign storage for gradient terms
            # Memory assigned from preallocated storage
            # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
            # (important for vectorized matricies to be strided if we want to use them for linear algebra)
            # Matricies are stored column major in Julia, so memory is assigned by column left to right
            grad_mu_L = view(model.storage_L_par[tid], 1:len_beta)
            grad_C_L = view(model.storage_LL2_par[tid], 1:len_beta, 1:len_beta)
            vech_grad_C_L = view(grad_C_L, [len_beta * (j - 1) + i for j in 1:len_beta for i in j:len_beta]) # Uses same memory as grad_C_L
            grad_mu_log_q = view(model.storage_L2_par[tid], 1:len_beta)
            vec_grad_V_log_q = view(model.storage_LL3_par[tid], 1:len_beta^2)
            grad_V_log_q = reshape(vec_grad_V_log_q, len_beta, len_beta) # Uses same memory as vec_grad_V_log_q
            # Assign storage for calculating intermediate terms for gradient
            Vinv_star_old_j = view(model.storage_LL_par[tid], 1:len_beta, 1:len_beta)
            beta_minus_mu = view(model.storage_L3_par[tid], 1:len_beta)
            L_beta_minus_mu = view(model.storage_L4_par[tid], 1:len_beta)
            C_star_old_j = view(model.storage_C_par[tid], 1:len_beta, 1:len_beta)
            vech_C_star_old_j = view(C_star_old_j, [len_beta * (j - 1) + i for j in 1:len_beta for i in j:len_beta]) # Uses same memory as C_star_old_j
            fill!(C_star_old_j, 0)
            storage_kron_prod = view(model.storage_L2L2_par[tid], 1:len_beta^2, 1:len_beta^2)
            storage_len_beta_sqr = view(model.storage_Lsqr_par[tid], 1:len_beta^2)
            storage_len_beta_sqr2 = view(model.storage_Lsqr2_par[tid], 1:len_beta^2)
            storage_gradC = view(model.storage_gradC_par[tid], 1:Int(len_beta * (len_beta + 1) / 2))
            # Generate commutation and duplication matrix
            comm_j = view(model.storage_comm_par[tid], 1:len_beta^2, 1:len_beta^2)
            dup_j = view(model.storage_dup_par[tid], 1:len_beta^2, 1:Int(len_beta * (len_beta + 1) / 2))
            get_comm!(comm_j, len_beta)
            get_dup!(dup_j, len_beta)
            # Assign len_beta by len_beta identity matrix
            I_j = view(model.I_LL, 1:len_beta, 1:len_beta)
            # # Get step iterator
            # step_iterator = step_iterator_factory(init_step)
            # # Initialize variables for tracking previous values
            # prev_ELBO = -Inf
            # prev_mu = view(model.storage_L4, 1:len_beta)
            # prev_V = view(model.storage_LL4, 1:len_beta, 1:len_beta)
            # prev_mu .= mu_star_old[j]
            # prev_V .= V_star_old[j]
            @inbounds for iter in 1:maxiter
                # Sample β from variational distribution
                sample_β(model, idx_question=j)
                fill!(grad_mu_L, 0)
                fill!(grad_C_L, 0)
                mu_star_old_j = mu_star_old[j]
                V_star_old_j = V_star_old[j]
                # Copy V* into storage
                copy!(Vinv_star_old_j, V_star_old_j)
                # Perform cholesky decomposition on V*
                # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                LAPACK.potrf!('L', Vinv_star_old_j)
                # Calculate log|V_j| from diagonal of cholesky decomposition
                logdet_V_j = 0
                for b in 1:len_beta
                    logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                end
                # Copy lower triangular cholesky factor into preallocated storage
                for k in 1:len_beta
                    for l in 1:k
                        C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                    end
                end
                # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                LAPACK.potri!('L', Vinv_star_old_j)
                LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                ELBO = 0
                log_prob = 0
                log_q_avg = 0
                prob_comp1 = 0
                prob_comp2 = 0

                # Calculate the gradient estimate of the m-th sample
                for m in 1:M
                    beta_jm = beta_sample[j][m]
                    fill!(grad_mu_log_q, 0)
                    # grad_mu_log_q = Vinv_star * β_jm
                    BLAS.gemv!('N', T(1), Vinv_star_old_j, beta_jm, T(1), grad_mu_log_q)
                    # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                    BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                    # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                    copy!(grad_V_log_q, Vinv_star_old_j)
                    BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                    # storage_kron_prod = I ⊗ C_j
                    collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                    # storage_len_beta_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_beta_sqr, 0))
                    # storage_kron_prod = C_j ⊗ I
                    collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                    # storage_len_beta_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_beta_sqr2, 0))
                    # storage_len_beta_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), comm_j, storage_len_beta_sqr, T(1), storage_len_beta_sqr2)
                    # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                    BLAS.gemv!('T', T(1), dup_j, storage_len_beta_sqr2, T(1), fill!(storage_gradC, 0))
                    # Calculate log(p(Y, β_(j)))
                    log_prob_Ybeta = 0
                    for i in 1:N
                        for t in 1:O
                            # fill!(model.storage_L3_par[tid], 0)
                            # BLAS.gemv!('N', (2 * Y[i, t, j] - 1), D[j], beta_jm, T(1), model.storage_L3_par[tid])
                            # log_prob_Ybeta += dot(Z_sample[i][t][m], log.(sigmoid.(model.storage_L3_par[tid])))

                            log_prob_Ybeta += log(sigmoid((2 * Y[i, t, j] - 1) * dot(D[j][argmax(Z_sample[i][t][m]), :], beta_jm)))
                        end
                    end

                    prob_comp1_samp = log_prob_Ybeta

                    beta_minus_mu .= beta_jm
                    beta_minus_mu .-= model.mu_beta_prior[j]
                    mul!(L_beta_minus_mu, model.L_beta_prior[j], beta_minus_mu)
                    log_prob_Ybeta -= 1/2 * dot(L_beta_minus_mu, L_beta_minus_mu)

                    prob_comp2_samp = log_prob_Ybeta - prob_comp1

                    beta_minus_mu .= beta_jm
                    beta_minus_mu .-= mu_star_old_j
                    log_q = -len_beta / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(beta_minus_mu, grad_mu_log_q)
                    # Update average gradient
                    grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Ybeta - log_q)
                    vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Ybeta - log_q)
                    # Update ELBO estimator
                    ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Ybeta - log_q)
                    log_prob = (m - 1) / m * log_prob + 1 / m * (log_prob_Ybeta)
                    log_q_avg = (m - 1) / m * log_q_avg + 1 / m * (log_q)
                    prob_comp1 = (m - 1) / m * prob_comp1 + 1 / m * prob_comp1_samp
                    prob_comp2 = (m - 1) / m * prob_comp2 + 1 / m * prob_comp2_samp
                end
                step = init_step
                if use_iter
                    step = step_iterator()
                end

                mu_star_old_j .+= sqrt(len_beta) .* step .* grad_mu_L ./ norm(grad_mu_L)
                vech_C_star_old_j .+= len_beta .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                # Set V_star_old_j = C * C'
                BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))

                if j==1
                    ELBO_tracker[iter] = ELBO
                    log_prob_tracker[iter] = log_prob
                    log_q_tracker[iter] = log_q_avg
                    prob_comp1_tracker[iter] = prob_comp1
                    prob_comp2_tracker[iter] = prob_comp2
                end
            end
        end
    end
    return ELBO_tracker, log_prob_tracker, log_q_tracker, prob_comp1_tracker, prob_comp2_tracker
end

function update_normal_variational_distribution2(
    model       :: TDCModel;
    init_step   :: T=1e-3,
    cov_step    :: T=1e-3,
    step_iterator=get_robbins_monroe_iterator(init_step, 20),
    # step_iterator_factory=get_robbins_monroe_iterator,
    use_iter    :: Bool=false,
    tol         :: T=1e-6,
    maxiter     :: Int=100000,
    verbose     :: Bool=true
) where T <: AbstractFloat

    obs = model.obs
    Y, D = Array{T, 3}(obs.Y), Vector{Matrix{T}}(obs.D)
    Z_sample, gamma_sample, omega_sample, tau_sample = model.Z_sample, model.gamma_sample, model.omega_sample, model.tau_sample
    mu_star_old, V_star_old = model.mu_gamma_star, model.V_gamma_star
    N, J, L, O, K = size(Y, 1), size(Y, 3), size(D[1], 1), size(obs.Y, 2), size(obs.Q, 2)
    M = model.M
    
    # Fully update parameters of each γ using noisy gradients before moving to update parameters of next γ
    if !model.enable_parallel
        @inbounds for k in 1:K
            for t in 1:O
                for z in 0:1
                    if t == 1 && z == 1
                        continue
                    end
                    for s in 1:S
                        mu_star_old_j = mu_star_old[k][t][z + 1][s]
                        V_star_old_j = V_star_old[k][t][z + 1][s]
                        # Perform gradient descent update of mu_j and V_j
                        len_gamma = length(mu_star_old_j)
                        # Assign storage for gradient terms
                        # Memory assigned from preallocated storage
                        # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
                        # (important for vectorized matricies to be strided if we want to use them for linear algebra)
                        # Matricies are stored column major in Julia, so memory is assigned by column left to right
                        grad_mu_L = view(model.storage_L, 1:len_gamma)
                        grad_C_L = view(model.storage_LL2, 1:len_gamma, 1:len_gamma)
                        vech_grad_C_L = view(grad_C_L, [len_gamma * (j - 1) + i for j in 1:len_gamma for i in j:len_gamma]) # Uses same memory as grad_C_L
                        grad_mu_log_q = view(model.storage_L2, 1:len_gamma)
                        vec_grad_V_log_q = view(model.storage_LL3, 1:len_gamma^2)
                        grad_V_log_q = reshape(vec_grad_V_log_q, len_gamma, len_gamma) # Uses same memory as vec_grad_V_log_q
                        # Assign storage for calculating intermediate terms for gradient
                        Vinv_star_old_j = view(model.storage_LL, 1:len_gamma, 1:len_gamma)
                        gamma_minus_mu = view(model.storage_L3, 1:len_gamma)
                        C_star_old_j = view(model.storage_C, 1:len_gamma, 1:len_gamma)
                        vech_C_star_old_j = view(C_star_old_j, [len_gamma * (j - 1) + i for j in 1:len_gamma for i in j:len_gamma]) # Uses same memory as C_star_old_j
                        fill!(C_star_old_j, 0)
                        storage_kron_prod = view(model.storage_L2L2, 1:len_gamma^2, 1:len_gamma^2)
                        storage_len_gamma_sqr = view(model.storage_Lsqr, 1:len_gamma^2)
                        storage_len_gamma_sqr2 = view(model.storage_Lsqr2, 1:len_gamma^2)
                        storage_gradC = view(model.storage_gradC, 1:Int(len_gamma * (len_gamma + 1) / 2))
                        # Generate commutation and duplication matrix
                        comm_j = view(model.storage_comm, 1:len_gamma^2, 1:len_gamma^2)
                        dup_j = view(model.storage_dup, 1:len_gamma^2, 1:Int(len_gamma * (len_gamma + 1) / 2))
                        get_comm!(comm_j, len_gamma)
                        get_dup!(dup_j, len_gamma)
                        # Assign len_gamma by len_gamma identity matrix
                        I_j = view(model.I_LL, 1:len_gamma, 1:len_gamma)
                        # # Get step size iterator
                        # step_iterator = step_iterator_factory(init_step)
                        for iter in 1:maxiter
                            # Sample β from variational distribution
                            sample_γ(model, s, t, k, z)
                            fill!(grad_mu_L, 0)
                            fill!(grad_C_L, 0)
                            # Copy V* into storage
                            copy!(Vinv_star_old_j, V_star_old_j)
                            # Perform cholesky decomposition on V*
                            # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                            LAPACK.potrf!('L', Vinv_star_old_j)
                            # Calculate log|V_j| from diagonal of cholesky decomposition
                            logdet_V_j = 0
                            for b in 1:len_gamma
                                logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                            end
                            # Copy lower triangular cholesky factor into preallocated storage
                            for k in 1:len_gamma
                                for l in 1:k
                                    C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                                end
                            end
                            # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                            LAPACK.potri!('L', Vinv_star_old_j)
                            LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                            ELBO = 0

                            # Calculate the gradient estimate of the m-th sample
                            for m in 1:M
                                gamma_ktzsm = gamma_sample[k][t][z + 1][s][m]
                                fill!(grad_mu_log_q, 0)
                                # grad_mu_log_q = Vinv_star * β_jm
                                BLAS.gemv!('N', T(1), Vinv_star_old_j, gamma_ktzsm, T(1), grad_mu_log_q)
                                # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                                BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                                # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                                copy!(grad_V_log_q, Vinv_star_old_j)
                                BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                                # storage_kron_prod = I ⊗ C_j
                                collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                                # storage_len_gamma_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_gamma_sqr, 0))
                                # storage_kron_prod = C_j ⊗ I
                                collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                                # storage_len_gamma_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_gamma_sqr2, 0))
                                # storage_len_gamma_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), comm_j, storage_len_gamma_sqr, T(1), storage_len_gamma_sqr2)
                                # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), dup_j, storage_len_gamma_sqr2, T(1), fill!(storage_gradC, 0))
                                # Calculate log(p(Y, β_(j)))
                                log_prob_Ygamma = 0
                                for i in 1:N
                                    if obs.group[i] != s
                                        continue
                                    end
                                    if t > 1
                                        prev_skill_profile = obs.skill_dict[argmax(Z_sample[i][t - 1][m])]
                                        if prev_skill_profile[k] != z
                                            continue
                                        end
                                    end
                                    skill_profile = obs.skill_dict[argmax(Z_sample[i][t][m])]
                                    log_prob_Ygamma += log(sigmoid((2*skill_profile[k] - 1) * 
                                        dot(gamma_sample[k][t][z + 1][s][m], obs.X[k][t][i, :])))
                                end
                                num_features = length(gamma_ktzsm)
                                for feature in 1:num_features
                                    tau = tau_sample[k][t][z + 1][feature][m]
                                    omega = omega_sample[k][t][z + 1][feature][m]
                                    u = obs.U[k][t][s, :]
                                    log_prob_Ygamma += - 1/(2 * tau) * (gamma_ktzsm[feature] - dot(u, omega))^2
                                end
                                gamma_minus_mu .= gamma_ktzsm
                                gamma_minus_mu .-= mu_star_old_j
                                log_q = -len_gamma / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(gamma_minus_mu, grad_mu_log_q)
                                # Update average gradient
                                grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Ygamma - log_q)
                                vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Ygamma - log_q)
                                # Update ELBO estimator
                                ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Ygamma - log_q)
                            end
                            # Print ELBO, parameter and gradient if verbose
                            if verbose
                                println("ELBO: $ELBO")
                                # println("mu*_$j: mu_star_old_j")
                                # println("gradient mu: $grad_mu_L")
                                println("C*_$k$t$z$s: $C_star_old_j")
                                println("gradient C: $grad_C_L")
                            end

                            # Update mu and C with one step
                            step = init_step
                            if use_iter
                                step = step_iterator()
                            end
                            mu_star_old_j .+= sqrt(len_gamma) .* step .* grad_mu_L ./ norm(grad_mu_L)
                            vech_C_star_old_j .+= len_gamma .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                            # Set V_star_old_j = C * C'
                            BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
                        end
                    end
                end
            end
        end
    else
        @inbounds for k in 1:K
            for t in 1:O
                for z in 0:1
                    if t == 1 && z == 1
                        continue
                    end
                    Threads.@threads for s in 1:S
                        # Get thread id
                        tid = Threads.threadid()

                        mu_star_old_j = mu_star_old[k][t][z + 1][s]
                        V_star_old_j = V_star_old[k][t][z + 1][s]
                        # Perform gradient descent update of mu_j and V_j
                        len_gamma = length(mu_star_old_j)
                        # Assign storage for gradient terms
                        # Memory assigned from preallocated storage
                        # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
                        # (important for vectorized matricies to be strided if we want to use them for linear algebra)
                        # Matricies are stored column major in Julia, so memory is assigned by column left to right
                        grad_mu_L = view(model.storage_L_par[tid], 1:len_gamma)
                        grad_C_L = view(model.storage_LL2_par[tid], 1:len_gamma, 1:len_gamma)
                        vech_grad_C_L = view(grad_C_L, [len_gamma * (j - 1) + i for j in 1:len_gamma for i in j:len_gamma]) # Uses same memory as grad_C_L
                        grad_mu_log_q = view(model.storage_L2_par[tid], 1:len_gamma)
                        vec_grad_V_log_q = view(model.storage_LL3_par[tid], 1:len_gamma^2)
                        grad_V_log_q = reshape(vec_grad_V_log_q, len_gamma, len_gamma) # Uses same memory as vec_grad_V_log_q
                        # Assign storage for calculating intermediate terms for gradient
                        Vinv_star_old_j = view(model.storage_LL_par[tid], 1:len_gamma, 1:len_gamma)
                        gamma_minus_mu = view(model.storage_L3_par[tid], 1:len_gamma)
                        C_star_old_j = view(model.storage_C_par[tid], 1:len_gamma, 1:len_gamma)
                        vech_C_star_old_j = view(C_star_old_j, [len_gamma * (j - 1) + i for j in 1:len_gamma for i in j:len_gamma]) # Uses same memory as C_star_old_j
                        fill!(C_star_old_j, 0)
                        storage_kron_prod = view(model.storage_L2L2_par[tid], 1:len_gamma^2, 1:len_gamma^2)
                        storage_len_gamma_sqr = view(model.storage_Lsqr_par[tid], 1:len_gamma^2)
                        storage_len_gamma_sqr2 = view(model.storage_Lsqr2_par[tid], 1:len_gamma^2)
                        storage_gradC = view(model.storage_gradC_par[tid], 1:Int(len_gamma * (len_gamma + 1) / 2))
                        # Generate commutation and duplication matrix
                        comm_j = view(model.storage_comm_par[tid], 1:len_gamma^2, 1:len_gamma^2)
                        dup_j = view(model.storage_dup_par[tid], 1:len_gamma^2, 1:Int(len_gamma * (len_gamma + 1) / 2))
                        get_comm!(comm_j, len_gamma)
                        get_dup!(dup_j, len_gamma)
                        # Assign len_gamma by len_gamma identity matrix
                        I_j = view(model.I_LL, 1:len_gamma, 1:len_gamma)
                        # # Get step size iterator
                        # step_iterator = step_iterator_factory(init_step)
                        for iter in 1:maxiter
                            # Sample β from variational distribution
                            sample_γ(model, s, t, k, z)
                            fill!(grad_mu_L, 0)
                            fill!(grad_C_L, 0)
                            # Copy V* into storage
                            copy!(Vinv_star_old_j, V_star_old_j)
                            # Perform cholesky decomposition on V*
                            # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                            LAPACK.potrf!('L', Vinv_star_old_j)
                            # Calculate log|V_j| from diagonal of cholesky decomposition
                            logdet_V_j = 0
                            for b in 1:len_gamma
                                logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                            end
                            # Copy lower triangular cholesky factor into preallocated storage
                            for k in 1:len_gamma
                                for l in 1:k
                                    C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                                end
                            end
                            # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                            LAPACK.potri!('L', Vinv_star_old_j)
                            LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                            ELBO = 0

                            # Calculate the gradient estimate of the m-th sample
                            for m in 1:M
                                gamma_ktzsm = gamma_sample[k][t][z + 1][s][m]
                                fill!(grad_mu_log_q, 0)
                                # grad_mu_log_q = Vinv_star * β_jm
                                BLAS.gemv!('N', T(1), Vinv_star_old_j, gamma_ktzsm, T(1), grad_mu_log_q)
                                # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                                BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                                # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                                copy!(grad_V_log_q, Vinv_star_old_j)
                                BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                                # storage_kron_prod = I ⊗ C_j
                                collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                                # storage_len_gamma_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_gamma_sqr, 0))
                                # storage_kron_prod = C_j ⊗ I
                                collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                                # storage_len_gamma_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_gamma_sqr2, 0))
                                # storage_len_gamma_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), comm_j, storage_len_gamma_sqr, T(1), storage_len_gamma_sqr2)
                                # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                                BLAS.gemv!('T', T(1), dup_j, storage_len_gamma_sqr2, T(1), fill!(storage_gradC, 0))
                                # Calculate log(p(Y, β_(j)))
                                log_prob_Ygamma = 0

                                for i in 1:N
                                    if obs.group[i] != s
                                        continue
                                    end
                                    if t > 1
                                        prev_skill_profile = obs.skill_dict[argmax(Z_sample[i][t - 1][m])]
                                    end

                                    if t > 1 && prev_skill_profile[k] != z
                                        continue
                                    end

                                    skill_profile = obs.skill_dict[argmax(Z_sample[i][t][m])]
                                    log_prob_Ygamma += log(sigmoid((2*skill_profile[k] - 1) * 
                                        dot(gamma_sample[k][t][z + 1][s][m], obs.X[k][t][i, :])))
                                end
                               
                                num_features = length(gamma_ktzsm)
                                for feature in 1:num_features
                                    tau = tau_sample[k][t][z + 1][feature][m]
                                    omega = omega_sample[k][t][z + 1][feature][m]
                                    u = obs.U[k][t][s, :]
                                    log_prob_Ygamma += - 1/(2 * tau) * (gamma_ktzsm[feature] - dot(u, omega))^2
                                end

                                gamma_minus_mu .= gamma_ktzsm
                                gamma_minus_mu .-= mu_star_old_j
                                log_q = -len_gamma / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(gamma_minus_mu, grad_mu_log_q)
                                # Update average gradient
                                grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Ygamma - log_q)
                                vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Ygamma - log_q)
                                # Update ELBO estimator
                                ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Ygamma - log_q)
                            end

                            # Update mu and C with one step
                            step = init_step
                            if use_iter
                                step = step_iterator()
                            end
                            mu_star_old_j .+= sqrt(len_gamma) .* step .* grad_mu_L ./ norm(grad_mu_L)
                            vech_C_star_old_j .+= len_gamma .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                            # Set V_star_old_j = C * C'
                            BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
                        end
                    end
                end
            end
        end
    end
end

function update_normal_variational_distribution3(
    model       :: TDCModel;
    init_step   :: T=1e-3,
    step_iterator=get_robbins_monroe_iterator(init_step, 20),
    # step_iterator_factory=get_robbins_monroe_iterator,
    use_iter    :: Bool=false,
    tol         :: T=1e-6,
    maxiter     :: Int=100000,
    verbose     :: Bool=true
) where T <: AbstractFloat
    obs = model.obs
    Y, D = Array{T, 3}(obs.Y), Vector{Matrix{T}}(obs.D)
    gamma_sample, omega_sample, tau_sample = model.gamma_sample, model.omega_sample, model.tau_sample
    mu_star_old, V_star_old = model.mu_omega_star, model.V_omega_star
    N, J, L, O, K = size(Y, 1), size(Y, 3), size(D[1], 1), size(obs.Y, 2), size(obs.Q, 2)
    M = model.M
    # Fully update parameters of each γ using noisy gradients before moving to update parameters of next γ
    if !model.enable_parallel
        @inbounds for idx in Iterators.product(1:K, 1:O, 0:1)
            k, t, z = idx[1], idx[2], idx[3]
            if t == 1 && z == 1
                continue
            end
            num_features = size(obs.X[k][t], 2)
            for feature in 1:num_features
                mu_star_old_j = mu_star_old[k][t][z + 1][feature]
                V_star_old_j = V_star_old[k][t][z + 1][feature]
                # Perform gradient descent update of mu_j and V_j
                len_omega = length(mu_star_old_j)
                # Assign storage for gradient terms
                # Memory assigned from preallocated storage
                # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
                # (important for vectorized matricies to be strided if we want to use them for linear algebra)
                # Matricies are stored column major in Julia, so memory is assigned by column left to right
                grad_mu_L = view(model.storage_L, 1:len_omega)
                grad_C_L = view(model.storage_LL2, 1:len_omega, 1:len_omega)
                vech_grad_C_L = view(grad_C_L, [len_omega * (j - 1) + i for j in 1:len_omega for i in j:len_omega]) # Uses same memory as grad_C_L
                grad_mu_log_q = view(model.storage_L2, 1:len_omega)
                vec_grad_V_log_q = view(model.storage_LL3, 1:len_omega^2)
                grad_V_log_q = reshape(vec_grad_V_log_q, len_omega, len_omega) # Uses same memory as vec_grad_V_log_q
                # Assign storage for calculating intermediate terms for gradient
                Vinv_star_old_j = view(model.storage_LL, 1:len_omega, 1:len_omega)
                omega_minus_mu = view(model.storage_L3, 1:len_omega)
                L_omega_minus_mu = view(model.storage_L4, 1:len_omega)
                C_star_old_j = view(model.storage_C, 1:len_omega, 1:len_omega)
                vech_C_star_old_j = view(C_star_old_j, [len_omega * (j - 1) + i for j in 1:len_omega for i in j:len_omega]) # Uses same memory as C_star_old_j
                fill!(C_star_old_j, 0)
                storage_kron_prod = view(model.storage_L2L2, 1:len_omega^2, 1:len_omega^2)
                storage_len_omega_sqr = view(model.storage_Lsqr, 1:len_omega^2)
                storage_len_omega_sqr2 = view(model.storage_Lsqr2, 1:len_omega^2)
                storage_gradC = view(model.storage_gradC, 1:Int(len_omega * (len_omega + 1) / 2))
                # Generate commutation and duplication matrix
                comm_j = view(model.storage_comm, 1:len_omega^2, 1:len_omega^2)
                dup_j = view(model.storage_dup, 1:len_omega^2, 1:Int(len_omega * (len_omega + 1) / 2))
                get_comm!(comm_j, len_omega)
                get_dup!(dup_j, len_omega)
                # Assign len_omega by len_omega identity matrix
                I_j = view(model.I_LL, 1:len_omega, 1:len_omega)
                # # Get step size iterator
                # step_iterator = step_iterator_factory(init_step)
                for iter in 1:maxiter
                    # Sample β from variational distribution
                    sample_ω(model, k, t, z, feature)
                    fill!(grad_mu_L, 0)
                    fill!(grad_C_L, 0)
                    # Copy V* into storage
                    copy!(Vinv_star_old_j, V_star_old_j)
                    # Perform cholesky decomposition on V*
                    # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                    LAPACK.potrf!('L', Vinv_star_old_j)
                    # Calculate log|V_j| from diagonal of cholesky decomposition
                    logdet_V_j = 0
                    for b in 1:len_omega
                        logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                    end
                    # Copy lower triangular cholesky factor into preallocated storage
                    for k in 1:len_omega
                        for l in 1:k
                            C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                        end
                    end
                    # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                    LAPACK.potri!('L', Vinv_star_old_j)
                    LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                    ELBO = 0
                    # Calculate the gradient estimate of the m-th sample
                    for m in 1:M
                        omega_ktzm = omega_sample[k][t][z + 1][feature][m]
                        fill!(grad_mu_log_q, 0)
                        # grad_mu_log_q = Vinv_star * β_jm
                        BLAS.gemv!('N', T(1), Vinv_star_old_j, omega_ktzm, T(1), grad_mu_log_q)
                        # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                        BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                        # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                        copy!(grad_V_log_q, Vinv_star_old_j)
                        BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                        # storage_kron_prod = I ⊗ C_j
                        collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                        # storage_len_omega_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_omega_sqr, 0))
                        # storage_kron_prod = C_j ⊗ I
                        collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                        # storage_len_omega_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_omega_sqr2, 0))
                        # storage_len_omega_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), comm_j, storage_len_omega_sqr, T(1), storage_len_omega_sqr2)
                        # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), dup_j, storage_len_omega_sqr2, T(1), fill!(storage_gradC, 0))
                        # Calculate log(p(Y, β_(j)))
                        log_prob_Yomega = 0
                        for s in 1:S
                            tau = tau_sample[k][t][z + 1][feature][m]
                            gamma = gamma_sample[k][t][z + 1][s][m][feature]
                            u = obs.U[k][t][s, :]
                            log_prob_Yomega += - 1/(2 * tau) * (gamma - dot(u, omega_ktzm))^2
                        end
                        omega_minus_mu .= omega_ktzm
                        omega_minus_mu .-= model.mu_omega_prior[k][t][z + 1][feature]
                        mul!(L_omega_minus_mu, model.L_omega_prior[k][t][z + 1][feature], omega_minus_mu)
                        log_prob_Yomega -= 1/2 * dot(L_omega_minus_mu, L_omega_minus_mu)
                        omega_minus_mu .= omega_ktzm
                        omega_minus_mu .-= mu_star_old_j
                        log_q = -len_omega / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(omega_minus_mu, grad_mu_log_q)
                        # Update average gradient
                        grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Yomega - log_q)
                        vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Yomega - log_q)
                        # Update ELBO estimator
                        ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Yomega - log_q)
                    end
                    # Print ELBO, parameter and gradient if verbose
                    if verbose
                        println("ELBO: $ELBO")
                        # println("mu*_$j: mu_star_old_j")
                        # println("gradient mu: $grad_mu_L")
                        println("C*_$k$t$z$feature: $C_star_old_j")
                        println("gradient C: $grad_C_L")
                    end
                    # Update mu and C with one step
                    step = init_step
                    if use_iter
                        step = step_iterator()
                    end
                    mu_star_old_j .+= sqrt(len_omega) .* step .* grad_mu_L ./ norm(grad_mu_L)
                    vech_C_star_old_j .+= len_omega .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                    # Set V_star_old_j = C * C'
                    BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
                end
            end
        end
    else
        Threads.@threads for idx in collect(Iterators.product(1:K, 1:O, 0:1))
            k, t, z = idx[1], idx[2], idx[3]
            if t == 1 && z == 1
                continue
            end
            num_features = size(obs.X[k][t], 2)
            for feature in 1:num_features
                # Get thread id
                tid = Threads.threadid()

                mu_star_old_j = mu_star_old[k][t][z + 1][feature]
                V_star_old_j = V_star_old[k][t][z + 1][feature]
                # Perform gradient descent update of mu_j and V_j
                len_omega = length(mu_star_old_j)
                # Assign storage for gradient terms
                # Memory assigned from preallocated storage
                # Memory has to be strided (equal stride between memory addresses) to work with BLAS and LAPACK 
                # (important for vectorized matricies to be strided if we want to use them for linear algebra)
                # Matricies are stored column major in Julia, so memory is assigned by column left to right
                grad_mu_L = view(model.storage_L_par[tid], 1:len_omega)
                grad_C_L = view(model.storage_LL2_par[tid], 1:len_omega, 1:len_omega)
                vech_grad_C_L = view(grad_C_L, [len_omega * (j - 1) + i for j in 1:len_omega for i in j:len_omega]) # Uses same memory as grad_C_L
                grad_mu_log_q = view(model.storage_L2_par[tid], 1:len_omega)
                vec_grad_V_log_q = view(model.storage_LL3_par[tid], 1:len_omega^2)
                grad_V_log_q = reshape(vec_grad_V_log_q, len_omega, len_omega) # Uses same memory as vec_grad_V_log_q
                # Assign storage for calculating intermediate terms for gradient
                Vinv_star_old_j = view(model.storage_LL_par[tid], 1:len_omega, 1:len_omega)
                omega_minus_mu = view(model.storage_L3_par[tid], 1:len_omega)
                L_omega_minus_mu = view(model.storage_L4_par[tid], 1:len_omega)
                C_star_old_j = view(model.storage_C_par[tid], 1:len_omega, 1:len_omega)
                vech_C_star_old_j = view(C_star_old_j, [len_omega * (j - 1) + i for j in 1:len_omega for i in j:len_omega]) # Uses same memory as C_star_old_j
                fill!(C_star_old_j, 0)
                storage_kron_prod = view(model.storage_L2L2_par[tid], 1:len_omega^2, 1:len_omega^2)
                storage_len_omega_sqr = view(model.storage_Lsqr_par[tid], 1:len_omega^2)
                storage_len_omega_sqr2 = view(model.storage_Lsqr2_par[tid], 1:len_omega^2)
                storage_gradC = view(model.storage_gradC_par[tid], 1:Int(len_omega * (len_omega + 1) / 2))
                # Generate commutation and duplication matrix
                comm_j = view(model.storage_comm_par[tid], 1:len_omega^2, 1:len_omega^2)
                dup_j = view(model.storage_dup_par[tid], 1:len_omega^2, 1:Int(len_omega * (len_omega + 1) / 2))
                get_comm!(comm_j, len_omega)
                get_dup!(dup_j, len_omega)
                # Assign len_omega by len_omega identity matrix
                I_j = view(model.I_LL, 1:len_omega, 1:len_omega)
                # # Get step size iterator
                # step_iterator = step_iterator_factory(init_step)
                for iter in 1:maxiter
                    # Sample β from variational distribution
                    sample_ω(model, k, t, z, feature)
                    fill!(grad_mu_L, 0)
                    fill!(grad_C_L, 0)
                    # Copy V* into storage
                    copy!(Vinv_star_old_j, V_star_old_j)
                    # Perform cholesky decomposition on V*
                    # After this step, the lower triangle of Vinv_star_old_j will contain the lower triangular cholesky factor of V*
                    LAPACK.potrf!('L', Vinv_star_old_j)
                    # Calculate log|V_j| from diagonal of cholesky decomposition
                    logdet_V_j = 0
                    for b in 1:len_omega
                        logdet_V_j += 2 * log(Vinv_star_old_j[b, b])
                    end
                    # Copy lower triangular cholesky factor into preallocated storage
                    for k in 1:len_omega
                        for l in 1:k
                            C_star_old_j[k, l] = Vinv_star_old_j[k, l]
                        end
                    end
                    # Perform in place matrix inverse on positive definite V* matrix to get V* inverse
                    LAPACK.potri!('L', Vinv_star_old_j)
                    LinearAlgebra.copytri!(Vinv_star_old_j, 'L')
                    ELBO = 0
                    # Calculate the gradient estimate of the m-th sample
                    for m in 1:M
                        omega_ktzm = omega_sample[k][t][z + 1][feature][m]
                        fill!(grad_mu_log_q, 0)
                        # grad_mu_log_q = Vinv_star * β_jm
                        BLAS.gemv!('N', T(1), Vinv_star_old_j, omega_ktzm, T(1), grad_mu_log_q)
                        # grad_mu_log_q = Vinv_star_j * β_jm - Vinv_star_j * mu_star_j
                        BLAS.gemv!('N', T(-1), Vinv_star_old_j, mu_star_old_j, T(1), grad_mu_log_q)
                        # grad_V_log_q = -1/2(Vinv_star_j - Vinv_star_j * (β_jm - mu_star_j) * (β_jm - mu_star_j)^T * Vinv_star_j)
                        copy!(grad_V_log_q, Vinv_star_old_j)
                        BLAS.gemm!('N', 'T', T(1 / 2), grad_mu_log_q, grad_mu_log_q, T(-1 / 2), grad_V_log_q)
                        # storage_kron_prod = I ⊗ C_j
                        collect!(storage_kron_prod, kronecker(I_j, C_star_old_j))
                        # storage_len_omega_sqr = (I ⊗ C_j)'vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_omega_sqr, 0))
                        # storage_kron_prod = C_j ⊗ I
                        collect!(storage_kron_prod, kronecker(C_star_old_j, I_j))
                        # storage_len_omega_sqr2 = (C_j ⊗ I)'vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), storage_kron_prod, vec_grad_V_log_q, T(1), fill!(storage_len_omega_sqr2, 0))
                        # storage_len_omega_sqr2 = ((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), comm_j, storage_len_omega_sqr, T(1), storage_len_omega_sqr2)
                        # storage_gradC = D'((C_j ⊗ I)' + K'(I ⊗ C_j)')vec(grad_V_log_q)
                        BLAS.gemv!('T', T(1), dup_j, storage_len_omega_sqr2, T(1), fill!(storage_gradC, 0))
                        # Calculate log(p(Y, β_(j)))
                        log_prob_Yomega = 0
                        for s in 1:S
                            tau = tau_sample[k][t][z + 1][feature][m]
                            gamma = gamma_sample[k][t][z + 1][s][m][feature]
                            u = obs.U[k][t][s, :]
                            log_prob_Yomega += - 1/(2 * tau) * (gamma - dot(u, omega_ktzm))^2
                        end
                        omega_minus_mu .= omega_ktzm
                        omega_minus_mu .-= model.mu_omega_prior[k][t][z + 1][feature]
                        mul!(L_omega_minus_mu, model.L_omega_prior[k][t][z + 1][feature], omega_minus_mu)
                        log_prob_Yomega -= 1/2 * dot(L_omega_minus_mu, L_omega_minus_mu)
                        omega_minus_mu .= omega_ktzm
                        omega_minus_mu .-= mu_star_old_j
                        log_q = -len_omega / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(omega_minus_mu, grad_mu_log_q)
                        # Update average gradient
                        grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Yomega - log_q)
                        vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Yomega - log_q)
                        # Update ELBO estimator
                        ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Yomega - log_q)
                    end
                    # Print ELBO, parameter and gradient if verbose
                    if verbose
                        println("ELBO: $ELBO")
                        # println("mu*_$j: mu_star_old_j")
                        # println("gradient mu: $grad_mu_L")
                        println("C*_$k$t$z$feature: $C_star_old_j")
                        println("gradient C: $grad_C_L")
                    end
                    # Update mu and C with one step
                    step = init_step
                    if use_iter
                        step = step_iterator()
                    end
                    mu_star_old_j .+= sqrt(len_omega) .* step .* grad_mu_L ./ norm(grad_mu_L)
                    vech_C_star_old_j .+= len_omega .* step .* vech_grad_C_L ./ norm(vech_grad_C_L)
                    # Set V_star_old_j = C * C'
                    BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
                end
            end
        end
    end
end

function update_inverse_gamma_distribution(
    model           :: TDCModel;
    step            :: T,
    tol             :: T = 1e-6,
    maxiter         :: Int = 100000,
    verbose         :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    O, K, S = size(obs.Y, 2), size(obs.Q, 2), size(obs.U[1][1], 1)
    tau_sample, gamma_sample, omega_sample = model.tau_sample, model.gamma_sample, model.omega_sample
    M = model.M
    # Perform gradient ascent
    if !model.enable_parallel
        @inbounds for idx in Iterators.product(1:K, 1:O, 0:1)
            k, t, z = idx[1], idx[2], idx[3]
            if t == 1 && z == 1
                continue
            end
            num_features = size(obs.X[k][t], 2)
            for feature in 1:num_features
                for iter in 1:maxiter
                    # Sample sigma^2 from variational distribution
                    sample_τ(model, k, t, z, feature)
                    # Variable for tracking ELBO approximation
                    ELBO = 0
                    # Variables for tracking gradient
                    # grad_a_L = 0
                    grad_b_L = 0
                    # Terms used in gradient calculation
                    a_star = model.a_tau_star[k][t][z + 1][feature]
                    b_star = model.b_tau_star[k][t][z + 1][feature]
                    # log_a = log(a_star)
                    log_b = log(b_star)
                    loggamma_a = loggamma(a_star)
                    # digamma_a = digamma(a_star)
                    for m in 1:M
                        tau_ktzm = tau_sample[k][t][z + 1][feature][m]
                        log_q = a_star * log_b - loggamma_a - (a_star + 1) * log(tau_ktzm) - b_star/tau_ktzm
                        # Not actually gradient wrt to a and b, but gradient wrt log(a) and log(b)
                        # grad_a_log_q = (log_b - digamma_a - log(tau_ktzm)) * a_star
                        grad_b_log_q = (a_star/b_star - 1/tau_ktzm) * b_star
                        a_prior = model.a_tau_prior[k][t][z + 1][feature]
                        b_prior = model.b_tau_prior[k][t][z + 1][feature]
                        log_prob_Ytau = -(a_prior - 1) * log(tau_ktzm) - b_prior / tau_ktzm
                        for s in 1:S
                            gamma = gamma_sample[k][t][z + 1][s][m][feature]
                            omega = omega_sample[k][t][z + 1][feature][m]
                            u = obs.U[k][t][s, :]
                            log_prob_Ytau += -1/2 * log(tau_ktzm) - 1/(2 * tau_ktzm) * (gamma - dot(u, omega))^2
                        end
                        # Update Gradient estimators
                        # grad_a_L = (m-1)/m * grad_a_L + 1/m * grad_a_log_q * (log_prob_Ytau - log_q)
                        grad_b_L = (m-1)/m * grad_b_L + 1/m * grad_b_log_q * (log_prob_Ytau - log_q)
                        # Update ELBO estimator
                        ELBO = (m-1)/m * ELBO + 1/m * (log_prob_Ytau - log_q)
                    end
                    # Print ELBO, parameter and gradient if verbose
                    if verbose
                        println("ELBO: $ELBO")
                        # println("a*: $a_star")
                        # println("gradient log(a*): $grad_a_L")
                        println("b*: $b_star")
                        println("gradient log(b*): $grad_b_L")
                    end
                    
                    #TODO: Stop condition

                    # Update parameters
                    # log_a += step * grad_a_L
                    log_b += step * grad_b_L
                    # model.a_tau_star[k][t][z + 1][feature] = exp(log_a)
                    model.b_tau_star[k][t][z + 1][feature] = exp(log_b)
                end
            end
        end
    else
        Threads.@threads for idx in collect(Iterators.product(1:K, 1:O, 0:1))
            k, t, z = idx[1], idx[2], idx[3]
            if t == 1 && z == 1
                continue
            end
            num_features = size(obs.X[k][t], 2)
            @inbounds for feature in 1:num_features
                for iter in 1:maxiter
                    # Sample sigma^2 from variational distribution
                    sample_τ(model, k, t, z, feature)
                    # Variable for tracking ELBO approximation
                    ELBO = 0
                    # Variables for tracking gradient
                    # grad_a_L = 0
                    grad_b_L = 0
                    # Terms used in gradient calculation
                    a_star = model.a_tau_star[k][t][z + 1][feature]
                    b_star = model.b_tau_star[k][t][z + 1][feature]
                    log_a = log(a_star)
                    log_b = log(b_star)
                    loggamma_a = loggamma(a_star)
                    # digamma_a = digamma(a_star)
                    for m in 1:M
                        tau_ktzm = tau_sample[k][t][z + 1][feature][m]
                        log_q = a_star * log_b - loggamma_a - (a_star + 1) * log(tau_ktzm) - b_star/tau_ktzm
                        # Not actually gradient wrt to a and b, but gradient wrt log(a) and log(b)
                        # grad_a_log_q = (log_b - digamma_a - log(tau_ktzm)) * a_star
                        grad_b_log_q = (a_star/b_star - 1/tau_ktzm) * b_star
                        a_prior = model.a_tau_prior[k][t][z + 1][feature]
                        b_prior = model.b_tau_prior[k][t][z + 1][feature]
                        log_prob_Ytau = -(a_prior + 1) * log(tau_ktzm) - b_prior / tau_ktzm
                        omega = omega_sample[k][t][z + 1][feature][m]
                        for s in 1:S
                            gamma = gamma_sample[k][t][z + 1][s][m][feature]
                            u = obs.U[k][t][s, :]
                            log_prob_Ytau += -1/2 * log(tau_ktzm) - 1/(2 * tau_ktzm) * (gamma - dot(u, omega))^2
                        end
                        # Update Gradient estimators
                        # grad_a_L = (m-1)/m * grad_a_L + 1/m * grad_a_log_q * (log_prob_Ytau - log_q)
                        grad_b_L = (m-1)/m * grad_b_L + 1/m * grad_b_log_q * (log_prob_Ytau - log_q)
                        # Update ELBO estimator
                        ELBO = (m-1)/m * ELBO + 1/m * (log_prob_Ytau - log_q)
                    end
                    # Print ELBO, parameter and gradient if verbose
                    if verbose
                        println("ELBO: $ELBO")
                        # println("a*: $a_star")
                        # println("gradient log(a*): $grad_a_L")
                        println("b*: $b_star")
                        println("gradient log(b*): $grad_b_L")
                    end
                    
                    #TODO: Stop condition

                    # Update parameters
                    # log_a += step * grad_a_L
                    log_b += step * grad_b_L
                    # model.a_tau_star[k][t][z + 1][feature] = exp(log_a)
                    model.b_tau_star[k][t][z + 1][feature] = exp(log_b)
                end
            end
        end
    end
end

function update_inverse_wishart_distribution(
    model       :: TDCModel,
    step        :: T,
    maxiter     :: Int
) where T <: AbstractFloat

    
end
;