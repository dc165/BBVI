using LinearAlgebra, Distributions, Combinatorics, Random, Kronecker, SpecialFunctions
include("BBVI_utils.jl")

# Object for observed data
struct DCMObs{T <: Int}
    # data
    Y          :: Matrix{T}
    Q          :: Matrix{T}
    D          :: Vector{Matrix{T}}
end

function DCMObs(
    Y::Matrix{T}, 
    Q::Matrix{T}) where T <: Int
    D = generate_delta(Q)
    DCMObs(Y, Q, D)
end

# Object for model and samples
# TODO: Allow custom initialization

struct DCModel{T <: AbstractFloat}
    # data
    obs             :: DCMObs
    # Prior distribution parameters
    d0              :: Vector{T}
    a0              :: T
    b0              :: T
    # Variational distribution parameters
    pi_star         :: Vector{Vector{T}}
    mu_star         :: Vector{Vector{T}}
    V_star          :: Vector{Matrix{T}}
    d_star          :: Vector{T}
    a_star          :: Vector{T}
    b_star          :: Vector{T}
    # Number of samples for noisy gradient
    M               :: Int
    # Preallocated storage for samples from variational distribution
    Z_sample        :: Vector{Vector{Vector{Int}}}
    beta_sample     :: Vector{Vector{Vector{T}}}
    pi_sample       :: Vector{Vector{T}}
    sigma2_sample   :: Vector{T}
    # Preallocated storage for noisy gradient descent calculations
    storage_L       :: Vector{T}
    storage_L2      :: Vector{T}
    storage_L3      :: Vector{T}
    storage_L4      :: Vector{T}
    storage_LL      :: Matrix{T}
    storage_LL2     :: Matrix{T}
    storage_LL3     :: Matrix{T}
    storage_LL4     :: Matrix{T}
    # Preallocated storage for matrix vectorization operations
    storage_comm    :: Matrix{T}
    storage_dup     :: Matrix{T}
    storage_Lsqr    :: Vector{T}
    storage_Lsqr2   :: Vector{T}
    storage_L2L2    :: Matrix{T}
    storage_C       :: Matrix{T}
    storage_gradC   :: Vector{T}
    # Preallocated Identity matrix
    I_LL            :: Matrix{T}
end

function DCModel(
    obs             :: DCMObs,
    d0              :: Vector{T},
    a0              :: T,
    b0              :: T,
    M               :: Int) where T <: AbstractFloat
    N, J, L = size(obs.Y, 1), size(obs.Y, 2), size(obs.D[1], 1)
    # Initialize variational distribution parameters
    pi_star = Vector{Vector{T}}(undef, N)
    for i in 1:N
        pi_star[i] = ones(L) ./ L
    end
    mu_star = Vector{Vector{T}}(undef, J)
    V_star = Vector{Matrix{T}}(undef, J)
    for j in 1:J
        num_features = size(obs.D[j], 2)
        mu_star[j] = zeros(num_features)
        V_star[j] = Matrix(1.0I, num_features, num_features)
    end
    d_star = ones(L) ./ L
    a_star = [3.0]
    b_star = [3.0]
    # Preallocate space for samples from variational distribution
    Z_sample = Vector{Vector{Vector{Int}}}(undef, N)
    for i in 1:N
        Z_sample[i] = Vector{Vector{Int}}(undef, M)
        for m in 1:M
            Z_sample[i][m] = Vector{Int}(undef, L)
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
    pi_sample = Vector{Vector{T}}(undef, M)
    for m in 1:M
        pi_sample[m] = Vector{T}(undef, L)
    end
    sigma2_sample = Vector{T}(undef, M)
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
    # Initialize DCModel object
    DCModel(obs, d0, a0, b0, 
    pi_star, mu_star, V_star, d_star, a_star, b_star, M,
    Z_sample, beta_sample, pi_sample, sigma2_sample,
    storage_L, storage_L2, storage_L3, storage_L4, storage_LL, storage_LL2, storage_LL3, storage_LL4,
    storage_comm, storage_dup, storage_Lsqr, storage_Lsqr2, storage_L2L2, storage_C, storage_gradC,
    I_LL)
end

# Function for sampling variational distribution
function sample_variational_distribution(
    model           :: DCModel;
    sample_Z        :: Bool = false,
    idx_Z           :: Int = -1,
    sample_β        :: Bool = false,
    idx_β           :: Int = -1,
    sample_pi       :: Bool = false,
    sample_sigma2   :: Bool = false)
    M = model.M
    obs = model.obs
    N, J, L = size(obs.Y, 1), size(obs.Y, 2), size(obs.D[1], 1)
    if sample_Z
        if idx_Z == -1
            for i in 1:N
                # Create variational distribution from model parameters for each Z_i
                Z_i_variational_distribution = Multinomial(1, model.pi_star[i])
                # Populate preallocated arrays with samples from variational distribution
                rand!(Z_i_variational_distribution, model.Z_sample[i])
            end
        else
            # Create variational distribution from model parameters for specific Z
            Z_i_variational_distribution = Multinomial(1, model.pi_star[idx_Z])
            # Populate preallocated arrays with samples from variational distribution
            rand!(Z_i_variational_distribution, model.Z_sample[idx_Z])
        end
    end
    if sample_β
        if idx_β == -1
            for j in 1:J
                # Create variational distribution from model parameters for each β_j
                beta_j_variational_distribution = MvNormal(model.mu_star[j], model.V_star[j])
                # Populate preallocated arrays with samples from variational distribution
                rand!(beta_j_variational_distribution, model.beta_sample[j])
            end
        else
            # Create variational distribution from model parameters for specific β_j
            beta_j_variational_distribution = MvNormal(model.mu_star[idx_β], model.V_star[idx_β])
            # Populate preallocated arrays with samples from variational distribution
            rand!(beta_j_variational_distribution, model.beta_sample[idx_β])
        end
    end
    if sample_pi
        # Create variational distribution from model parameters for pi
        pi_variational_distribution = Dirichlet(model.d_star)
        # Populate preallocated arrays with samples from variational distribution
        rand!(pi_variational_distribution, model.pi_sample)
    end
    if sample_sigma2
        # Create variational distribution from model parameters for σ²
        sigma2_variational_distribution = InverseGamma(model.a_star[1], model.b_star[1])
        # Populate preallocated array with samples from variational distribution
        rand!(sigma2_variational_distribution, model.sigma2_sample)
    end
    nothing
end

"""
    update_pi_star(Y, D, Z_sample, beta_sample, pi_sample, pi_star_old, 
                    step, tol = 1e-6, maxiter = 100000, verbose = true)

Find the parameters of the variational distribution of Z_i that maximize 
the ELBO via gradient descent. The gradient is an expectation which is 
appoximated from the inputs `Z_sample`, `beta_sample`, and `pi_sample` 
which are sampled from the variational distribution. The same sample is 
used for every gradient descent update for the sake of computational 
efficiency. `pi_star_old` are the parameters that are updated in place.

"""
function update_pi_star(
    model           :: DCModel;
    step            :: T = 1e-2,
    tol             :: T = 1e-6,
    maxiter         :: Int = 100000,
    verbose         :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    Y, D = obs.Y, obs.D
    Z_sample, beta_sample, pi_sample = model.Z_sample, model.beta_sample, model.pi_sample
    pi_star_old = model.pi_star
    N, J, L = size(Y, 1), size(Y, 2), size(D[1], 1)
    M = model.M
    # Sample Z, β, and pi from variational distribution. Only samples of Z will be updated as the parameters update
    sample_variational_distribution(model, sample_Z = true, sample_β = true, sample_pi = true)
    # Fully update parameters of each Z_i using noisy gradients before moving to update parameters of next Z_i
    @inbounds for i in 1:N
        # Storage for gradient terms
        grad_log_q = model.storage_L2
        grad_L = model.storage_L3
        # Storage for intermediate term in gradient calculations
        D_beta = model.storage_L
        rho_star_old_i = view(model.storage_LL3, 1:L)
        # Get parameters for variational distribution of skill of i-th student
        pi_star_old_i = pi_star_old[i]
        # Perform gradient descent update of i-th π*    
        @inbounds for iter in 1:maxiter
            # Rho is unique up to a constant addative term
            rho_star_old_i = log.(pi_star_old_i)
            # Sample Z with updated π*
            sample_variational_distribution(model, sample_Z = true, idx_Z = i)
            # Set gradient of ELBO to 0
            fill!(grad_L, 0)
            # Rao Blackwellized ELBO
            ELBO = 0
            # Calculate the gradient estimate of the m-th sample
            @inbounds for m in 1:M
                z_im = Z_sample[i][m]
                # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                grad_log_q .= z_im .- pi_star_old_i
                # Calculate log(p(Y, Z_(i)))
                log_prob_YZ = 0
                for j in 1:J
                    mul!(D_beta, D[j], beta_sample[j][m])
                    log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i,j] - 1) .* D_beta)))
                end
                log_prob_YZ += dot(z_im, log.(pi_sample[m]))
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

# Gradient ascent of variational distribution of β
# TODO: Investigate why gradient explodes after convergence for small sample sizes
function update_mu_star_V_star(
    model       :: DCModel;
    init_step   :: T=1e-3,
    step_iterator=get_robbins_monroe_iterator(init_step),
    use_iter    :: Bool=false,
    tol         :: T=1e-6,
    maxiter     :: Int=100000,
    verbose     :: Bool=true
) where T <: AbstractFloat
    obs = model.obs
    Y, D = Matrix{T}(obs.Y), Vector{Matrix{T}}(obs.D)
    Z_sample, beta_sample, sigma2_sample = model.Z_sample, model.beta_sample, model.sigma2_sample
    mu_star_old, V_star_old = model.mu_star, model.V_star
    N, J, L = size(Y, 1), size(Y, 2), size(D[1], 1)
    M = model.M
    # Sample Z, β, and sigma^2. Only β samples will update as the parameters update
    sample_variational_distribution(model, sample_Z=true, sample_β=true, sample_sigma2=true)
    # Fully update parameters of each β_j using noisy gradients before moving to update parameters of next β_j
    @inbounds for j in 1:J
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
        # # Initialize variables for tracking previous values
        # prev_ELBO = -Inf
        # prev_mu = view(model.storage_L4, 1:len_beta)
        # prev_V = view(model.storage_LL4, 1:len_beta, 1:len_beta)
        # prev_mu .= mu_star_old[j]
        # prev_V .= V_star_old[j]
        @inbounds for iter in 1:maxiter
            # Sample β from variational distribution
            sample_variational_distribution(model, sample_β=true, idx_β=j)
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
            # Calculate the gradient estimate of the m-th sample
            @inbounds for m in 1:M
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
                    fill!(model.storage_L3, 0)
                    BLAS.gemv!('N', (2 * Y[i, j] - 1), D[j], beta_jm, T(1), model.storage_L3)
                    log_prob_Ybeta += dot(Z_sample[i][m], log.(sigmoid.(model.storage_L3)))
                end
                log_prob_Ybeta -= 1 / (2 * sigma2_sample[m]) * dot(beta_jm, beta_jm)
                beta_minus_mu .= beta_jm
                beta_minus_mu .-= mu_star_old_j
                log_q = -len_beta / 2 * log(2 * pi) - 1 / 2 * logdet_V_j - 1 / 2 * dot(beta_minus_mu, grad_mu_log_q)
                # Update average gradient
                grad_mu_L .= (m - 1) / m .* grad_mu_L + 1 / m .* grad_mu_log_q .* (log_prob_Ybeta - log_q)
                vech_grad_C_L .= (m - 1) / m .* vech_grad_C_L + 1 / m .* storage_gradC .* (log_prob_Ybeta - log_q)
                # Update ELBO estimator
                ELBO = (m - 1) / m * ELBO + 1 / m * (log_prob_Ybeta - log_q)
            end
            # Print ELBO, parameter and gradient if verbose
            # if verbose
            #     println("ELBO: $ELBO")
            #     println("mu*_$j: $mu_star_old_j")
            #     println("gradient mu: $grad_mu_L")
            #     println("C*_$j: $C_star_old_j")
            #     println("gradient C: $grad_C_L")
            # end
            # Stop condition TODO: update to more appropriate stop condition
            # if abs2(norm(vech_C_star_old_j)) > 1e6
            #     break
            # end
            # # If ELBO decreases, go to previous step
            # if ELBO < prev_ELBO
            #     mu_star_old_j .= prev_mu
            #     V_star_old_j .= prev_V
            # else
            #     # Save current values
            #     prev_mu .= mu_star_old_j
            #     prev_V .= V_star_old_j
            #     prev_ELBO = ELBO
            #     # Update mu and C with one step
            #     mu_star_old_j .+= step .* grad_mu_L
            #     vech_C_star_old_j .+= step .* vech_grad_C_L
            #     # Set V_star_old_j = C * C'
            #     BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
            # end
            # Update mu and C with one step
            step = init_step
            if use_iter
                step = step_iterator()
            end
            mu_star_old_j .+= step .* grad_mu_L ./ norm(grad_mu_L)
            vech_C_star_old_j .+= step .* vech_grad_C_L ./ norm(vech_grad_C_L)
            # Set V_star_old_j = C * C'
            BLAS.gemm!('N', 'T', T(1), C_star_old_j, C_star_old_j, T(1), fill!(V_star_old_j, 0))
        end
    end
end

# Gradient ascent of variational distribution of pi
function update_d_star(
    model           :: DCModel;
    step            :: T,
    tol             :: T = 1e-6,
    maxiter         :: Int = 100000,
    verbose         :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    N, L = size(obs.Y, 1), size(obs.D[1], 1)
    Z_sample, pi_sample = model.Z_sample, model.pi_sample
    M = model.M
    # Sample Z from variational distribution. Only samples of pi will be resampled as the parameters update
    sample_variational_distribution(model, sample_Z = true)
    # Parameters being updated
    d_star_old = model.d_star
    # Storage for gradient terms
    grad_log_q = model.storage_L
    grad_L = model.storage_L2
    # Storage for intermediate terms for gradient calculation
    log_pi_m = model.storage_L3
    gamma_star_old = view(model.storage_LL, 1:L)
    # Perform gradient ascent updates
    @inbounds for iter in 1:maxiter
        # Reset gradients
        fill!(grad_log_q, 0)
        fill!(grad_L, 0)
        # Sample pi from variational distribution
        sample_variational_distribution(model, sample_pi = true)
        sum_d = sum(d_star_old)
        # Variable for tracking ELBO approximation
        ELBO = 0
        @inbounds for m in 1:M
            # Calculate log q(pi)
            log_q = loggamma(sum_d)
            for l in 1:L
                log_q -= loggamma(d_star_old[l])
                log_q += (d_star_old[l] - 1) * log(pi_sample[m][l])
            end
            # Calculate gradient of log q(pi) with respect to γ
            grad_log_q .= (digamma(sum_d) .- digamma.(d_star_old) .+ log.(pi_sample[m])) .* d_star_old
            # Calculate log(P(Y, pi))
            log_prob_Ypi = 0
            log_pi_m .= log.(pi_sample[m])
            @inbounds for i in 1:N
                log_prob_Ypi += dot(Z_sample[i][m], log_pi_m)
            end
            log_prob_Ypi += dot(model.d0, log_pi_m) - sum(log_pi_m)
            # Update average gradient
            grad_L .= (m - 1)/m .* grad_L + 1/m .* grad_log_q .* (log_prob_Ypi - log_q)
            # Update ELBO estimator
            ELBO = (m-1)/m * ELBO + 1/m * (log_prob_Ypi - log_q)
        end
        # Print ELBO, parameter and gradient if verbose
        if verbose
            println("ELBO: $ELBO")
            println("d*: $d_star_old")
            println("gradient: $grad_L")
        end
        # TODO: Stop condition

        # Update d* with one step
        gamma_star_old .= log.(d_star_old)
        gamma_star_old .+= step .* grad_L
        d_star_old .= exp.(gamma_star_old)
    end
end

# Gradient ascent of variational distribution of sigma squared
function update_a_star_b_star(
    model           :: DCModel;
    step            :: T,
    tol             :: T = 1e-6,
    maxiter         :: Int = 100000,
    verbose         :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    J, L = size(obs.Y, 2), size(obs.D[1], 1)
    beta_sample, sigma2_sample = model.beta_sample, model.sigma2_sample
    M = model.M
    # Sample β from variational distribution. Only samples of sigma^2 will be resampled as the parameters update
    sample_variational_distribution(model, sample_β = true)
    # Perform gradient ascent
    @inbounds for iter in 1:maxiter
        # Sample sigma^2 from variational distribution
        sample_variational_distribution(model, sample_sigma2 = true)
        # Variable for tracking ELBO approximation
        ELBO = 0
        # Variables for tracking gradient
        grad_a_L = 0
        grad_b_L = 0
        # Terms used in gradient calculation
        a_star = model.a_star[1]
        b_star = model.b_star[1]
        log_a = log(a_star)
        log_b = log(b_star)
        loggamma_a = loggamma(a_star)
        digamma_a = digamma(a_star)
        @inbounds for m in 1:M
            log_q = a_star * log_b - loggamma_a - (a_star + 1) * log(sigma2_sample[m]) - b_star/sigma2_sample[m]
            # Not actually gradient wrt to a and b, but gradient wrt log(a) and log(b)
            grad_a_log_q = (log_b - digamma_a - log(sigma2_sample[m])) * a_star
            grad_b_log_q = (a_star/b_star - 1/sigma2_sample[m]) * b_star
            log_prob_Ysigma2 = -J*L/2 * log(2*π*sigma2_sample[m]) - (model.a0 + 1) * log(sigma2_sample[m]) - model.b0/sigma2_sample[m]
            @inbounds for j in 1:J
                log_prob_Ysigma2 -= 1/(2 * sigma2_sample[m]) * dot(beta_sample[j][m], beta_sample[j][m])
            end
            # Update Gradient estimators
            grad_a_L = (m-1)/m * grad_a_L + 1/m * grad_a_log_q * (log_prob_Ysigma2 - log_q)
            grad_b_L = (m-1)/m * grad_b_L + 1/m * grad_b_log_q * (log_prob_Ysigma2 - log_q)
            # Update ELBO estimator
            ELBO = (m-1)/m * ELBO + 1/m * (log_prob_Ysigma2 - log_q)
        end
        # Print ELBO, parameter and gradient if verbose
        if verbose
            println("ELBO: $ELBO")
            println("a*: $a_star")
            println("gradient log(a*): $grad_a_L")
            println("b*: $b_star")
            println("gradient log(b*): $grad_b_L")
        end
        
        #TODO: Stop condition

        # Update parameters
        log_a += step * grad_a_L
        log_b += step * grad_b_L
        model.a_star[1] = exp(log_a)
        model.b_star[1] = exp(log_b)
    end
end


# Update all independent distributions at once

struct DCModel2{T <: AbstractFloat}
    # data
    obs             :: DCMObs
    # Prior distribution parameters
    d0              :: Vector{T}
    a0              :: T
    b0              :: T
    # Variational distribution parameters
    pi_star         :: Vector{Vector{T}}
    mu_star         :: Vector{Vector{T}}
    V_star          :: Vector{Matrix{T}}
    d_star          :: Vector{T}
    a_star          :: Vector{T}
    b_star          :: Vector{T}
    # Number of samples for noisy gradient
    M               :: Int
    # Preallocated storage for samples from variational distribution
    Z_sample        :: Vector{Vector{Vector{Int}}}
    beta_sample     :: Vector{Vector{Vector{T}}}
    pi_sample       :: Vector{Vector{T}}
    sigma2_sample   :: Vector{T}
    # Preallocated storage for noisy gradient descent calculations
    storage_L       :: Vector{T}
    storage_L2      :: Vector{T}
    storage_L3      :: Vector{T}
    storage_L4      :: Vector{T}
    storage_LL      :: Matrix{T}
    storage_LL2     :: Matrix{T}
    storage_LL3     :: Matrix{T}
    storage_LL4     :: Matrix{T}
    # Preallocated storage for matrix vectorization operations
    storage_comm    :: Matrix{T}
    storage_dup     :: Matrix{T}
    storage_Lsqr    :: Vector{T}
    storage_Lsqr2   :: Vector{T}
    storage_L2L2    :: Matrix{T}
    storage_C       :: Matrix{T}
    storage_gradC   :: Vector{T}
    # Preallocated Identity matrix
    I_LL            :: Matrix{T}
end

function DCModel2(
    obs             :: DCMObs,
    d0              :: Vector{T},
    a0              :: T,
    b0              :: T,
    M               :: Int) where T <: AbstractFloat
    N, J, L = size(obs.Y, 1), size(obs.Y, 2), size(obs.D[1], 1)
    # Initialize variational distribution parameters
    pi_star = Vector{Vector{T}}(undef, N)
    for i in 1:N
        pi_star[i] = ones(L) ./ L
    end
    mu_star = Vector{Vector{T}}(undef, J)
    V_star = Vector{Matrix{T}}(undef, J)
    for j in 1:J
        num_features = size(obs.D[j], 2)
        mu_star[j] = zeros(num_features)
        V_star[j] = Matrix(1.0I, num_features, num_features)
    end
    d_star = ones(L) ./ L
    a_star = [3.0]
    b_star = [3.0]
    # Preallocate space for samples from variational distribution
    Z_sample = Vector{Vector{Vector{Int}}}(undef, N)
    for i in 1:N
        Z_sample[i] = Vector{Vector{Int}}(undef, M)
        for m in 1:M
            Z_sample[i][m] = Vector{Int}(undef, L)
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
    pi_sample = Vector{Vector{T}}(undef, M)
    for m in 1:M
        pi_sample[m] = Vector{T}(undef, L)
    end
    sigma2_sample = Vector{T}(undef, M)
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
    # Initialize DCModel object
    DCModel2(obs, d0, a0, b0, 
    pi_star, mu_star, V_star, d_star, a_star, b_star, M,
    Z_sample, beta_sample, pi_sample, sigma2_sample,
    storage_L, storage_L2, storage_L3, storage_L4, storage_LL, storage_LL2, storage_LL3, storage_LL4,
    storage_comm, storage_dup, storage_Lsqr, storage_Lsqr2, storage_L2L2, storage_C, storage_gradC,
    I_LL)
end

function update_pi_star2(
    model           :: DCModel;
    step            :: T = 1e-2,
    tol             :: T = 1e-6,
    maxiter         :: Int = 100000,
    verbose         :: Bool = true
) where T <: AbstractFloat
    obs = model.obs
    Y, D = obs.Y, obs.D
    Z_sample, beta_sample, pi_sample = model.Z_sample, model.beta_sample, model.pi_sample
    pi_star_old = model.pi_star
    N, J, L = size(Y, 1), size(Y, 2), size(D[1], 1)
    M = model.M
    # Sample Z, β, and pi from variational distribution. Only samples of Z will be updated as the parameters update
    sample_variational_distribution(model, sample_Z = true, sample_β = true, sample_pi = true)
    # Fully update parameters of each Z_i using noisy gradients before moving to update parameters of next Z_i
    @inbounds for i in 1:N
        # Storage for gradient terms
        grad_log_q = model.storage_L2
        grad_L = model.storage_L3
        # Storage for intermediate term in gradient calculations
        D_beta = model.storage_L
        rho_star_old_i = view(model.storage_LL3, 1:L)
        # Get parameters for variational distribution of skill of i-th student
        pi_star_old_i = pi_star_old[i]
        # Perform gradient descent update of i-th π*    
        @inbounds for iter in 1:maxiter
            # Rho is unique up to a constant addative term
            rho_star_old_i = log.(pi_star_old_i)
            # Sample Z with updated π*
            sample_variational_distribution(model, sample_Z = true, idx_Z = i)
            # Set gradient of ELBO to 0
            fill!(grad_L, 0)
            # Rao Blackwellized ELBO
            ELBO = 0
            # Calculate the gradient estimate of the m-th sample
            @inbounds for m in 1:M
                z_im = Z_sample[i][m]
                # Calculate gradient of log(q_1i(Z_i)) w.r.t. π*_i
                grad_log_q .= z_im .- pi_star_old_i
                # Calculate log(p(Y, Z_(i)))
                log_prob_YZ = 0
                for j in 1:J
                    mul!(D_beta, D[j], beta_sample[j][m])
                    log_prob_YZ += dot(z_im, log.(sigmoid.((2*Y[i,j] - 1) .* D_beta)))
                end
                log_prob_YZ += dot(z_im, log.(pi_sample[m]))
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
;