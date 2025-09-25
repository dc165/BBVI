using Distributions
include("../DCM_model/BBVI_utils.jl")
include("TDCM_full_transition.jl")


function data_generation(
    Q                       :: Matrix{Int},
    X                       :: Vector{Vector{Matrix{T}}},
    U                       :: Vector{Vector{Matrix{T}}},
    N_time                  :: Int,
    N_skill                 :: Int,
    N_school                :: Int,
    N_student_per_school    :: Int;
    beta_interact           :: Bool=true
) where T <: AbstractFloat
    D = generate_delta(Q)
    N, O, J, K, L, S = N_school * N_student_per_school, N_time, size(Q, 1),  size(Q, 2), size(D[1], 1), N_school
    # Generate item response parameters
    beta_intercept_dist = Distributions.Uniform(-2, 0.5)
    beta_feature_dist = Distributions.Normal(2, 1)
    mu_beta_star = Vector{Vector{T}}(undef, J)
    for j in 1:J
        num_features = size(D[j], 2)
        mu_beta_star[j] = zeros(num_features)
        rand!(beta_feature_dist, mu_beta_star[j])
        mu_beta_star[j][1] = rand(beta_intercept_dist)
    end
    # Generate group level parameters
    omega_feature_dist = Distributions.Normal(0, 1.5)
    mu_omega_star = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    a_tau_star = Vector{Vector{Vector{Vector{T}}}}(undef, K)
    b_tau_star = Vector{Vector{Vector{Vector{T}}}}(undef, K)
    for k in 1:K
        mu_omega_star[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        a_tau_star[k] = Vector{Vector{Vector{T}}}(undef, O)
        b_tau_star[k] = Vector{Vector{Vector{T}}}(undef, O)
        for t in 1:O
            num_features_gamma = size(X[k][t], 2)
            num_features_omega = size(U[k][t], 2)
            mu_omega_star[k][t] = Vector{Vector{Vector{T}}}(undef, 2^(t-1))
            a_tau_star[k][t] = Vector{Vector{T}}(undef, 2^(t-1))
            b_tau_star[k][t] = Vector{Vector{T}}(undef, 2^(t-1))
            for z in 1:(2^(t-1))
                mu_omega_star[k][t][z] = Vector{Vector{T}}(undef, num_features_gamma)
                a_tau_star[k][t][z] = ones(num_features_gamma) .* S
                b_tau_star[k][t][z] = ones(num_features_gamma) .* .04 * S
                for m in 1:num_features_gamma
                    mu_omega_star[k][t][z][m] = Vector{T}(undef, num_features_omega)
                    rand!(omega_feature_dist, mu_omega_star[k][t][z][m])
                end
            end
        end
    end
    # Generate individual level transition parameters
    mu_gamma_star = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    for k in 1:K
        mu_gamma_star[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        for t in 1:O
            num_features = size(X[k][t], 2)
            mu_gamma_star[k][t] = Vector{Vector{Vector{T}}}(undef, 2^(t - 1))
            for z in 1:(2^(t - 1))
                mu_gamma_star[k][t][z] = Vector{Vector{T}}(undef, S)
                for s in 1:S
                    mu_gamma_star[k][t][z][s] = Vector{T}(undef, num_features)
                    for feature in 1:num_features
                        tau_distribution = Distributions.InverseGamma(a_tau_star[k][t][z][feature], b_tau_star[k][t][z][feature])
                        gamma_feature_distribution = Distributions.Normal(dot(mu_omega_star[k][t][z][feature], U[k][t][s, :]), rand(tau_distribution))
                        mu_gamma_star[k][t][z][s][feature] = rand(gamma_feature_distribution)
                    end
                end
            end
        end
    end
    # Assign students to schools
    group = repeat(1:S, inner = [N_student_per_school])
    # Create skill dictionary
    skill_dict = Dict{Int, Vector{Int}}()
    for l in 0:(L - 1)
        skill_dict[l + 1] = reverse(digits(l, base=2, pad=K))
    end
    # Generate profiles
    pi_star = Vector{Vector{Vector{T}}}(undef, N)
    for i in 1:N
        pi_star[i] = Vector{Vector{T}}(undef, O)
        for t in 1:O
            # probability vector of possible mastery partern over time for k-th attribute
            pi_star[i][t] = ones(2^K) ./ 2^K
            profile = 0
            for k in 1:K
                transition_idx = 1
                if time > 1
                    for o in 1:(time - 1)
                        transition_idx += skill_dict[argmax(Z_sample[i][o][m])][k] * 2^(o-1)
                    end
                end
                skill_k_distribution = Distributions.Bernoulli(sigmoid(dot(mu_gamma_star[k][t][transition_idx][group[i]], X[k][t][i,:])))
                profile += rand(skill_k_distribution) * 2^(K - k)
            end
            pi_star[i][t][profile + 1] = 1
        end
    end
    # Generate item responses
    Y = zeros(O, N, J)
    for t in 1:O
        for i in 1:N
            for j in 1:J
                Y_tij_distribution = Distributions.Bernoulli(sigmoid(dot(D[j][argmax(pi_star[i][t]), :], mu_beta_star[j])))
                Y[t, i, j] = rand(Y_tij_distribution)
            end
        end
    end
    return (Y = Y, X = X, U = U, group = group, Q = Q, mu_beta_star = mu_beta_star, pi_star = pi_star, 
            mu_gamma_star = mu_gamma_star, mu_omega_star = mu_omega_star, a_tau_star = a_tau_star, 
            b_tau_star = b_tau_star)
end