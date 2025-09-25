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
    N, O, J, K, L, S = N_school * N_student_per_school, N_time, size(Q, 1),  size(obs.Q, 2), size(obs.D[1], 1), N_school
    # Generate item response parameters
    beta_intercept_dist = Distributions.Uniform(-2, 0.5)
    beta_feature_dist = Distributions.Normal(2, 1)
    mu_beta_star = Vector{Vector{T}}(undef, J)
    for j in 1:J
        num_features = size(obs.D[j], 2)
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
            num_features_gamma = size(obs.X[k][t], 2)
            num_features_omega = size(obs.U[k][t], 2)
            if t == 1
                mu_omega_star[k][t] = Vector{Vector{Vector{T}}}(undef, 1)
                mu_omega_star[k][t][1] = Vector{Vector{T}}(undef, num_features_gamma)
                for m in 1:num_features_gamma
                    mu_omega_star[k][t][1][m] = rand(omega_feature_dist, num_features_omega)
                end

                a_tau_star[k][t] = Vector{Vector{T}}(undef, 1)
                a_tau_star[k][t][1] = ones(1) .* S

                b_tau_star[k][t] = Vector{Vector{T}}(undef, 1)
                b_tau_star[k][t][1] = ones(1) .* .04 .* S
            else
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
    end
    # Generate individual level transition parameters
    mu_gamma_star = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, K)
    for k in 1:K
        mu_gamma_star[k] = Vector{Vector{Vector{Vector{T}}}}(undef, O)
        for t in 1:O
            num_features = size(obs.X[k][t], 2)
            if t == 1
                mu_gamma_star[k][t] = Vector{Vector{Vector{T}}}(undef, 1)

                mu_gamma_star[k][t][1] = Vector{Vector{T}}(undef, S)
                for s in 1:S
                    mu_gamma_star[k][t][1][s] = zeros(num_features)
                end
            else
                mu_gamma_star[k][t] = Vector{Vector{Vector{T}}}(undef, 2^(t - 1))
                for z in 1:(2^(t - 1))
                    mu_gamma_star[k][t][z] = Vector{Vector{T}}(undef, S)
                    for s in 1:S
                        mu_gamma_star[k][t][z][s] = zeros(num_features)
                    end
                end
            end
        end
    end
end