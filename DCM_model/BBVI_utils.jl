using LinearAlgebra, Combinatorics

function generate_features(q::Vector{Int})
    num_skills = length(q)
    # enumerate skills and determine which skills are measured
    masteries = [1:1:num_skills;]'[q .== 1]
    # add intercept term to features
    features = [zeros(Int64, num_skills)]
    # combinations produce main effect and interaction terms in terms of enumerated skills
    for comb in combinations(masteries)
        # turn enumerated skill combinations back into binary skill labels
        feature = zeros(Int64, num_skills)
        for k in comb
            feature[k] = 1
        end
        # add main effect and interaction terms to features
        push!(features, feature)
    end
    features
end

function generate_delta(
    Q::Matrix{Int})
    J, A, L = size(Q, 1), size(Q, 2), 2^size(Q, 2)
    D = Matrix{Int64}[]
    profiles = [ digits(n, base=2, pad=2) for n in 0:(L-1) ]
    # For each question construct feature matrix
    for j in 1:J
        # number of skills measured by question j
        num_masteries = sum(Q[j, :])
        features = generate_features(Q[j,:])
        d = zeros(L, 2^num_masteries)
        for k in 1:2^num_masteries
            for l in 1:L
                # d[l, k] indicates whether all skills in feature k are mastered by attribute profile l
                d[l, k] = Int64(dot(profiles[l], features[k]) >= sum(features[k]))
            end
        end
        push!(D, d)
    end    
    D
end

function sigmoid(z::AbstractFloat)
    return 1/(1 + exp(-z))
end

function get_dup!(
    out         :: Matrix{T},
    u_storage   :: Vector{T},
    T_storage   :: Matrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    fill!(u_storage, 0)
    fill!(T_storage, 0)
    for j in 0:(k - 1)
        for i in j:(k - 1)
            u_idx = Int(j*k+i-((j+1)*j)/2 + 1)
            u_storage[u_idx] = 1
            T_storage[i + 1, j + 1] = 1
            BLAS.gemm!('N', 'T', T(1), view(T_storage, 1:k^2), u_storage, T(1), out)
            u_storage[u_idx] = 0
            T_storage[i + 1, j + 1] = 0
        end
    end
end

function get_elim!(
    out         :: Matrix{T},
    u_storage   :: Vector{T},
    T_storage   :: Matrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    fill!(u_storage, 0)
    fill!(T_storage, 0)
    for j in 0:(k - 1)
        for i in j:(k - 1)
            u_idx = Int(j*k+i-((j+1)*j)/2 + 1)
            u_storage[u_idx] = 1
            T_storage[i + 1, j + 1] = 1
            BLAS.gemm!('N', 'T', T(1), u_storage, view(T_storage, 1:k^2), T(1), out)
            u_storage[u_idx] = 0
            T_storage[i + 1, j + 1] = 0
        end
    end
end

function get_comm!(
    out         :: Matrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    for j in 0:(k-1)
        for i in 0:(k-1)
            out[j*k + i + 1, k*i + j + 1] = 1
        end
    end
end
;