using LinearAlgebra, Combinatorics, ResumableFunctions

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
    out         :: AbstractMatrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    for j in 1:k
        for i in j:k
            u_idx = Int((j-1)*k+i-((j-1)*j)/2)
            T_idx = Int((j-1)*k + i)
            out[T_idx, u_idx] = 1
        end
    end
end

# bad implementation; can remove matrix multiplication
function get_elim!(
    out         :: AbstractMatrix{T},
    u_storage   :: AbstractVector{T},
    T_storage   :: AbstractMatrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    fill!(u_storage, 0)
    fill!(T_storage, 0)
    vec_T = view(T_storage, 1:k^2)
    for j in 0:(k - 1)
        for i in j:(k - 1)
            u_idx = Int(j*k+i-((j+1)*j)/2 + 1)
            u_storage[u_idx] = 1
            T_storage[i + 1, j + 1] = 1
            BLAS.gemm!('N', 'T', T(1), u_storage, vec_T, T(1), out)
            u_storage[u_idx] = 0
            T_storage[i + 1, j + 1] = 0
        end
    end
end

function get_comm!(
    out         :: AbstractMatrix{T}, 
    k           :: Int
) where T <: AbstractFloat
    fill!(out, 0)
    for j in 0:(k-1)
        for i in 0:(k-1)
            out[j*k + i + 1, k*i + j + 1] = 1
        end
    end
end

# The following functions are no longer used in the BBVI for DCM implementation

function h(
    u   :: T,
    pi  :: Vector{T}) where T <: AbstractFloat
    ret = -1
    for i in 1:length(pi)
        if pi[i] > u
            ret += pi[i] - u
        end
    end
    ret
end

function h_prime(
    u   :: T,
    pi  :: Vector{T}) where T <: AbstractFloat
    ret = 0
    for i in 1:length(pi)
        if pi[i] > u
            ret -= 1
        end
    end
    ret
end
function project_to_unit_simplex!(pi::Vector{T}) where T <: AbstractFloat
    u = minimum(pi)
    while h(u, pi) != 0
        u -= h(u, pi)/h_prime(u, pi)
    end
    for i in 1:length(pi)
        if pi[i] > u
            pi[i] -= u
        else
            pi[i] = 0
        end
    end
end

@resumable function get_robbins_monroe_iterator(initial_size::AbstractFloat, decay_const::Int)
    counter = decay_const
    while 1/counter > 1e8
        @yield initial_size / counter
        counter += 1
    end
end
;