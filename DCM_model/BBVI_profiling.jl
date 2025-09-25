using Profile
using RCall
include("BBVI_modeling.jl")

R"""
load("C:/Users/14842/Desktop/BBVI/DCM_model/data.RData")
"""
data = @rget data
Y = data[:Y]
# Convert Y to vector of vectors
# Y = [Y[i,:] for i in 1:size(Y,1)]
Q = convert(Matrix{Int64}, data[:Q])
obs = DCMObs(Y, Q)

a0 = 1e-2
b0 = 1e-4
d0 = ones(size(obs.D[1], 1))
M = 100
model = DCModel(obs, d0, a0, b0, M)

@profview update_pi_star(model, step = 1e-2, maxiter = 100, verbose = false)