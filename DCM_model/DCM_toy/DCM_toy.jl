using JSON
using TickTock
using RCall
include("../BBVI_modeling.jl")

using RCall

R"""
load("C:\\Users\\14842\\Desktop\\BBVI\\DCM_model\\data.RData")
"""
data = @rget data
Y = data[:Y]
Q = convert(Matrix{Int64}, data[:Q])
obs = DCMObs(Y, Q);

a0 = 1e-2
b0 = 1e-4
d0 = ones(size(obs.D[1], 1))
M = 100
model = DCModel(obs, d0, a0, b0, M, enable_parallel=true)

tick()
for k in 1:100
    update_pi_star(model, step = 1e-2, maxiter = 10, verbose = false)
    update_mu_star_V_star(model, init_step = .0005, use_iter = false, maxiter = 20, verbose = false)
    update_d_star(model, step = 1e-4, maxiter = 20, verbose = false)
    update_a_star_b_star(model, step = 1e-4, maxiter = 30, verbose = false)
end
runtime = tok()

dirpath = "C:\\Users\\14842\\Desktop\\BBVI\\DCM_model\\DCM_toy\\script_out"

fname = "beta_toy.json"
fpath = joinpath(dirpath, fname)

open(fpath, "w") do file
    write(file, JSON.json(model.mu_star))
end

fname = "profile_toy.json"
fpath = joinpath(dirpath, fname)

open(fpath, "w") do file
    write(file, JSON.json(model.pi_star))
end

fname = "runtime_toy.json"
fpath = joinpath(dirpath, fname)

open(fpath, "w") do file
    write(file, JSON.json(runtime))
end