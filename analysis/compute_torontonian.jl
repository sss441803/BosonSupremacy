using LinearAlgebra
using CSV
using DataFrames
using NPZ

include("../torontonian-julia/torontonian.jl")

function get_tor(cov, samples, chi)
    nrows, _ = size(samples)

    for N in 5:14
        sub_array = vcat([row' for row in eachrow(samples) if count(x -> x != 0, row) == N]...)

        results = Vector{Float64}()
        
        nrows_sub, _ = size(sub_array)

        for i in 1:min(1000, nrows_sub) # 1000 is the number of samples to be computed
            append!(results, real(threshold_detection_prob(cov, sub_array[i, :])))
        end
        npzwrite(ARGS[1] * "tors_$(N).npy", results)
    end

    return results
end

cov = npzread(ARGS[1] * "cov.npy")
samples = npzread(ARGS[1] * ARGS[2])
get_tor(cov, samples, chi)