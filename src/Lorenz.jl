include(joinpath(@__DIR__, "TimeIntegrationMethods.jl"))

using Printf
using LinearAlgebra
using TimeIntegrationMethods

function dXdt_Lorenz(
    t :: Float64,
    X :: AbstractArray{Float64, 1};
    p,
)
    
    x = X[1]
    y = X[2]
    z = X[3]

    A = [
        (-p.σ)       p.σ     0.0  ;
        (p.ρ - z)   -1.0     0.0  ;
        0.0            x    -p.β  ;
    ]

    return A

end



p = (
    ρ = 28.0,
    σ = 10.0,
    β = 8.0/3.0,
)


mydXdt(t, X) = dXdt_Lorenz(t, X; p=p)

#TimeIntegrationMethod = EulerForward
TimeIntegrationMethod = TimeIntegrationMethods.RK2

X0 = [ 1.0, 1.0, 1.0 ]
Δt = 0.01
t = 0.0

total_time = 10.0
steps = Int64(ceil(total_time / Δt))

println("Record")
record = (
    t = zeros(Float64, steps+1),
    X = zeros(Float64, steps+1, length(X0)),
)


record.X[1, :] = X0



println("Ready to run the model.")
for step = 1:steps
    if mod(step, 100) == 1 || step == steps
        @printf("Step %d\n", step)
    end

    X_now = record.X[step, :]

    AX = TimeIntegrationMethod(mydXdt, Δt, t, X_now)
    
    record.X[step+1, :] = X_now + Δt * AX
    record.t[step+1] = record.t[step] + Δt

    x, y, z = record.X[step+1, :]

    @printf("[Step = %d] (x, y, z) = (%.2f, %.2f, %.2f)\n", step, x, y, z)
end


t = record.t
x = record.X[:, 1]
y = record.X[:, 2]
z = record.X[:, 3]


println("Loading PyPlot...")
using PyPlot
plt = PyPlot
println("Done loading")

fig, ax = plt.subplots(2, 3)

ax[1, 1].plot(t, x)
ax[1, 2].plot(t, y)
ax[1, 3].plot(t, z)

ax[2, 1].plot(x, y)
ax[2, 2].plot(x, z)

println("Showing results...")
plt.show()




