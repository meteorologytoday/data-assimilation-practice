using Printf



function dXdt_Lorenz(
    t :: Float64,
    X :: AbstractArray{Float64, 1};
    p,
)
    
    x = X[1]
    y = X[2]
    z = X[3]

    A = [
        (-p.σ)      p.σ  0.0 ;
        (p.ρ - z)   -1   0.0 ;
        0.0         x    -p.β  ;
    ]


    return A

end

function RK2(
    f :: Function,
    Δt :: Float64,
    t :: Float64,
    X :: AbstractArray{Float64, 1},
)
    # f = dXdt

    k1 = f(t, X) * X

    X_mid = X + Δt * k1

    k2 = f(t + Δt, X + Δt * k1) * X

    A = k1 * 0.5 + k2 * 0.5

    return A

end



p = (
    ρ = 28.0,
    σ = 10.0,
    β = 8.0/3.0,
)


mydXdt(t, X) = dXdt_Lorenz(t, X; p=p)


X0 = [ 0.0, 0.0, 1.0 ]
Δt = 0.1
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

    A = RK2(mydXdt, Δt, t, X_now)
    
    record.X[step+1, :] = X_now + Δt * A

end










