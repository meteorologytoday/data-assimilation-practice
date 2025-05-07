
module TimeIntegrationMethods

    function EulerForward(
        f :: Function,
        Δt :: Float64,
        t :: Float64,
        X :: AbstractArray{Float64, 1},
    )

        # f = dXdt

        k1 = f(t, X) * X
        AX = k1 

        return AX

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

        AX = k1 * 0.5 + k2 * 0.5

        return AX

    end




end
