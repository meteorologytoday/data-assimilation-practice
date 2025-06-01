
# numpy or jnp array assumed
def RK2(f, dt, t, X):
    
    k1 = f(t, X) @ X

    X_mid = X + dt * k1

    k2 = f(t + dt, X + dt * k1) @ X

    AX = k1 * 0.5 + k2 * 0.5

    return AX

