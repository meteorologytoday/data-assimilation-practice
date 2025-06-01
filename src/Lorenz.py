
print("Importing Libraries")
import jax
import jax.numpy as jnp
import numpy as np
print("done")

def runModelEnsemble(
    run_model_func,
    dXdt,
    time_int_method,
    t0,
    ens_X0,
    steps,
    dt,
):
    # In case it is just a nested array
    ens_X0 = jnp.array(ens_X0)
    
    full_X = []
    t = None
    for n in range(ens_X0.shape[0]):
        
        X0 = ens_X0[n, :]
        _record = run_model_func(dXdt, time_int_method, t0, X0, steps, dt)
        full_X.append(_record["X"])
        if t is None:
            t = _record["t"]

    full_X = jnp.stack(full_X, axis=0)
    
    record = dict(
        X = full_X,
        t = t,
    )

    return record

def runModel(
    dXdt,
    time_int_method,
    t0,
    X0,
    steps,
    dt,
    
):

    #print("Record")
    record = dict(
        t = np.zeros((steps+1,), dtype=np.float32),
        X = np.zeros((steps+1, len(X0)), dtype=np.float32),
    )

    
    record["t"][0] = t0
    record["X"][0, :] = X0
    #print("Ready to run the model.")
    for step in range(0, steps):
#        if step % 100 == 0 or step == steps-1:
#            print("Step %d" % (step+1,))

        t_now = record["t"][step]
        X_now = record["X"][step, :]

        AX = time_int_method(dXdt, dt, t_now, X_now)
        
        record["X"][step+1, :] = X_now + dt * AX
        record["t"][step+1] = record["t"][step] + dt


    return record



def dXdt_Lorenz(t, X, p):
   
    x = X[0]
    y = X[1]
    z = X[2]


    #print("x, y, z = ", x, "; ", y, "; ", z)
    A = jnp.array([
        [ - p["sigma"],  p["sigma"], 0.0 ],
        [ p["rho"] - z,  -1.0,       0.0 ],
        [ 0.0         ,  x,          - p["beta"]],
    ])

    return A


# Example
p = dict(
    rho   = 28.0,
    sigma = 10.0,
    beta  = 8.0/3.0,
)

myLorenz_dXdt = lambda t, X: dXdt_Lorenz(t, X, p)

