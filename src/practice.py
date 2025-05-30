
print("Importing Libraries")
import jax
import jax.numpy as jnp
import numpy as np
print("done")



def runModelEnsemble(
    run_model_func,
    dXdt,
    time_int_method,
    ens_X0,
    steps,
    dt,
):
    # In case it is just a nested array
    ens_X0 = jnp.array(ens_X0)
    
    full_X = []
    t = None
    for n in range(ens_X0.shape[0]):
        print("Running ensemble: ", n) 
        X0 = ens_X0[n, :]
        _record = run_model_func(dXdt, time_int_method, X0, steps, dt)
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
    X0,
    steps,
    Δt,
    
):

    #print("Record")
    record = dict(
        t = np.zeros((steps+1,), dtype=np.float32),
        X = np.zeros((steps+1, len(X0)), dtype=np.float32),
    )

    record["X"][0, :] = X0
    #print("Ready to run the model.")
    for step in range(0, steps):
#        if step % 100 == 0 or step == steps-1:
#            print("Step %d" % (step+1,))

        X_now = record["X"][step, :]

        AX = time_int_method(dXdt, Δt, t, X_now)
        
        record["X"][step+1, :] = X_now + Δt * AX
        record["t"][step+1] = record["t"][step] + Δt


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


def RK2(f, dt, t, X):
    
    k1 = f(t, X) @ X

    X_mid = X + dt * k1

    k2 = f(t + dt, X + dt * k1) @ X

    AX = k1 * 0.5 + k2 * 0.5

    return AX

p = dict(
    rho   = 28.0,
    sigma = 10.0,
    beta  = 8.0/3.0,
)

myLorenz_dXdt = lambda t, X: dXdt_Lorenz(t, X, p)


X0 = jnp.array([ 1.0, 1.0, 1.0 ])

Ne = 10
dt = 0.01
t = 0.0
total_time = 2.0

ens_X0 = jnp.zeros((Ne, len(X0),), dtype=X0.dtype)
ens_X0 = ens_X0 + np.random.randn(*ens_X0.shape) * 1.0


steps = int(np.ceil(total_time / dt))

record = runModelEnsemble(
    runModel,
    myLorenz_dXdt,
    RK2,
    ens_X0,
    steps,
    dt,
)

t = record["t"]
x = record["X"][:, :, 0]
y = record["X"][:, :, 1]
z = record["X"][:, :, 2]

# add noise
"""
obs_sig = 0.01
obs_interval = 1.0
obs_interval_N = int(np.floor(obs_interval / dt))

X_obs = record["X"] + np.random.randn(*record["X"].shape) * obs_sig


X_obs = record["X"][::obs_interval_N, :]
t_obs = record["t"][::obs_interval_N]
x_obs = X_obs[:, 0]
y_obs = X_obs[:, 1]
z_obs = X_obs[:, 2]
"""

print("Loading matplotlib...")
import matplotlib
#if args.no_display:
matplotlib.use('Agg')
#else:
#    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

fig, ax = plt.subplots(2, 3)

ax[0, 0].plot(t, x.T)
ax[0, 1].plot(t, y.T)
ax[0, 2].plot(t, z.T)

#ax[0, 0].scatter(t_obs, x_obs)
#ax[0, 1].scatter(t_obs, y_obs)
#ax[0, 2].scatter(t_obs, z_obs)



ax[1, 0].plot(x.T, y.T)
ax[1, 1].plot(x.T, z.T)
ax[1, 2].plot(y.T, z.T)

print("Showing results...")
#plt.show()
plt.savefig("test.svg")
