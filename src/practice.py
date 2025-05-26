
print("Importing Libraries")
import jax
import jax.numpy as jnp
import numpy as np
print("done")

def runModel(
    dXdt,
    time_int_method,
    X0,
    steps,
    Δt,
    
):

    print("Record")
    record = dict(
        t = np.zeros((steps+1,), dtype=np.float32),
        X = np.zeros((steps+1, len(X0)), dtype=np.float32),
    )

    record["X"][0, :] = X0
    print("Ready to run the model.")
    for step in range(0, steps):
        if step % 100 == 0 or step == steps-1:
            print("Step %d" % (step+1,))

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
dt = 0.01
t = 0.0


total_time = 10.0
steps = int(np.ceil(total_time / dt))

record = runModel(
    myLorenz_dXdt,
    RK2,
    X0,
    steps,
    dt,
)

t = record["t"]
x = record["X"][:, 0]
y = record["X"][:, 1]
z = record["X"][:, 2]

# add noise
obs_sig = 0.01
obs_interval = 1.0
obs_interval_N = int(np.floor(obs_interval / dt))

X_obs = record["X"] + np.random.randn(*record["X"].shape) * obs_sig


X_obs = record["X"][::obs_interval_N, :]
t_obs = record["t"][::obs_interval_N]
x_obs = X_obs[:, 0]
y_obs = X_obs[:, 1]
z_obs = X_obs[:, 2]









print("Loading matplotlib...")
import matplotlib
#if args.no_display:
#    matplotlib.use('Agg')
#else:
#    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

fig, ax = plt.subplots(2, 3)

ax[0, 0].plot(t, x)
ax[0, 1].plot(t, y)
ax[0, 2].plot(t, z)

ax[0, 0].scatter(t_obs, x_obs)
ax[0, 1].scatter(t_obs, y_obs)
ax[0, 2].scatter(t_obs, z_obs)



ax[1, 0].plot(x, y)
ax[1, 1].plot(x, z)

print("Showing results...")
plt.show()

