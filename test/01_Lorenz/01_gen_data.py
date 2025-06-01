import Lorenz
import TimeIntegrationMethods
import xarray as xr
import jax.numpy as jnp
import numpy as np

X0 = jnp.array([ 1.0, 1.0, 1.0 ])

dt = 0.01
t0 = 0.0
total_time = 10.0

obs_dstep = 10


steps = int(np.ceil(total_time / dt))

record = Lorenz.runModel(
    Lorenz.myLorenz_dXdt,
    TimeIntegrationMethods.RK2,
    t0,
    X0,
    steps,
    dt,
)

t = record["t"]
X = record["X"]
x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

print("Making true dataset")
ds = xr.Dataset(
    data_vars=dict(
        t = (["time",], t),
        X = (["time", "var"], X),
    ),
    coords = dict(
        time = (["time",], t),
        var  = (["var",], ["x", "y", "z"]),
    )
)

output_file = "data_truth.nc"



print("Output file: ", output_file)
ds.to_netcdf(output_file, unlimited_dims="time")


print("Making observation dataset")

t_obs = t[::obs_dstep]
X_obs = X[::obs_dstep, :]

X_obs += np.random.randn(*X_obs.shape) * 1.0

ds = xr.Dataset(
    data_vars=dict(
        t = (["time",], t_obs),
        X = (["time", "var"], X_obs),
    ),
    coords = dict(
        time = (["time",], t_obs),
        var  = (["var",], ["x", "y", "z"]),
    )
)
output_file = "data_obs.nc"
print("Output file: ", output_file)
ds.to_netcdf(output_file, unlimited_dims="time")



print("Loading matplotlib...")
import matplotlib
#if args.no_display:
#matplotlib.use('Agg')
#else:
#    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

fig, ax = plt.subplots(2, 3)

ax[0, 0].scatter(t, x)
ax[0, 1].scatter(t, y)
ax[0, 2].scatter(t, z)

ax[1, 0].plot(x.T, y.T)
ax[1, 1].plot(x.T, z.T)
ax[1, 2].plot(y.T, z.T)

print("Showing results...")
plt.show()
