import numpy as np
from farms_network.core import network, node, options
from farms_network.core.data import NetworkData


nstates = 100
niterations = 1000

net_opts = options.NetworkOptions(
    logs=options.NetworkLogOptions(
        n_iterations=niterations,
    )
)

data = NetworkData.from_options(net_opts)


net = network.Network(nnodes=10)

n1_opts = options.NodeOptions(
    name="n1",
    parameters=options.NodeParameterOptions(),
    visual=options.NodeVisualOptions(),
    state=options.NodeStateOptions(initial=[0, 0]),
)
n1 = node.Node.from_options(n1_opts)
n1_opts.save("/tmp/opts.yaml")


print(n1.name)
n1.name = "n2"
print(n1.model_type)
print(n1.name)

states = np.empty((1,))
dstates = np.empty((1,))
inputs = np.empty((10,))
weights = np.empty((10,))
noise = np.empty((10,))
drive = 0.0

print(
    n1.ode_rhs(0.0, states, dstates, inputs, weights, noise, drive)
)

print(
    n1.output(0.0, states)
)

n2 = li_danner.PyLIDannerNode("n2", ninputs=50)

print(n2.name)
print(n2.model_type)
n2.name = "n2"
print(n2.name)

states = np.empty((1,))
dstates = np.empty((1,))
inputs = np.empty((10,))
weights = np.empty((10,))
noise = np.empty((10,))
drive = 0.0

print(
    n2.output(0.0, states)
)
