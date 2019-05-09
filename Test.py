import Erdos_Renyi as er 
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

# Parameters
T = 100
K = 10
p = 0.5
dt_max  =2

network_hypers = {"p": p, "allow_self_connections": False}

true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
    K=K, dt_max=dt_max,
    network_hypers=network_hypers)
assert true_model.check_stability()

# Sample from the true model
S,R = true_model.generate(T=T, keep=True, print_interval=50)


ld_network = er.LatentDistanceAdjacencyModel(K=K, dim=2, v=None, alpha=1.0, beta=1.0,kappa=1.0,p = p)
test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
    K, dt_max=dt_max,
    network=ld_network)

import pdb; pdb.set_trace()
test_model.add_data(S)
test_figure, test_handles = test_model.plot(color="#e41a1c", T_slice=(0,100))
plt.show()