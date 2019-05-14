import Erdos_Renyi as er 
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
import json 
import pandas as pd 
from collections import Counter
import numpy as np

# Parameters
p = 0.5
dt_max=200

if False:
    T = 100
    network_hypers = {"p": p, "allow_self_connections": False}

    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max,
        network_hypers=network_hypers)
    assert true_model.check_stability()

    # Synthetic data
    S,R = true_model.generate(T=T, keep=True, print_interval=50)

# Real data
settings = json.load(open("settings.json","r"))
# Path to file
events_embd = settings["out_path"]+"concepts_date_embd.csv"
# load events
print("Reading in data")
events = pd.read_csv(events_embd,header=None)
subset = 200

L = events.head(subset).iloc[:,:-1].values
S_real = np.eye(subset,dtype=np.int)#events.head(subset)[300].values
#S_real = [[itm] for _,itm in Counter(S_real).items()]
K = subset

ld_network = er.LatentDistanceAdjacencyModel(K=K, L = L, dim=2, v=None, alpha=1.0, beta=1.0,kappa=1.0,p = p)

# A,W obtain
weight_model = er.SpikeAndSlabGammaWeights(model = ld_network, parallel_resampling=False)
weight_model.resample(S_real)
# resample graph with new data 
ld_network.resample(data=[A,W])
# test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt_max=dt_max,network=ld_network)

# test_model.add_data(S_real)
# test_model.resample_model()

# test_figure, test_handles = test_model.plot(color="#e41a1c", T_slice=(0,subset))
# plt.show()