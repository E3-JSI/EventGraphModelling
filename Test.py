import Erdos_Renyi as er 
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
import json 
import pandas as pd 
from collections import Counter
import numpy as np
import pdb 
import networkx as nx 
import matplotlib.pyplot as plt

# Parameters
p = 0.5
subset = 100
const = 60*60*24*7 # 1 week

# Load settings file
settings = json.load(open("settings.json","r"))
# Path to file
events_embd = settings["out_path"]+"concepts_date_embd.csv"
# Load events
print("Reading in data")
events = pd.read_csv(events_embd,header=None)
# Timestamps of events in UNIX format
time = np.unique(events.iloc[:,-1].values)

# Iterate through data
for t in time: 
    t_start = t - const 
    # Get only events in the time period
    tmp=events[(events.iloc[:,-1]<=t) & (events.iloc[:,-1]>=t_start)]
    # Take a subset, ideally we could take all
    L = tmp.head(subset).iloc[:,:-1].values
    print("Calculating W from data")
    W_real = er.get_W(L)

    # Set to minimum of limit and shape of data (due to starting period with less events)
    K = min(subset,tmp.shape[0])
    S_real = np.eye(K,dtype=np.int)

    # Define network
    ld_network = er.LatentDistanceAdjacencyModel(K=K, L = None, dim=2, v=None, alpha=1.0, beta=1.0,kappa=1.0,p = p)

    print("Resampling ld network")
    ld_network.resample(data=[S_real,W_real])

    # Define weight model
    weight_model = er.SpikeAndSlabGammaWeights(model = ld_network, parallel_resampling=False)

    print("Resampling weight model")
    weight_model.resample_new(data=[S_real,W_real])

    # Print adjecancy matrix and it's properties 
    print(weight_model.A)
    print(sum(weight_model.A))
    print(max(sum(weight_model.A)))

    # Plot graph
    G = nx.from_numpy_matrix(weight_model.A)
    nx.draw(G,with_labels=True,with_edges=True)
    plt.show()