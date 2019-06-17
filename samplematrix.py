import Erdos_Renyi as er 
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
import json 
import pandas as pd 
from collections import Counter
import numpy as np
import pdb 
import networkx as nx 
import matplotlib.pyplot as plt

from Erdos_Renyi import get_W 

# Parameters
p = 0.3
subset = 15
K = subset
#const = 60*60*24*7 # 1 week

# Load settings file
settings = json.load(open("settings.json","r"))
# Path to file
events_embd = settings["out_path"]+"concepts_date_embd.csv"
# Load events
print("Reading in data")
events = pd.read_csv(events_embd,header=None)
L = events.head(subset)

print("Calculating W from data")
W_real = er.get_W(L)

# Set to minimum of limit and shape of data (due to starting period with less events)
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

def W_sample(A,W):
    return A*W

print(W_sample(W_real,weight_model.A))

