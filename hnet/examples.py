import numpy as np
import hnet
print(dir(hnet))
print(hnet.__version__)


# %% Import class
from hnet import hnet
print(dir(hnet))

# %%
from hnet import hnet
df = hnet.import_example('titanic')

df = hnet.import_example('student')

df = hnet.import_example('sprinkler')


# %% Run with default settings
from hnet import hnet

hn = hnet()
# Load data
df = hn.import_example('titanic')
# Structure learning
hn.association_learning(df)

# %% Plot with clustering nodes

# Plot dynamic graph
G_dynamic = hn.d3graph(node_color='cluster')

# Plot static graph
G_static = hn.plot(node_color='cluster')

# Plot heatmap
P_heatmap = hn.heatmap(cluster=True)


#%% Association learning

from hnet import hnet

hn = hnet()
# Structure learning
out = hn.fit_transform(df, return_as_dict=True)

# Import hnet functionalities
import hnet
# Examine differences between models
[scores, adjmat] = hnet.compare_networks(out['simmatP'], out['simmatP'], showfig=True)

# Make undirected matrix
adjmat_undirected = hnet.to_undirected(out['simmatLogP'])


# %% Enrichment
import hnet

df = hnet.import_example('titanic')
y = df['Survived'].values
out = hnet.enrichment(df, y)
