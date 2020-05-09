# %%
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
out = hn.association_learning(df)

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
out = hn.association_learning(df)

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

# %% Sklearn examples
import pandas as pd
from hnet import hnet

from sklearn.datasets import fetch_kddcup99
df = fetch_kddcup99(return_X_y=False)
df = pd.DataFrame(data=df['data'])
df.columns = df.columns.astype(str)
hn = hnet()
out = hn.association_learning(df)

