# %%
import hnet

import pandas as pd
# df = pd.read_csv('D://stack//GITLAB//PROJECTS//portscans//portscans//data//portscan_export_maart.zip')
df = pd.read_csv('D://stack//GITLAB//PROJECTS//portscans//portscans//data///portscan_export_april.zip')
df['_time']=pd.to_datetime(df['_time'])
df['dayname'] = df['_time'].dt.dayofweek
# df['month'] = df['_time'].dt.month
df['_time'].dt.strftime('%d-%m-%Y').values
df['test_all_false'] = False
df['test_false_and_true'] = False
df['test_false_and_true'].iloc[0]=True

from hnet import hnet
hn = hnet(y_min=10)
hn.association_learning(df)
hn.plot()
hn.heatmap(cluster=True)
hn.d3graph()

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
