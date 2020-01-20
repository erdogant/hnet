# %% Tests

# %%
import numpy as np
import pandas as pd
# import hnet as hnet
import hnet.hnet as hnet

# %% Example with random categorical and numerical values
nfeat=100
nobservations=50
df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
dtypes = np.array(['cat']*nobservations)
dtypes[np.random.randint(0,2,nobservations)==1]='num'
y   = np.random.randint(0,2,nfeat)
out = hnet.enrichment(df,y, dtypes=dtypes)
out = hnet.fit(df,dtypes=dtypes)

# %% Example with 1 true positive column
nfeat=100
nobservations=50
df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
y  = np.random.randint(0,2,nfeat)
df['positive_one'] = y
dtypes = np.array(['cat']*(nobservations+1))
dtypes[np.random.randint(0,2,nobservations+1)==1]='num'
dtypes[-1]='cat'
out = hnet.enrichment(df,y, alpha=0.05, dtypes=dtypes)

# %% Example most simple manner
nfeat=100
nobservations=50
df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
y  = np.random.randint(0,2,nfeat)
out  = hnet.enrichment(df,y)
out  = hnet.enrichment(df,y, multtest=None)

# %%
df    = pd.read_csv('../DATA/OTHER/titanic/titanic.zip')
out   = hnet.enrichment(df, y=df['Survived'].values, alpha=0.05, multtest='holm')
rules = hnet.combined_rules(out)

# %%
out  = hnet.fit(df)
hnet.plot_heatmap(out['simmatP'], cluster=1, cmap='Reds_r')

# %%
df    = pd.read_csv('../../../DATA/OTHER/titanic/titanic.zip')
out   = hnet.fit(df)
out   = hnet.fit(df, k=10)
G     = hnet.plot_network(out, dist_between_nodes=0.4, scale=2)
A     = hnet.plot_d3graph(out, savepath='c://temp/titanic3/', directed=False)

# %%
import hnet.hnet as hnet

df    = hnet.import_example('sprinkler')
out   = hnet.fit(df, alpha=0.05, multtest='holm', excl_background=['0.0'])

G     = hnet.plot_network(out, dist_between_nodes=0.1, scale=2)
G     = hnet.plot_network(out)
G     = hnet.plot_network(out, savepath='c://temp/sprinkler/')

A     = hnet.plot_heatmap(out, savepath='c://temp/sprinkler/', cluster=False)
A     = hnet.plot_heatmap(out, savepath='c://temp/sprinkler/', cluster=True)
A     = hnet.plot_heatmap(out, cluster=False)

A     = hnet.plot_d3graph(out)
A     = hnet.plot_d3graph(out, savepath='c://temp/sprinkler/', directed=False)

# %%
out   = hnet.fit(df, alpha=0.05, multtest='holm')
G     = hnet.plot_network(out, dist_between_nodes=0.1, scale=2)
G     = hnet.plot_network(out)
G     = hnet.plot_network(out, savepath='c://temp/sprinkler/')
A     = hnet.plot_heatmap(out, savepath='c://temp/sprinkler/', cluster=False)
A     = hnet.plot_d3graph(out, savepath='c://temp/sprinkler/', directed=False)

# %%
df    = pd.read_csv('../../../DATA/OTHER/elections/USA_2016_election_primary_results.zip')
out   = hnet.fit(df, alpha=0.05, multtest='holm', dtypes=['cat','','','','cat','cat','num','num'])
G     = hnet.plot_network(out, dist_between_nodes=0.4, scale=2)
A     = hnet.plot_d3graph(out, savepath='c://temp/USA_2016_elections/')

# %%
df    = pd.read_csv('../DATA/OTHER/titanic/titanic.zip')
out   = hnet.fit(df)
out   = hnet.fit(df, alpha=1, dropna=False)
G     = hnet.plot_d3graph(out, savepath='c://temp/magweg/')

