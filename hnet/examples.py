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

# %%
import matplotlib.pyplot as plt
P = hn.results['simmatLogP'].values.ravel()
P = P[P>0]

plt.figure(figsize=(15,10))
plt.hist(P, bins=50)
plt.grid(True)
plt.xlabel('-log10(P)')
plt.ylabel('Frequency')
plt.title('P-value Associaties')


# %% Association learning

from hnet import hnet
df = hnet.import_example('titanic')

hn1 = hnet(y_min=50)
hn2 = hnet()
# Structure learning
out1 = hn1.association_learning(df)
out2 = hn2.association_learning(df)

# Import hnet functionalities
import hnet
# Examine differences between models
[scores, adjmat] = hnet.compare_networks(out1['simmatLogP'], out2['simmatLogP'], showfig=True)

# Make undirected matrix
adjmat_undirected = hnet.to_undirected(out['simmatLogP'])


# %% Enrichment
import hnet

df = hnet.import_example('titanic')
y = df['Survived'].values
out = hnet.enrichment(df, y)

# %% White list and black list examples
from hnet import hnet
dtypes=['', 'cat', 'cat', '', 'cat', 'num', 'cat', 'cat', 'cat', 'num', 'cat', 'cat']
hn = hnet(black_list=['Survived','Embarked','Fare','CABIN'], dtypes=dtypes)
df = hn.import_example('titanic')
out = hn.association_learning(df)

hn = hnet(black_list=['Survived','Embarked','Fare','CABIN'])
dtypes=['', 'cat', 'cat', '', 'cat', 'num', 'cat', 'cat', 'cat', 'num', 'cat', 'cat']
df = hn.import_example('titanic')
out = hn.association_learning(df)

hn = hnet(white_list=['Survived','Embarked','Fare','Cabin'], dtypes=['num','cat','cat','cat'])
df = hn.import_example('titanic')
out = hn.association_learning(df)

hn = hnet(white_list=['Survived','Embarked','Fare','Cabin'])
df = hn.import_example('titanic')
out = hn.association_learning(df)
# hn.d3graph()

hn = hnet(black_list=['Survived','Embarked','Fare','CABIN'])
df = hn.import_example('titanic')
out = hn.association_learning(df)


# %% Run with blacklist
from hnet import hnet
hn = hnet(black_list=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp'])
# Load data
df = hn.import_example('titanic')
# Structure learning
out = hn.association_learning(df)

# %% Run with whitelist
from hnet import hnet
hn = hnet(white_list=['Survived', 'Pclass', 'Age', 'SibSp'])
# Load data
df = hn.import_example('titanic')
# Structure learning
out = hn.association_learning(df)

# %% Sklearn examples
import pandas as pd
from hnet import hnet

from sklearn.datasets import fetch_kddcup99
df = fetch_kddcup99(return_X_y=False)
df = pd.DataFrame(data=df['data'])
df.columns = df.columns.astype(str)
hn = hnet()
out = hn.association_learning(df)

