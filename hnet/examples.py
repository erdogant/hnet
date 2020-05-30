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

# %%
from hnet import hnet
import pandas as pd

df = pd.read_csv('D://PY//DATA//OTHER//marketing_data_online_retail_small.csv', sep=';')
hn1 = hnet()
results1 = hn1.association_learning(df)
hn1.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/marketing_data_online_retail_small/')
# hn1.plot()
# hn1.d3graph()
# hn1.heatmap()

df = pd.read_csv('D://PY//DATA//OTHER//waterpump//train_set_values.zip', sep=',')
hn2 = hnet(black_list=['date_recorded','id'])
results2 = hn2.association_learning(df)
hn2.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/waterpump/')
# hn2.plot()
# hn2.d3graph()
# hn2.heatmap()

df = pd.read_csv('D://PY//DATA//CLASSIF//FIFA 2018 Statistics.csv', sep=',')
hn3 = hnet()
results3 = hn3.association_learning(df)
hn3.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/fifa_2018/')
# hn3.plot()
# hn3.heatmap()

# df = pd.read_csv('D://PY//DATA//CANCER//cancer_xy.csv', sep=',')
# hn4 = hnet(black_list=['x','y','PC1','PC2'])
# results4 = hn4.association_learning(df)
# hn4.plot()
# hn4.plot(node_color='cluster', directed=True)
# hn4.plot(node_color='cluster', directed=False)
# hn4.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/cancer_d3graph/')
# hn4.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/cancer_d3graph_color/', node_color='cluster')
# hn4.d3graph(node_color='cluster', directed=False)
# hn4.d3graph(node_color='cluster', directed=True)
# hn4.heatmap()

# hn3.plot()
# hn3.plot(directed=False, black_list=['Man of the Match_No'], node_color='cluster')
# hn3.plot(directed=True, black_list=['Man of the Match_No'], node_color='cluster')
# hn3.plot(directed=True, black_list=['Man of the Match_No'])
# hn3.d3graph(directed=False, black_list=['Man of the Match_No'])
# hn3.heatmap(black_list=['Man of the Match_No'])
