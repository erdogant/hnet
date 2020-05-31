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

    # [hnet] >Removing features from the black list..
    # [DTYPES] Auto detecting dtypes
    # [DTYPES] [age]             > [float]->[num] [74]
    # [DTYPES] [sex]             > [obj]  ->[cat] [2]
    # [DTYPES] [survival_months] > [force]->[num] [1591]
    # [DTYPES] [death_indicator] > [float]->[num] [2]
    # [DTYPES] [labx]            > [obj]  ->[cat] [19]
    # [DTYPES] Setting dtypes in dataframe
    # [DF2ONEHOT] Working on age
    # [DF2ONEHOT] Working on sex.....[3]
    # [DF2ONEHOT] Working on survival_months
    #   0%|          | 0/22 [00:00<?, ?it/s]
    # [DF2ONEHOT] Working on labx.....[19]
    # [DF2ONEHOT] Total onehot features: 22
    # [hnet] >Association learning across [22] categories.
    # 100%|██████████| 22/22 [00:07<00:00,  2.77it/s]
    # [hnet] >Total number of computations: [969]
    # [hnet] >Multiple test correction using holm
    # [hnet] >Dropping age
    # [hnet] >Dropping survival_months
    # [hnet] >Dropping death_indicator
    # [hnet] >Fin.



# %%
from hnet import hnet
import pandas as pd
from tabulate import tabulate


df = hn.import_example('cancer')
hn = hnet(black_list=['tsneX','tsneY','PC1','PC2'])
results = hn.association_learning(df)
hn.d3graph(black_list=['sex'])
out = tabulate(hn.results['rules'].iloc[1:,:].head(), tablefmt="grid", headers="keys")
print(tabulate(df.head(), tablefmt="grid", headers="keys"))

# %% Small retail
df = hn.import_example('retail')
hn1 = hnet()
results1 = hn1.association_learning(df)
hn1.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/marketing_data_online_retail_small/')
# hn1.plot()
# hn1.d3graph()
# hn1.heatmap()

# %%

df = hn.import_example('waterpump')
hn2 = hnet(black_list=['id','longitude','latitude'])
results2 = hn2.association_learning(df)
hn2.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/waterpump/')
# hn2.plot()
# hn2.d3graph()
# hn2.heatmap()

# df['id'].head().values

# %%
df = hn.import_example('fifa')


hn3 = hnet(dtypes=['None', 'cat', 'cat', 'cat', 'num',
       'num', 'num', 'num', 'num', 'num', 'num',
       'num', 'num', 'num', 'num',
       'num', 'cat', 'cat',
       'cat', 'cat', 'cat', 'cat', 'cat', 'cat',
       'cat', 'cat', 'num'])

results3 = hn3.association_learning(df)
hn3.d3graph()
# hn3.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/fifa_2018/')
# hn3.plot()
# hn3.heatmap()

out = tabulate(hn3.results['rules'].iloc[1:,:].head(), tablefmt="grid", headers="keys")
print(tabulate(df.head(), tablefmt="grid", headers="keys"))

# hn3.plot()
# hn3.plot(directed=False, black_list=['Man of the Match_No'], node_color='cluster')
# hn3.plot(directed=True, black_list=['Man of the Match_No'], node_color='cluster')
# hn3.plot(directed=True, black_list=['Man of the Match_No'])
# hn3.d3graph(directed=False, black_list=['Man of the Match_No'])
# hn3.heatmap(black_list=['Man of the Match_No'])
