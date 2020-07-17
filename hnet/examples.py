# import hnet
# print(dir(hnet))
# print(hnet.__version__)
import numpy as np
from hnet import hnet
hn = hnet()
df = hn.import_example('sprinkler')

hn1 = hnet(dtypes=np.array(['bool']*df.shape[1]))
out1 = hn1.association_learning(df.astype(bool))

hn2 = hnet()
out2 = hn2.association_learning(df.astype(bool))

# %%
import hnet
df = hnet.import_example('sprinkler')
out = hnet.enrichment(df.astype(bool), y=df.iloc[:,0].values)
print(out)


import hnet
dtypes = np.array(['bool']*df.shape[1])
df = hnet.import_example('sprinkler')
out = hnet.enrichment(df.astype(bool), y=df.iloc[:,0].values, dtypes=dtypes)
print(out)

# %%

# %%
# Load data
# df = hn.import_example('titanic')
hn = hnet()
df = hn.import_example('sprinkler')

hn = hnet()
out2 = hn.association_learning(df.astype(bool))


hn = hnet(excl_background='0.0')
out1 = hn.association_learning(df)
out3 = hn.association_learning(df.astype(int))
hn.plot()

hn = hnet(excl_background='False')
out5 = hn.association_learning(df.astype(bool).astype(str))

hn = hnet(excl_background='0')
out7 = hn.association_learning(df.astype(int).astype(str))

hn = hnet(excl_background='0.0')
out6 = hn.association_learning(df.astype(float).astype(str))

# Should raise exception
out4 = hn.association_learning(df.astype(float))


# %% Import class
from hnet import hnet
print(dir(hnet))

# %% Download dataset from url
# from tabulate import tabulate
import hnet
df = hnet.import_example(url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
df.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','earnings']
cols_as_float = ['age','hours-per-week','capital-loss','capital-gain']
df[cols_as_float]=df[cols_as_float].astype(float)

# print(tabulate(df.head(), tablefmt="grid", headers="keys"))

from hnet import hnet
hn = hnet(black_list=['fnlwgt'])
results = hn.association_learning(df)

# out = tabulate(hn.results['rules'].iloc[1:,:].head(), tablefmt="grid", headers="keys")

hn.d3graph()
hn.d3graph(min_edges=5)
# hn.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/income/')


# %% Import examples
import hnet

df = hnet.import_example('titanic')

df = hnet.import_example('student')

df = hnet.import_example('sprinkler')

# %% Run with default settings
from hnet import hnet

hn = hnet()
# Load data
# df = hn.import_example('titanic')
df = hn.import_example('sprinkler')
# Association learning
out = hn.association_learning(df.astype(bool))

# %% Plot with clustering nodes
G_static = hn.plot()
G_dynamic = hn.d3graph()

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
hn1 = hnet(y_min=50)
hn2 = hnet()
# Data
df = hn1.import_example('titanic')
# Association learning
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
# Association learning
out = hn.association_learning(df)

# %% Run with whitelist
from hnet import hnet
hn = hnet(white_list=['Survived', 'Pclass', 'Age', 'SibSp'])
# Load data
df = hn.import_example('titanic')
# Association learning
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

# %% Download dataset from url
from tabulate import tabulate
import hnet
df = hnet.import_example(url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
df.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','earnings']
cols_as_float = ['age','hours-per-week','capital-loss','capital-gain']
df[cols_as_float]=df[cols_as_float].astype(float)

# print(tabulate(df.head(), tablefmt="grid", headers="keys"))

from hnet import hnet
hn = hnet(black_list=['fnlwgt'])
results = hn.association_learning(df)

out = tabulate(hn.results['rules'].iloc[1:,:].head(), tablefmt="grid", headers="keys")

hn.d3graph(savepath='D://PY/REPOSITORIES/erdogant.github.io/docs/d3graph/income/')
hn.d3graph(min_edges=1)

# %% Covid-19 dataset
from hnet import hnet
import pandas as pd
df = pd.read_csv('D://covid19_us_county.zip')

dtypes = ['', 'cat', 'cat', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num']
hn = hnet(dtypes=dtypes)
results = hn.association_learning(df)
