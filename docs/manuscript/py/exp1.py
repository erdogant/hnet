# %% Load data
import time
from STATS.hypotesting import hypotesting
import pypickle
import STATS.hnet as hnet
import NETWORKS.network as network
import STATS.bayes as bayes
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
%reset -f


# %%
k=1
# it=[100,1000,2000,3000,5000,8000,10000,100000]
it=[100, 1000, 2000, 5000, 10000]
it[k]

# %% Arguments
arg=dict()
arg['n_sampling'] = it[k]
#arg['n_sampling'] = 100
arg['alpha'] = 0.05
arg['multtest'] = 'holm'
#arg['DAG']        = 'asia'
#arg['DAG']        = 'sachs'
#arg['DAG']        = 'sprinkler'
arg['DAG'] = 'alarm'
#arg['DAG']        = 'andes'

# %% Get TRUE model
modelTrue = bayes.DAG_example(arg['DAG'], verbose=0)
modelTrue['G'] = bayes.plot(modelTrue)

# %% Sampling using the DAG to create dataframe
df = bayes.sampling(modelTrue, n=arg['n_sampling'])

# %% Association learning hnets
start = time.time()
out_hnets = hnet.main(df, multtest=arg['multtest'], alpha=arg['alpha'], drop_empty=False, excl_background=['0.0'])
runtime = time.time() -start
arg['hnet_runtime'] = runtime

# Structure learning bayes
start = time.time()
out_bayes = bayes.structure_learning(df, methodtype='hc', scoretype='bic')
runtime = time.time() -start
arg['bayes_runtime'] = runtime

#out_asso = hnet.association_rules(out_hnets)

# %% Make directed model
modelTrue['adjmat_undirected'] = bayes.to_undirected(modelTrue['adjmat'])
#out_bayes['adjmat_undirected'] = bayes.to_undirected(out_bayes['adjmat'])
_=bayes.plot(modelTrue['adjmat_undirected'], pos=modelTrue['G']['pos'])
_=bayes.plot(modelTrue['adjmat'], pos=modelTrue['G']['pos'])

# %% Restructure adjmat of HNET to make comparison with bayesian model
adjmat=out_hnets['simmatLogP'].copy()
Icol=adjmat.columns.str.contains('_0.0')
Iidx=adjmat.index.str.contains('_0.0')
Icol=adjmat.columns.str.contains('_0')
Iidx=adjmat.index.str.contains('_0')
adjmat=adjmat.loc[~Iidx, ~Icol]
adjmat.columns=adjmat.columns.str.replace('_1', '', regex=True)
adjmat.index=adjmat.index.str.replace('_1', '', regex=True)
np.fill_diagonal(adjmat.values, 0)
out_hnets['adjmat']=adjmat

# %% Compute scores
[golden_truth_directed, _] = bayes.compare_networks(modelTrue['adjmat'], modelTrue['adjmat'], showfig=False, verbose=0)
[golden_truth_undirected, _] = bayes.compare_networks(modelTrue['adjmat'], modelTrue['adjmat_undirected'], showfig=False, verbose=0)
# HNET compared to undirected edges
[hnet_directed, _] = bayes.compare_networks(modelTrue['adjmat'], out_hnets['adjmat'], showfig=False, verbose=0)
[hnet_undirected, _] = bayes.compare_networks(modelTrue['adjmat_undirected'], out_hnets['adjmat'], showfig=False, verbose=0)
# BAYES compared to DIRECTED edges
[bayes_directed, _] = bayes.compare_networks(modelTrue['adjmat'], out_bayes['adjmat'], showfig=False, verbose=0)
[bayes_undirected, _] = bayes.compare_networks(modelTrue['adjmat_undirected'], out_bayes['adjmat'], showfig=False, verbose=0)
# RANDOM
random_adjmat = pd.DataFrame(data=np.random.randint(0, 2, size=modelTrue['adjmat'].shape, dtype=bool), index=modelTrue['adjmat'].index, columns=modelTrue['adjmat'].columns)
[random_directed, _] = bayes.compare_networks(modelTrue['adjmat'], random_adjmat, showfig=False, verbose=0)
[random_undirected, _] = bayes.compare_networks(modelTrue['adjmat'], random_adjmat, showfig=False, verbose=0)

metrics=['f1', 'MCC', 'average_precision']
for metric in metrics:
    print('[%s] GOLDEN TRUTH (directed): %.3f' %(metric, golden_truth_directed[metric]))
    print('[%s] GOLDEN TRUTH (undirected): %.3f' %(metric, golden_truth_undirected[metric]))
    print('[%s] HNET (directed): %.3f' %(metric, hnet_directed[metric]))
    print('[%s] HNET (undirected): %.3f' %(metric, hnet_undirected[metric]))
    print('[%s] BAYES (directed: %.3f' %(metric, bayes_directed[metric]))
    print('[%s] BAYES (undirected: %.3f' %(metric, bayes_undirected[metric]))
    print('[%s] RANDOM (directed: %.3f' %(metric, random_directed[metric]))
    print('[%s] RANDOM (undirected: %.3f' %(metric, random_undirected[metric]))
    print('')

# %%
df1 = pd.DataFrame(index=['f1_undirected', 'f1_directed', 'mcc_undirected', 'mcc_directed'], data=[], columns=['hnet', 'bayes', 'golden', 'random'])
df1['hnet'].iloc[0]=hnet_undirected['f1']
df1['bayes'].iloc[0]=bayes_undirected['f1']
df1['golden'].iloc[0]=golden_truth_undirected['f1']
df1['random'].iloc[0]=random_undirected['f1']
df1['hnet'].iloc[1]=hnet_directed['f1']
df1['bayes'].iloc[1]=bayes_directed['f1']
df1['golden'].iloc[1]=golden_truth_directed['f1']
df1['random'].iloc[1]=random_directed['f1']
df1['hnet'].iloc[2]=hnet_undirected['MCC']
df1['bayes'].iloc[2]=bayes_undirected['MCC']
df1['golden'].iloc[2]=golden_truth_undirected['MCC']
df1['random'].iloc[2]=random_undirected['MCC']
df1['hnet'].iloc[3]=hnet_directed['MCC']
df1['bayes'].iloc[3]=bayes_directed['MCC']
df1['golden'].iloc[3]=golden_truth_directed['MCC']
df1['random'].iloc[3]=random_directed['MCC']

df_method_comparison_scores=df1.copy()
print(df_method_comparison_scores)

# %%
network.compare_networks(modelTrue['adjmat'], out_hnets['adjmat'])

# %% CREATE NULL DISTRIBUTION WITH PERMUTED LABELS
n=100
scores_rand=np.zeros((n, 6), dtype=float)

# Compute random scores
for i in tqdm(range(0, n)):
    # Create random dataframe
    random_adjmat = np.random.permutation(modelTrue['adjmat_undirected'].values)
#    random_adjmat = np.random.randint(0,2,size=modelTrue['adjmat'].shape, dtype=bool)
    random_adjmat = pd.DataFrame(data=random_adjmat, index=modelTrue['adjmat'].index, columns=modelTrue['adjmat'].columns)
    # Compare networks directed as golden truth
    [random_accuracy_directed, _]=bayes.compare_networks(modelTrue['adjmat'], random_adjmat, showfig=False, verbose=0)
    # Compare networks undirected as golden truth
    [random_accuracy_undirected, _]=bayes.compare_networks(modelTrue['adjmat_undirected'], random_adjmat, showfig=False, verbose=0)
    # Store
    scores_rand[i, 0]=random_accuracy_directed['f1']
    scores_rand[i, 1]=random_accuracy_undirected['f1']
    scores_rand[i, 2]=random_accuracy_directed['average_precision']
    scores_rand[i, 3]=random_accuracy_undirected['average_precision']
    scores_rand[i, 4]=random_accuracy_directed['MCC']
    scores_rand[i, 5]=random_accuracy_undirected['MCC']

# Make DF
scores_rand = pd.DataFrame(data=scores_rand, columns=['f1_directed', 'f1_undirected', 'ap_directed', 'ap_undirected', 'mcc_directed', 'mcc_undirected'])

# %% Compute P-value compared to random distribution
P_f1_directed = hypotesting([hnet_directed['f1'], bayes_directed['f1'], golden_truth_directed['f1'], random_directed['f1']], scores_rand['f1_directed'].values, bound='up')
P_f1_undirected = hypotesting([hnet_undirected['f1'], bayes_undirected['f1'], golden_truth_undirected['f1'], random_undirected['f1']], scores_rand['f1_undirected'].values, bound='up')
P_mcc_directed = hypotesting([hnet_directed['MCC'], bayes_directed['MCC'], golden_truth_directed['MCC'], random_directed['MCC']], scores_rand['mcc_directed'].values, bound='up')
P_mcc_undirected = hypotesting([hnet_undirected['MCC'], bayes_undirected['MCC'], golden_truth_undirected['MCC'], random_undirected['MCC']], scores_rand['mcc_undirected'].values, bound='up')
#P_ap_directed    = hypotesting([hnet_directed['average_precision'],bayes_directed['average_precision'], golden_truth_directed['average_precision'], random_directed['average_precision']], scores_rand['ap_directed'].values, bound='up')
#P_ap_undirected  = hypotesting([hnet_undirected['average_precision'], bayes_undirected['average_precision'], golden_truth_undirected['average_precision'], random_undirected['average_precision']], scores_rand['ap_undirected'].values, bound='up')

df_method_comparison = pd.DataFrame(index=['f1_undirected'], data=P_f1_undirected['Praw'].reshape(1, -1), columns=['hnet', 'bayes', 'golden', 'random'])
df_method_comparison = df_method_comparison.append(pd.DataFrame(index=['f1_directed'], data=P_f1_directed['Praw'].reshape(1, -1), columns=['hnet', 'bayes', 'golden', 'random']))
df_method_comparison = df_method_comparison.append(pd.DataFrame(index=['mcc_directed'], data=P_mcc_directed['Praw'].reshape(1, -1), columns=['hnet', 'bayes', 'golden', 'random']))
df_method_comparison = df_method_comparison.append(pd.DataFrame(index=['mcc_undirected'], data=P_mcc_undirected['Praw'].reshape(1, -1), columns=['hnet', 'bayes', 'golden', 'random']))
#df_method_comparison = df_method_comparison.append(pd.DataFrame(index=['ap_directed'], data=P_ap_directed['Praw'].reshape(1,-1), columns=['hnet','bayes','golden','random']))
#df_method_comparison = df_method_comparison.append(pd.DataFrame(index=['ap_undirected'], data=P_ap_undirected['Praw'].reshape(1,-1), columns=['hnet','bayes','golden','random']))

# %% HNET compared to undirected edges
tmpEdges=out_hnets['adjmat'].stack().reset_index()
minvalues=np.unique(tmpEdges[0])

HNETGradientScores = data=np.zeros((len(minvalues), 2))
HNETGradientScoresDirected=[]
HNETGradientScoresUnDirected=[]

# Set adjmat labels false in a gradient way
for i, minvalue in enumerate(tqdm(minvalues)):
    tmpAdjmat = out_hnets['adjmat'][out_hnets['adjmat']>minvalue].fillna(0)
    [hnet_directed, _] = bayes.compare_networks(modelTrue['adjmat'], tmpAdjmat, showfig=False, verbose=0)
    [hnet_undirected, _] = bayes.compare_networks(modelTrue['adjmat_undirected'], tmpAdjmat, showfig=False, verbose=0)
    HNETGradientScoresDirected.append(hnet_directed)
    HNETGradientScoresUnDirected.append(hnet_undirected)
    HNETGradientScores[i, 0]=np.sum(np.sum(out_hnets['adjmat']>minvalue))
    HNETGradientScores[i, 1]=minvalue

# %% MAke plot for the gradient results
HNETGradientScores = pd.DataFrame(data=HNETGradientScores, columns=['edges', 'minP'])
HNETGradientScores['f1_directed'] = list(map(lambda x: x['f1'], HNETGradientScoresDirected))
HNETGradientScores['f1_undirected'] = list(map(lambda x: x['f1'], HNETGradientScoresUnDirected))
HNETGradientScores['mcc_directed'] = list(map(lambda x: x['MCC'], HNETGradientScoresDirected))
HNETGradientScores['mcc_undirected'] = list(map(lambda x: x['MCC'], HNETGradientScoresUnDirected))
#HNETGradientScores['ap_directed'] = list(map(lambda x: x['average_precision'], HNETGradientScoresDirected))
#HNETGradientScores['ap_undirected'] = list(map(lambda x: x['average_precision'], HNETGradientScoresUnDirected))
HNETGradientScores['nrCorrect_directed'] = list(map(lambda x: x['confmatrix'][0, 1] +x['confmatrix'][1, 0], HNETGradientScoresDirected))
HNETGradientScores['nrCorrect_undirected'] = list(map(lambda x: x['confmatrix'][0, 1] +x['confmatrix'][1, 0], HNETGradientScoresUnDirected))
HNETGradientScores['nrWrong_directed'] = list(map(lambda x: x['confmatrix'][0, 0] +x['confmatrix'][1, 1], HNETGradientScoresDirected))
HNETGradientScores['nrWrong_undirected'] = list(map(lambda x: x['confmatrix'][0, 0] +x['confmatrix'][1, 1], HNETGradientScoresUnDirected))

# %% Store model
out=dict()
# Arguments
out['arg']=arg
# Model detected
out['model_true']=modelTrue
out['out_hnets']=out_hnets
out['out_bayes']=out_bayes
# Comparison model with true model for (un)directed edges
out['golden_truth_directed']=golden_truth_directed
out['golden_truth_undirected']=golden_truth_undirected
out['hnet_directed']=hnet_directed
out['hnet_undirected']=hnet_undirected
out['bayes_directed']=bayes_directed
out['bayes_undirected']=bayes_undirected
out['random_directed']=random_directed
out['random_undirected']=random_undirected
# Compare scoring by cutting edges for hnet using min pvalue
out['HNETGradientScoresDirected']=HNETGradientScoresDirected
out['HNETGradientScoresUnDirected']=HNETGradientScoresUnDirected
out['HNETGradientScores']=HNETGradientScores
out['df_method_comparison']=df_method_comparison
out['df_method_comparison_scores']=df_method_comparison_scores
# Random scores
out['scores_rand']=scores_rand
# Best performing results
idx=HNETGradientScores['mcc_undirected'].argmax()
out['hnet_undirected_bestmodel_args']=HNETGradientScores.iloc[idx, :]
out['hnet_directed_bestmodel_args']=HNETGradientScores.iloc[idx, :]
out['hnet_undirected_bestmodel']=out_hnets['adjmat'][out_hnets['adjmat']>HNETGradientScores.iloc[idx, :]['minP']].fillna(0)

# %% Save
savename='hnet_' +arg['DAG'] +'_' +str(arg['n_sampling']) +'.pkl'
pypickle.save(os.path.join('../PROJECTS/hnet/results/', savename), out)

# %%
print(arg)
print(df_method_comparison)
print(df_method_comparison_scores)

# %% Plot all results with these arguments
# Real model
_ = bayes.plot(modelTrue, pos=modelTrue['G']['pos'])
# Learned model using bayes
_=bayes.plot(out_bayes['adjmat'], pos=modelTrue['G']['pos'])
# Learned model using hnets
_=bayes.plot(out_hnets['adjmat']>0, pos=modelTrue['G']['pos'])

# %% MAKE FIGURE
fig, ax=plt.subplots(1, 2, figsize=(40, 12))
ax[0].plot(HNETGradientScores['edges'], HNETGradientScores['mcc_directed'], label='MCC directed')
ax[0].plot(HNETGradientScores['edges'], HNETGradientScores['mcc_undirected'], label='MCC undirected')
ax[0].set_xlabel('Number of edges')
ax[0].set_ylabel('MCC score')
ax[0].legend()

ax[1].plot(HNETGradientScores['edges'], HNETGradientScores['nrCorrect_directed'], label='Nr correct directed')
ax[1].plot(HNETGradientScores['edges'], HNETGradientScores['nrCorrect_undirected'], label='Nr. correct undirected')
ax[1].plot(HNETGradientScores['edges'], HNETGradientScores['nrWrong_directed'], label='Nr wrong directed')
ax[1].plot(HNETGradientScores['edges'], HNETGradientScores['nrWrong_undirected'], label='Nr. wrong undirected')
ax[1].set_xlabel('Number of edges')
ax[1].set_ylabel('Number of (in)correct')
ax[1].legend()

plt.grid(True)
plt.show()

# %% GET BEST SCORING AND COMPARE RESULTS TO REAL
# Get best scoring HNET model
idx=HNETGradientScores['mcc_undirected'].argmax()
tmpAdjmat = out_hnets['adjmat'][out_hnets['adjmat']>HNETGradientScores.iloc[idx, :]['minP']].fillna(0)
#[hnet_directed, _] = bayes.compare_networks(modelTrue['adjmat'], tmpAdjmat,showfig=True, verbose=0)

# MAKE REAL PLOT AND STORE THE GRAPH
G=bayes.plot(modelTrue['adjmat'])

# Difference in network REAL vs PREDICTED
[hnet_undirected, _] = bayes.compare_networks(modelTrue['adjmat_undirected'], tmpAdjmat, showfig=True, pos=G['pos'], verbose=0)

# PREDICTED MODEL
_=bayes.plot(tmpAdjmat>0, pos=G['pos'])

# REAL MODEL
_=bayes.plot(modelTrue['adjmat'], pos=G['pos'])

# %%
#[score0,adjmat]=bayes.compare_networks(modelTrue['adjmat'], modelTrue['adjmat'], pos=G['pos'], showfig=True, verbose=0)
#[score1,adjmat]=bayes.compare_networks(modelTrue['adjmat'], out_bayes['adjmat'], pos=G['pos'], showfig=True, verbose=0)
##[score1,adjmat]=bayes.compare_networks(modelTrue['adjmat_undirected'], out_bayes['adjmat'], pos=G['pos'], showfig=True, verbose=0)
#[score2,adjmat]=bayes.compare_networks(modelTrue['adjmat'], out_hnets['adjmat'], pos=G['pos'], showfig=True, verbose=0)
#[score3,adjmat]=bayes.compare_networks(modelTrue['adjmat_undirected'], out_hnets['adjmat'], pos=G['pos'], showfig=True, verbose=0)

#from EMBEDDINGS.flameplot import flameplot
#out1 = flameplot(modelTrue['adjmat'].astype(int), (adjmat>0).astype(int), nn=50, steps=1)
