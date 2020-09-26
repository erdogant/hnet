#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pypickle

#%% Load result-data
arg=dict()
#arg['n_sampling']=[1000,2000,3000,5000,8000,10000]
arg['n_sampling']=[100,1000,5000,10000]
arg['DAG']=['alarm']

#%%
#loadname='hnet_'+arg['DAG'][0]+'_'+str(10000)+'.pkl'
#out=pypickle.load(os.path.join('../PROJECTS/hnet/results/',loadname))
#out_hnets=out['out_hnets']
#modelTrue=out['model_true']
#out_bayes=out['out_bayes']
#
#
#golden_truth_directed=out['golden_truth_directed']
#golden_truth_undirected=out['golden_truth_undirected']
#hnet_directed=out['hnet_directed']
#hnet_undirected=out['hnet_undirected']
#bayes_directed=out['bayes_directed']
#bayes_undirected=out['bayes_undirected']
#random_directed=out['random_directed']
#random_undirected=out['random_undirected']
#
#
#df1 = pd.DataFrame(index=['f1_undirected','f1_directed','mcc_undirected','mcc_directed'], data=[], columns=['hnet','bayes','golden','random'])
#df1['hnet'].iloc[0]=hnet_undirected['f1']
#df1['bayes'].iloc[0]=bayes_undirected['f1']
#df1['golden'].iloc[0]=golden_truth_undirected['f1']
#df1['random'].iloc[0]=random_undirected['f1']
#df1['hnet'].iloc[1]=hnet_directed['f1']
#df1['bayes'].iloc[1]=bayes_directed['f1']
#df1['golden'].iloc[1]=golden_truth_directed['f1']
#df1['random'].iloc[1]=random_directed['f1']
#
#df1['hnet'].iloc[2]=hnet_undirected['MCC']
#df1['bayes'].iloc[2]=bayes_undirected['MCC']
#df1['golden'].iloc[2]=golden_truth_undirected['MCC']
#df1['random'].iloc[2]=random_undirected['MCC']
#df1['hnet'].iloc[3]=hnet_directed['MCC']
#df1['bayes'].iloc[3]=bayes_directed['MCC']
#df1['golden'].iloc[3]=golden_truth_directed['MCC']
#df1['random'].iloc[3]=random_directed['MCC']
#
#print(df1)

#%%
#out['df_method_comparison']

#%%
dagtype='alarm'
#dagtype='asia'
#scoretype='MCC'

columns=['hnet','bayes','golden','random','n_sampling','DAG']
df=pd.DataFrame()
df_score=pd.DataFrame()
dfScore=pd.DataFrame(columns=columns)
for n_sampling in arg['n_sampling']:
    loadname='hnet_'+dagtype+'_'+str(n_sampling)+'.pkl'
    d=pypickle.load(os.path.join('../PROJECTS/hnet/results/',loadname))

    # F1-score
    tmpdf=d['df_method_comparison_scores']
#    tmpdf=d['df_method_comparison']
    tmpdf['n_sampling']=d['arg']['n_sampling']
    tmpdf['DAG']=d['arg']['DAG']
    df=df.append(tmpdf)




    # F1-score
#    dfScore=pd.DataFrame(columns=columns)
#    score=[]
#    score.append([d['hnet_undirected']['f1'], d['bayes_undirected']['f1'], d['golden_truth_undirected']['f1'], d['random_undirected']['f1'], d['arg']['n_sampling'], d['arg']['DAG']])
#    dfScore=pd.concat([dfScore, pd.DataFrame(index=['f1_undirected'], data=np.array(score).reshape(1,-1), columns=columns)], axis=0)
#    score=[]
#    score.append([d['hnet_directed']['f1'], d['bayes_directed']['f1'], d['golden_truth_directed']['f1'], d['random_directed']['f1'], d['arg']['n_sampling'], d['arg']['DAG']])
#    dfScore=pd.concat([dfScore, pd.DataFrame(index=['f1_directed'], data=np.array(score).reshape(1,-1), columns=columns)], axis=0)
#    score=[]
#    score.append([d['hnet_undirected']['MCC'], d['bayes_undirected']['MCC'], d['golden_truth_undirected']['MCC'], d['random_undirected']['MCC'], d['arg']['n_sampling'], d['arg']['DAG']])
#    dfScore=pd.concat([dfScore, pd.DataFrame(index=['MCC_undirected'], data=np.array(score).reshape(1,-1), columns=columns)], axis=0)
#    score=[]
#    score.append([d['hnet_directed']['MCC'], d['bayes_directed']['MCC'], d['golden_truth_directed']['MCC'], d['random_directed']['MCC'], d['arg']['n_sampling'], d['arg']['DAG']])
#    dfScore=pd.concat([dfScore, pd.DataFrame(index=['MCC_directed'], data=np.array(score).reshape(1,-1), columns=columns)], axis=0)
#    df_score=df_score.append(dfScore)


#    df1 = pd.DataFrame(index=['f1_undirected','f1_directed','mcc_undirected','mcc_directed'], data=[], columns=['hnet','bayes','golden','random'])
#    df1['hnet'].iloc[0]=hnet_undirected['f1']
#    df1['bayes'].iloc[0]=bayes_undirected['f1']
#    df1['golden'].iloc[0]=golden_truth_undirected['f1']
#    df1['random'].iloc[0]=random_undirected['f1']
#    df1['hnet'].iloc[1]=hnet_directed['f1']
#    df1['bayes'].iloc[1]=bayes_directed['f1']
#    df1['golden'].iloc[1]=golden_truth_directed['f1']
#    df1['random'].iloc[1]=random_directed['f1']
#    df1['hnet'].iloc[2]=hnet_undirected['MCC']
#    df1['bayes'].iloc[2]=bayes_undirected['MCC']
#    df1['golden'].iloc[2]=golden_truth_undirected['MCC']
#    df1['random'].iloc[2]=random_undirected['MCC']
#    df1['hnet'].iloc[3]=hnet_directed['MCC']
#    df1['bayes'].iloc[3]=bayes_directed['MCC']
#    df1['golden'].iloc[3]=golden_truth_directed['MCC']
#    df1['random'].iloc[3]=random_directed['MCC']
#    df1['n_sampling']=d['arg']['n_sampling']
#    df1['DAG']=d['arg']['DAG']
#    df_score=df_score.append(df1)

#    tmpdf.loc['f1_directed','hnet']=d['hnet_directed'][scoretype]
#    tmpdf.loc['f1_undirected','hnet']=d['hnet_undirected'][scoretype]
#    tmpdf.loc['f1_directed','bayes']=d['bayes_directed'][scoretype]
#    tmpdf.loc['f1_undirected','bayes']=d['bayes_undirected'][scoretype]
#    tmpdf.loc['f1_directed','random']=d['random_directed'][scoretype]
#    tmpdf.loc['f1_undirected','random']=d['random_undirected'][scoretype]
#    tmpdf.loc['f1_directed','golden']=np.nan
#    tmpdf.loc['f1_undirected','golden']=np.nan

#df.iloc[:,0:4]= -np.log10(df.iloc[:,0:4])
#values=df.iloc[:,0:4].values.flatten()
#I=values==np.inf
#maxP=np.sort(values[I==False])[-1]+1
#df=df.replace(np.inf, maxP)


#%%
print('directed:   HNet   MCC score: mean=%g +- %g' %(df.loc['mcc_directed','hnet'].mean(), df.loc['mcc_directed','hnet'].var()))
print('directed:   Bayes  MCC score: mean=%g +- %g' %(df.loc['mcc_directed','bayes'].mean(), df.loc['mcc_directed','bayes'].var()))
print('directed:   Random MCC score: mean=%g +- %g' %(df.loc['mcc_directed','random'].mean(), df.loc['mcc_directed','random'].var()))

print('undirected: HNet   MCC score: mean=%g +- %g' %(df.loc['mcc_undirected','hnet'].mean(), df.loc['mcc_undirected','hnet'].var()))
print('undirected: Bayes  MCC score: mean=%g +- %g' %(df.loc['mcc_undirected','bayes'].mean(), df.loc['mcc_undirected','bayes'].var()))
print('undirected: Golden MCC score: mean=%g +- %g' %(df.loc['mcc_undirected','golden'].mean(), df.loc['mcc_undirected','golden'].var()))
print('undirected: Random MCC score: mean=%g +- %g' %(df.loc['mcc_undirected','random'].mean(), df.loc['mcc_undirected','random'].var()))

#%%
#out['hnet_directed']['MCC']
#print(df_score)
#df=df_score
#%% Make plot all random 
#plt.figure(figsize=(15,8))
#plt.plot(df.loc['f1_undirected','n_sampling'].values, df.loc['f1_undirected','random'].values, label='Random F1 undirected')
#plt.plot(df.loc['f1_directed','n_sampling'].values, df.loc['f1_directed','random'].values, label='Random F1 directed')
#plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','random'].values, label='Random MCC undirected')
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','random'].values, label='Random MCC directed')
#
#plt.xlabel('Sampling (n)')
#plt.ylabel('Pvalue')
#plt.grid(True)
#plt.title(df.DAG.unique()[0])
#plt.legend(loc=1)

#%% Make plot
# F1-score

plt.figure(figsize=(15,8))
plt.plot(df.loc['f1_undirected','n_sampling'].values, df.loc['f1_undirected','hnet'].values, label='HNET', c='red', linestyle='--')
plt.plot(df.loc['f1_directed','n_sampling'].values, df.loc['f1_directed','hnet'].values, label='HNET (directed)', c='red', linestyle='-')
plt.plot(df.loc['f1_undirected','n_sampling'].values, df.loc['f1_undirected','bayes'].values, label='Bayes', c='b', linestyle='--')
plt.plot(df.loc['f1_directed','n_sampling'].values, df.loc['f1_directed','bayes'].values, label='Bayes (directed)', c='b', linestyle='-')
plt.plot(df.loc['f1_undirected','n_sampling'].values, df.loc['f1_undirected','golden'].values, label='Golden truth', c='orange', linestyle='--')
#plt.plot(df.loc['f1_directed','n_sampling'].values, df.loc['f1_directed','golden'].values, label='Golden truth (directed)', c='orange', linestyle='-')
plt.plot(df.loc['f1_undirected','n_sampling'].values, df.loc['f1_undirected','random'].values, label='Random ', c='k', linestyle='--')
plt.plot(df.loc['f1_directed','n_sampling'].values, df.loc['f1_directed','random'].values, label='Random (directed)', linestyle='-')
plt.xlabel('Sampling (n)')
plt.ylabel('Pvalue (F1)')
plt.grid(True)
#plt.title(df.DAG.unique()[0])
plt.legend(loc=1)

#%% Make plot
# MCC-score
# Undirected
plt.figure(figsize=(15,8))
plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','hnet'].values, label='HNET', c='red', linestyle='--')
plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','hnet'].values, label='HNET (directed)', c='red', linestyle='-')
plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','bayes'].values, label='Bayes', c='b', linestyle='--')
plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','bayes'].values, label='Bayes (directed)', c='b', linestyle='-')
plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','golden'].values, label='Golden truth', c='orange', linestyle='--')
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','golden'].values, label='Golden truth (directed)', c='orange', linestyle='-')
plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','random'].values, label='Random ', c='k', linestyle='--')
plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','random'].values, label='Random (directed)',c='grey', linestyle='-')
plt.xlabel('Sampling (N)')
plt.ylabel('MCC')
plt.grid(True)
#plt.title(df.DAG.unique()[0])
plt.legend(loc=1)
#
#plt.figure(figsize=(15,8))
#plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','hnet'].values, label='HNET')
#plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','bayes'].values, label='Bayes')
#plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','golden'].values, label='Golden truth')
#plt.plot(df.loc['mcc_undirected','n_sampling'].values, df.loc['mcc_undirected','random'].values, label='Random')
#plt.xlabel('Sampling (n)')
#plt.ylabel('MCC undirected pvalue')
#plt.grid(True)
#plt.title(df.DAG.unique()[0])
#plt.legend()

#%% Make plot
# MCC-score
# Undirected
#plt.figure(figsize=(15,8))
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','hnet'].values, label='HNET')
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','bayes'].values, label='Bayes')
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','golden'].values, label='Golden truth')
#plt.plot(df.loc['mcc_directed','n_sampling'].values, df.loc['mcc_directed','random'].values, label='Random')
#plt.xlabel('Sampling (n)')
#plt.ylabel('MCC directed pvalue')
#plt.grid(True)
#plt.title(df.DAG.unique()[0])
#plt.legend()

#%% Do runtme stuff
gettype='directed'
out=[]
#gettype='undirected'

for n_sampling in arg['n_sampling']:
    loadname='hnet_'+dagtype+'_'+str(n_sampling)+'.pkl'
    d=pypickle.load(os.path.join('../PROJECTS/hnet/results/',loadname))
    dict1={'bayes_runtime':d['arg']['bayes_runtime'],'hnet_runtime':d['arg']['hnet_runtime'], 'n_sampling':d['arg']['n_sampling']}
    if gettype=='directed':
        dict1.update(dict(zip('hnet_'+d['hnet_directed_bestmodel_args'].index.values, d['hnet_directed_bestmodel_args'].values)))
    else:
        dict1.update(dict(zip('hnet_'+d['hnet_undirected_bestmodel_args'].index.values, d['hnet_undirected_bestmodel_args'].values)))
    out.append(dict1)
    
df=pd.DataFrame(out)


plt.figure(figsize=(15,8))
#plt.plot(figsize=(10,15))
Xvalues=df.loc[:,'n_sampling'].values
plt.plot(Xvalues, df.loc[:,'hnet_runtime'].values, label='HNET', c='r')
plt.plot(Xvalues, df.loc[:,'bayes_runtime'].values, label='Bayes', c='b')
#plt.plot(df.loc['hnet_mcc_directed','hnet_n_sampling'].values, df.loc['hnet_mcc_directed','hnet_random'].values, label='Random')
plt.xlabel('Sampling (N)')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
#plt.title(gettype)
plt.legend()

#%%
