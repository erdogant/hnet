import STATS.hnet as hnet
import pandas as pd
import NETWORKS.network as network
import pandas as pd
import STATS.bayes as bayes
import STATS.hnet as hnet
import numpy as np

#%%
df_100   = pd.read_csv('../DATA/NETWORKS/bayesian/SPRINKLER/sprinkler_data_100.zip')
df_1000  = pd.read_csv('../DATA/NETWORKS/bayesian/SPRINKLER/sprinkler_data_1000.zip')
df_5000  = pd.read_csv('../DATA/NETWORKS/bayesian/SPRINKLER/sprinkler_data_5000.zip')
df_10000 = pd.read_csv('../DATA/NETWORKS/bayesian/SPRINKLER/sprinkler_data_10000.zip')
#out= hnet.main(df, alpha=0.05, multtest='holm', excl_background=['0.0'])
out_100= hnet.main(df_100, alpha=0.05, multtest='holm', drop_empty=False)
out_1000= hnet.main(df_1000, alpha=0.05, multtest='holm', drop_empty=False)
out_5000= hnet.main(df_5000, alpha=0.05, multtest='holm', drop_empty=False)
out_10000= hnet.main(df_10000, alpha=0.05, multtest='holm', drop_empty=False)
#G  = hnet.plot_network(out, dist_between_nodes=0.1, scale=2)

np.sum(np.sum((out_1000['simmatP']<=0.05)==(out_5000['simmatP']<=0.05)))
np.sum(np.sum((out_5000['simmatP']<=0.05)==(out_10000['simmatP']<=0.05)))
out_1000['simmatP'].shape[0]*out_1000['simmatP'].shape[1]

G1  = hnet.plot_network(out_100)
G2  = hnet.plot_network(out_1000)
G3  = hnet.plot_network(out_5000, pos=G2['pos'])
G4  = hnet.plot_network(out_10000, pos=G3['pos'])

#G=bayes.plot(modelTrue['adjmat'])

_  = hnet.plot_heatmap(out, savepath='c://temp/sprinkler/', cluster=False)
#Gd3  = hnet.plot_d3graph(out, savepath='c://temp/sprinkler/', directed=True)


#compareout=network.compare_networks(out_100['simmatP']<=0.05, out_1000['simmatP']<=0.05)
#%%

#### CREATE SPRINKLER DAG #####
model = bayes.DAG_example('sprinkler')
bayes.plot(model)

totals1=makerun(model, out_10000['simmatP'], 'holm')
totals2=makerun(model, out_10000['simmatP'], 'holm')
totals3=makerun(model, out_10000['simmatP'], 'holm')

totals4=makerun(model, out_10000['simmatP'], 'bonferroni')
totals5=makerun(model, out_10000['simmatP'], 'bonferroni')
totals6=makerun(model, out_10000['simmatP'], 'bonferroni')

totals7=makerun(model, out_10000['simmatP'], 'fdr_bh')
totals8=makerun(model, out_10000['simmatP'], 'fdr_bh')
totals9=makerun(model, out_10000['simmatP'], 'fdr_bh')

#%% Combine data
df_edges_holm             = pd.concat([pd.DataFrame(totals1).iloc[:,1], pd.DataFrame(totals2).iloc[:,1], pd.DataFrame(totals3).iloc[:,1]], axis=1)
df_edges_holm.index       = pd.DataFrame(totals1).iloc[:,0]
df_edges_bonferroni       = pd.concat([pd.DataFrame(totals4).iloc[:,1], pd.DataFrame(totals5).iloc[:,1], pd.DataFrame(totals6).iloc[:,1]], axis=1)
df_edges_bonferroni.index = pd.DataFrame(totals4).iloc[:,0]
df_edges_fdr_bh           = pd.concat([pd.DataFrame(totals7).iloc[:,1], pd.DataFrame(totals8).iloc[:,1], pd.DataFrame(totals9).iloc[:,1]], axis=1)
df_edges_fdr_bh.index     = pd.DataFrame(totals7).iloc[:,0]
# Make figure
makefig(df_edges_holm, df_edges_bonferroni, df_edges_fdr_bh, ylabel='Number of detected edges')
# Combine data
df_edges_holm             = pd.concat([pd.DataFrame(totals1).iloc[:,2], pd.DataFrame(totals2).iloc[:,2], pd.DataFrame(totals3).iloc[:,2]], axis=1)
df_edges_holm             = df_edges_holm/(out_10000['simmatP'].shape[0]*out_10000['simmatP'].shape[1])
df_edges_holm.index       = pd.DataFrame(totals1).iloc[:,0]
df_edges_bonferroni       = pd.concat([pd.DataFrame(totals4).iloc[:,2], pd.DataFrame(totals5).iloc[:,2], pd.DataFrame(totals6).iloc[:,2]], axis=1)
df_edges_bonferroni       = df_edges_bonferroni/(out_10000['simmatP'].shape[0]*out_10000['simmatP'].shape[1])
df_edges_bonferroni.index = pd.DataFrame(totals4).iloc[:,0]
df_edges_fdr_bh           = pd.concat([pd.DataFrame(totals7).iloc[:,2], pd.DataFrame(totals8).iloc[:,2], pd.DataFrame(totals9).iloc[:,2]], axis=1)
df_edges_fdr_bh           = df_edges_fdr_bh/(out_10000['simmatP'].shape[0]*out_10000['simmatP'].shape[1])
df_edges_fdr_bh.index     = pd.DataFrame(totals7).iloc[:,0]
# Make figure
makefig(df_edges_holm, df_edges_bonferroni, df_edges_fdr_bh, ylabel='Percentage similar with model>1000 samples')

#%%
def makefig(df_edges_holm, df_edges_bonferroni, df_edges_fdr_bh, ylabel=''):
    plt.figure(figsize=(10,6))
    
    SMALL_SIZE  = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    
    plt.rc('font',  size      = SMALL_SIZE)   # controls default text sizes
    plt.rc('axes',  titlesize = SMALL_SIZE)   # fontsize of the axes title
    plt.rc('xtick', labelsize = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('legend',fontsize  = SMALL_SIZE)   # legend fontsize
    plt.rc('axes',  labelsize = MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('figure',titlesize = BIGGER_SIZE)  # fontsize of the figure title
    axis_font = {'fontname':'Arial'}
    # axis_font = {'fontname':'Calibri Light', 'size':'14'}
    #plt.style.use('ggplot')
    #plt.style.use('default')
    
    
    plt.plot(df_edges_holm.index, df_edges_holm.mean(axis=1), 'g', label='Holm')
    #plt.fill_between(df_edges_holm.index, df_edges_holm.min(axis=1), df_edges_holm.max(axis=1), color='g', alpha=0.2)
    plt.plot(df_edges_bonferroni.index, df_edges_bonferroni.mean(axis=1), 'b', label='Bonferroni')
    #plt.fill_between(df_edges_bonferroni.index, df_edges_bonferroni.min(axis=1), df_edges_bonferroni.max(axis=1), color='b', alpha=0.2)
    plt.plot(df_edges_fdr_bh.index, df_edges_fdr_bh.mean(axis=1), 'r', label='Benjamini/Hochberg')
    #plt.fill_between(df_edges_fdr_bh.index, df_edges_fdr_bh.min(axis=1), df_edges_fdr_bh.max(axis=1), color='b', alpha=0.2)
    
    
    plt.ylim(ymin=0, ymax=df_edges_holm.max().max()*1.2)
    plt.xlim(xmin=100, xmax=1000)
    plt.xlabel('Number of samples')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    #ax = plt.axes()
    #ax.set_facecolor("white")

#%% CREATE DATAFRAME FROM MODEL #####
def makerun(model, simmatP, multtest):
    totals=[]
    for N in np.arange(100,1000,10):
        df=bayes.sampling(model, n=N, verbose=0)
        out = hnet.main(df, alpha=0.05, multtest=multtest, drop_empty=False, verbose=0)
        totals.append((N, np.sum(np.sum(out['simmatP']<=0.05)), np.sum(np.sum((out['simmatP']<=0.05)==(simmatP<=0.05)))))
    
    return(totals)