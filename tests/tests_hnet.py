import numpy as np
import pandas as pd
import hnet

def test_import_example():
    import hnet
    hn = hnet.hnet()
    df = hn.import_example('sprinkler')
    assert df.shape==(1000, 4)
    df = hn.import_example('titanic')
    assert df.shape==(891, 12)
    df = hn.import_example('student')
    assert df.shape==(649, 33)

def test_association_learning():
    from hnet import hnet
    # TEST WHITE LIST
    white_list=['Survived','Embarked','Fare','Cabin']
    hn = hnet(white_list=white_list)
    df = hn.import_example('titanic')
    out = hn.association_learning(df)
    assert np.all(np.isin(np.unique(out['labx']), white_list))

    # TEST WHITE LIST WITH DTYPES
    dtypes=['num','cat','cat','cat']
    hn = hnet(white_list=['Survived','Embarked','Fare','Cabin'], dtypes=dtypes)
    df = hn.import_example('titanic')
    out = hn.association_learning(df)
    assert np.all(out['dtypes'][:,1]==dtypes)

    # TEST BLACK LIST
    black_list=['Survived','Embarked','Fare','Cabin','Age','Ticket','PassengerId','Name']
    hn = hnet(black_list=black_list)
    df = hn.import_example('titanic')
    out = hn.association_learning(df)
    assert np.all(np.setdiff1d(df.columns, black_list)==['Parch', 'Pclass', 'Sex', 'SibSp'])
    
    # TEST VARIOUS INPUT DTYPES
    hn1 = hnet()
    df = hn1.import_example('sprinkler')
    out1 = hn1.association_learning(df.astype(bool))
    hn2 = hnet(excl_background='0.0')
    out2 = hn2.association_learning(df)
    assert np.all(out1['simmatP']==out2['simmatP'])

    hn3 = hnet(excl_background='0.0')
    out3 = hn3.association_learning(df.astype(float).astype(str))
    assert np.all(out1['simmatP']==out3['simmatP'])

    hn = hnet(excl_background='False')
    out5 = hn.association_learning(df.astype(bool).astype(str))
    # out5['simmatP']
    
    hn = hnet(excl_background='0')
    out7 = hn.association_learning(df.astype(int).astype(str))
    assert np.all(out7['simmatP'].values==out5['simmatP'].values)
    
    # hn8 = hnet(excl_background='0')
    # out8 = hn3.association_learning(df.astype(int))
    # assert np.all(out5['simmatP'].values==out8['simmatP'].values)
    
    # TEST FOR BOOL VALUES
    from hnet import hnet
    hn = hnet()
    df = hn.import_example('sprinkler')
    hn1 = hnet(dtypes=np.array(['bool']*df.shape[1]))
    hn1.association_learning(df.astype(bool))
    hn2 = hnet()
    hn2.association_learning(df.astype(bool))
    assert np.all(hn1.results['simmatP'].values==hn2.results['simmatP'].values)


def test_hnet():
    import hnet
    hn = hnet.hnet()
    df = hn.import_example('sprinkler')

    # Should start empty
    assert hasattr(hn, 'results')==False

    # Should contain results
    hn.association_learning(df)
    assert hasattr(hn, 'results')==True
    
    # Should contain the following keys
    assert [*hn.results.keys()]==['simmatP', 'simmatLogP', 'labx', 'dtypes', 'counts', 'rules']
    
    # Should have specified size
    assert hn.results['simmatP'].shape==(7,7)
    assert hn.results['simmatLogP'].shape==(7,7)
        
    # TEST INPUT PARAMTERS
    hn = hnet.hnet(excl_background=['0.0'])
    hn.association_learning(df)
    assert hn.results['simmatP'].shape==(4,4)
    
    # TEST 2: Compute across all combinations : should be empty
    nfeat = 100
    nobservations = 50
    df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
    dtypes = np.array(['cat']*nobservations)
    dtypes[np.random.randint(0,2,nobservations)==1]='num'
    y = np.random.randint(0,2,nfeat)

    # Should be empty
    hn = hnet.hnet(dtypes=dtypes)
    hn.association_learning(df)
    assert hn.results['simmatP'].shape==(0,0)


def test_combined_rules():
    hn = hnet.hnet()
    df = hn.import_example('sprinkler')
    hn.association_learning(df)
    rules = hn.combined_rules()
    # Check output
    assert np.all(rules.values[0,0]==['Cloudy', 'Wet_Grass'])
    assert rules.values[1,1]==hn.results['rules'].values[1,1]
    # Check column names
    assert np.all(rules.columns.values==['antecedents_labx', 'antecedents', 'consequents', 'Pfisher'])


def test_enrichment():
    import hnet
    # Example with random categorical and numerical values
    nfeat = 100
    nobservations = 50
    df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
    dtypes = np.array(['cat']*nobservations)
    dtypes[np.random.randint(0,2,nobservations)==1]='num'
    y = np.random.randint(0,2,nfeat)

    # TEST 1: Compute enrichment: should be empty
    out = hnet.enrichment(df,y, dtypes=dtypes)
    assert out.shape==(0,0)

    # TEST 2: Example with 1 true positive column
    nfeat=100
    nobservations=50
    df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
    y = np.random.randint(0,2,nfeat)
    df['positive_one'] = y
    dtypes = np.array(['cat']*(nobservations+1))
    dtypes[np.random.randint(0,2,nobservations+1)==1]='num'
    dtypes[-1]='cat'
    # Run model
    out = hnet.enrichment(df,y, alpha=0.05, dtypes=dtypes)
    assert out.shape[0]==1
    assert out.shape[1]>=11
    
    # No pvalue should give all resutls back
    out = hnet.enrichment(df,y, alpha=1, dtypes=dtypes)
    assert out.shape==(51, 13)

    # TEST 3 : Example most simple manner with and without multiple test correction
    nfeat=100
    nobservations=50
    df = pd.DataFrame(np.random.randint(0,2,(nfeat,nobservations)))
    y = np.random.randint(0,2,nfeat)
    out = hnet.enrichment(df,y)
    # With multiple testing : should be empty
    assert out.shape==(0,0)
    out = hnet.enrichment(df,y, multtest=None)
    
    # TEST 4 : CHECK P_VALUE ORDER
    hn = hnet.hnet()
    df = hn.import_example('titanic')
    out = hnet.enrichment(df, y=df['Survived'].values)
    
    # Check output column names
    assert np.all(out.columns.values==['category_label', 'P', 'logP', 'overlap_X', 'popsize_M',
       'nr_succes_pop_n', 'samplesize_N', 'dtype', 'y', 'category_name',
       'Padj', 'zscore', 'nr_not_succes_pop_n'])
    
    # Check detected results
    assert np.all(out['category_name'].values==['Survived', 'Pclass', 'Sex', 'SibSp', 'Fare', 'Embarked'])

    # TEST FOR DTYPES INT VS BOOL
    import hnet
    df = hnet.import_example('sprinkler')
    out1 = hnet.enrichment(df.astype(int), y=df.iloc[:,0].values)
    out2 = hnet.enrichment(df.astype(bool), y=df.iloc[:,0].values)
    assert np.all(out1.values==out2.values)

    # TEST FOR BOOL WITH AND WITHOUT DTYPE INPUT PARMETER
    import hnet
    df = hnet.import_example('sprinkler')
    out1 = hnet.enrichment(df.astype(bool), y=df.iloc[:,0].values)
    out2 = hnet.enrichment(df.astype(bool), y=df.iloc[:,0].values, dtypes=np.array(['bool']*df.shape[1]))
    assert np.all(out1.values==out2.values)



# %%
# df = hnet.import_example('titanic')
# model = hnet.fit(df)
# model = hnet.fit(df, k=10)
# G = hnet.plot(model, dist_between_nodes=0.4, scale=2)
# G = hnet.d3graph(model, savepath='c://temp/titanic3/')

# # %%
# import hnet.hnet as hnet

# df    = hnet.import_example('sprinkler')
# out   = hnet.fit(df, alpha=0.05, multtest='holm', excl_background=['0.0'])

# G     = hnet.plot(out, dist_between_nodes=0.1, scale=2)
# G     = hnet.plot(out)
# G     = hnet.plot(out, savepath='c://temp/sprinkler/')

# A     = hnet.heatmap(out, savepath='c://temp/sprinkler/', cluster=False)
# A     = hnet.heatmap(out, savepath='c://temp/sprinkler/', cluster=True)
# A     = hnet.heatmap(out)

# A     = hnet.d3graph(out)
# A     = hnet.d3graph(out, savepath='c://temp/sprinkler/', directed=False)

# # %%
# df    = hnet.import_example('sprinkler')
# out   = hnet.fit(df)
# G     = hnet.plot(out, dist_between_nodes=0.1, scale=2)
# G     = hnet.plot(out)
# G     = hnet.plot(out, savepath='c://temp/sprinkler/')
# A     = hnet.heatmap(out, cluster=False)
# A     = hnet.heatmap(out, cluster=True)
# A     = hnet.d3graph(out, savepath='c://temp/sprinkler/')

# # %%
# df    = pd.read_csv('../../../DATA/OTHER/elections/USA_2016_election_primary_results.zip')
# out   = hnet.fit(df, alpha=0.05, multtest='holm', dtypes=['cat','','','','cat','cat','num','num'])
# G     = hnet.plot(out, dist_between_nodes=0.4, scale=2)
# A     = hnet.d3graph(out, savepath='c://temp/USA_2016_elections/')
