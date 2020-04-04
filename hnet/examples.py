import hnet
print(hnet.__version__)
print(dir(hnet))


# %%
df = hnet.import_example('student')

df = hnet.import_example('titanic')

df = hnet.import_example('sprinkler')


# %%

model = hnet.fit(df)

G = hnet.plot(model)

G = hnet.heatmap(model, cluster=True)

G = hnet.d3graph(model)

[scores, adjmat] = hnet.compare_networks(model['simmatLogP'], model['simmatLogP'])

rules = hnet.combined_rules(model)

adjmatSymmetric = hnet.to_symmetric(model)

# %% Enrichment

df = hnet.import_example('titanic')
y = df['Survived'].values
out = hnet.enrichment(df, y)
