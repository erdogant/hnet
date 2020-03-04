import hnet
print(hnet.__version__)


# %%
df = hnet.import_example()

model = hnet.fit(df)

G = hnet.plot(model)

G = hnet.heatmap(model)

G = hnet.d3graph(model)

[scores, adjmat] = hnet.compare_networks(model['simmatLogP'], model['simmatLogP'])

rules = hnet.combined_rules(model)

adjmatSymmetric = hnet.to_symmetric(model)
