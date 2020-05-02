# HNET - Graphical Hypergeometric Networks

[![Python](https://img.shields.io/pypi/pyversions/hnet)](https://img.shields.io/pypi/pyversions/hnet)
[![PyPI Version](https://img.shields.io/pypi/v/hnet)](https://pypi.org/project/hnet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdogant/hnet/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/hnet/month)](https://pepy.tech/project/hnet/month)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/hnet/)

This package detects associations in datasets across features with unknown function.

* **Background** Real-world data sets often contain measurements with both continues and categorical values for the same sample. Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps, and it remains a challenge to describe the relationships of one variable on another. The data understanding part is crucial but without making any assumptions on the model form, the search space is super-exponential in the number of variables and therefore not a common practice.
* **Result** We propose graphical hypergeometric networks (HNet), a method where associations across variables are tested for significance by statistical inference. The aim is to determine a network with significant associations that can shed light on the complex relationships across variables. HNet processes raw unstructured data sets and outputs a network that consists of (partially) directed or undirected edges between the nodes (i.e., variables). To evaluate the accuracy of HNet, we used well known data sets, and generated data sets with known ground truth by Bayesian sampling. In addition, the performance of HNet for the same data sets is compared to Bayesian structure learning.
* **Conclusions** We demonstrate that HNet showed high accuracy and performance in the detection of node links. In the case of the Alarm data set we can demonstrate an average MCC score 0.33 + 0.0002 (P<1x10-6), whereas Bayesian structure learning showed an average MCC score of 0.52 + 0.006 (P<1x10-11), and randomly assigning edges resulted in a MCC score of 0.004 + 0.0003 (P=0.49). Although Bayesian structure learning showed slightly better results, HNet overcomes some of the limitations of existing methods as it processes raw unstructured data sets, it allows analysis of mixed data types, it easily scales up in number of variables, and allows detailed examination of the detected associations.

## Method overview
<p align="center">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig1.png" width="900" />
</p>

## Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install hnet from PyPI (recommended). Hnet is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
It is distributed under the Apache 2.0 license.

```
pip install hnet
```

```python
from hnet import hnet
```

- Simple example for the sprinkler data set
```python
Initialize hnet with default settings
from hnet import hnet

# Load example dataset
df = hnet.import_example('sprinkler')

# Print to screen
print(df)
#      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
# 0              1         0       3  ...   7.2500   NaN         S
# 1              2         1       1  ...  71.2833   C85         C
# 2              3         1       3  ...   7.9250   NaN         S
# 3              4         1       1  ...  53.1000  C123         S
# 4              5         0       3  ...   8.0500   NaN         S
# ..           ...       ...     ...  ...      ...   ...       ...
# 886          887         0       2  ...  13.0000   NaN         S
# 887          888         1       1  ...  30.0000   B42         S
# 888          889         0       3  ...  23.4500   NaN         S
# 889          890         1       1  ...  30.0000  C148         C
# 890          891         0       3  ...   7.7500   NaN         Q
```


Structure learning on the titanic dataset
```python
hn = hnet()
out = hn.fit_transform(df, return_dict=True)

```
<p align="center">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig2.png" width="900" />
</p>


```python
# Plot static graph
G_static = hn.plot(node_color='cluster')

# Plot heatmap
P_heatmap = hn.heatmap(cluster=True)

# Plot dynamic graph
G_dynamic = hn.d3graph(node_color='cluster')
```

<p align="center">
  <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">
     <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig4.png" width="900" />
  </a>
</p>


* <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">d3graph example</a> 
<link rel="import" href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">

### Performance
<p align="center">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig3.png" width="900" />
</p>

### Citation
Please cite hnet in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019hnet,
  title={hnet},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/hnet}},
}
```

### Licence
See [LICENSE](LICENSE) for details.

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant/hnet)
