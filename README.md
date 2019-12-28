# HNET - Hypergeometric networks
[![PyPI Version](https://img.shields.io/pypi/v/hnet)](https://pypi.org/project/hnet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdoganta/hnet/blob/master/LICENSE)

This package computes significance of the response variable with catagorical/numerical variables.

Detection of associations for datasets with mixed datatypes with unknown function.
These datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations. 
I solved this problem to minimize the amount of configurations required while still gaining many benefits of having schemas available.

The response variable (y) should be a vector with the same number of samples as for the input data. For each column in the dataframe significance is assessed for the labels in a two-class approach (y=1 vs y!=1).
Significane is assessed one tailed; only the fit for y=1 with an overrepresentation. Hypergeometric test is used for catagorical values. Wilcoxen rank-sum test for numerical values.

<p align="center">
  <img src="docs/figs/figure_1.png" width="900" />
</p>

## Contents
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
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
* Alternatively, install hnet from the GitHub source:

```bash
git clone https://github.com/erdoganta/hnet.git
cd hnet
python setup.py install
```  

## Quick Start
- Import hnet method

```python
from hnet import hnet
```

- Simple example with constant window, minimum percentage and threshold
```python
df=pd.read_csv('https://github.com/erdoganta/hnet/blob/master/hnet/data/fb.csv')['close']
out = cs.fit(df)
fig = cs.makefig(out)
```
The input is a pandas dataframe or series and looks like this:
<p align="left">
  <img src="docs/figs/input_example.png" width="110" />
</p>
The output looks as below:
<p align="center">
  <img src="docs/figs/figure_fb.png" width="900" />
</p>


```python
df=pd.read_csv('https://github.com/erdoganta/hnet/blob/master/hnet/data/btc.csv')['Close']
out = cs.fit(df)
fig = cs.makefig(out)
```
The output looks as below:
<p align="center">
  <img src="docs/figs/figure_btc.png" width="900" />
</p>


- Gridsearch walks over the windows and over the percentages to determine optimal window, minimum percentage and the threshold.
```python
df = pd.read_csv('https://github.com/erdoganta/hnet/blob/master/hnet/data/btc.csv')['close']
out = cs.gridsearch(df)
```

The output looks as below:
<p align="center">
  <img src="docs/figs/figure_gridsearch_btc.png" width="450" />
</p>


## Citation
Please cite hnet in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdoganta2019hnet,
  title={hnet},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdoganta/hnet}},
}
```

## Maintainers
* Erdogan Taskesen, github: [erdoganta](https://github.com/erdoganta)

## © Copyright
See [LICENSE](LICENSE) for details.
