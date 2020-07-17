## Zoo of algorithms for dimensionality reduction
While most of well-studied and robust algorithms for Dimensionality Reduction (DR) like
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
are implemented in Python, there are many new methods that are not.
This repository aggregate several algorithms which have been adapted and may be applied for
DR in the Nearest Neighbour Search (NNS) problem. Also main ideas or/and architecture hacks can be gained.


### Collection
##### VAE
Extremely popular generative model with huge number of modifications - [original paper](https://arxiv.org/abs/1312.6114).

Possible [implementation](https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated)

##### ACAI 
One of the variations of VAE. [Paper](https://arxiv.org/pdf/1807.07543.pdf)
and [initial code](https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0)

##### t-SNE 
Well-studied method from [Paper](https://www.semanticscholar.org/paper/Visualizing-Data-using-t-SNE-Maaten-Hinton/1c46943103bd7b7a2c7be86859995a4144d1938b),
very [useful](https://towardsdatascience.com/t-sne-python-example-1ded9953f26) for data visualisation.

Here I trying to adapt this algorithm for huge datasets (near 1M and more) via neural network learning.

##### UMAP
Another graph-based method from [Paper](https://arxiv.org/abs/1802.03426)


##### Triplet 
Simple and old method that can show state-of-the-art results in plenty of tasks.  
[Paper](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf),
and possible [implementation](https://github.com/facebookresearch/spreadingvectors)


#### What about running?
To running some test it you need to specify paths to the corresponding data location in `data.py` file and run 
`python train.py --database sift --method acai` for learning ACAI methon on [SIFT](http://corpus-texmex.irisa.fr/) dataset.

Most of hyper-parameters are fixed, but you can easily change them.

See [repo](https://github.com/Shekhale/gbnns_dim_red) for applying any methods for NNS task.


### Questions and suggestions

Please feel free to share any ideas and contact me.