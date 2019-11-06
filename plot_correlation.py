import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import pylab
import seaborn as sns


__author__ = "Yifeng Tao"


def plot_correlation(data, num_max_char=60, figsize=(12,12), labelsize=10):
  """ Plot the heatmap of feature correlations.
    Calculate correlation coefficients of features. Conduct hierarchical
    clustering on features. Visualize in proper composition.

  Parameters
  ----------
  data: pandas.DataFrame
    each row a sample, each column a feature
  num_max_char: int
    maximum number of characters to be shown for each feature
  figsize, labelsize: figure parameters

  """

  X = np.corrcoef(data.values.T)
  feats = [feat[0:num_max_char]+"..." if len(feat) > num_max_char else feat for feat in data.columns]

  fig = plt.figure(figsize=figsize)


  # y-axis: hierarchical clustering
  ax = fig.add_axes([0.01,0.1,0.09,0.59])

  Y = linkage(X, method="ward")
  Z1 = dendrogram(Y, orientation="left", no_plot=True) #do not show dendrogram
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis("off")


  # x-axis: hierarchical clustering and plot dendrogram
  ax = fig.add_axes([0.1,0.705,0.6,0.05])

  Y = linkage(X.T, method="ward")
  #Z2 = dendrogram(Y, link_color_func=lambda k:"gray") #uncomment if you want gray dendrogram
  Z2 = dendrogram(Y)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis("off")


  # heatmap: feature correlation
  axmatrix = fig.add_axes([0.1,0.1,0.6,0.6])

  X = X[Z1["leaves"],:]
  X = X[:,Z2["leaves"]]
  axmatrix.matshow(X, aspect="auto", origin="lower", cmap=plt.cm.get_cmap("bwr"))

  feats = [feats[idx] for idx in Z2["leaves"]]
  plt.yticks([i+0.0 for i in range(len(feats))], feats, rotation=0)

  axmatrix.yaxis.set_label_position("right")
  axmatrix.xaxis.set_ticks_position("bottom")
  axmatrix.set_xticks([])
  plt.tick_params(labelsize=labelsize)


  # colorbar
  ax = fig.add_axes([0.70,0.1,0.1,0.6])

  gradient = [0.01*i for i in range(100,-100,-1)]
  gradient = np.vstack((gradient, gradient))
  gradient = np.transpose(gradient)
  ax.imshow(gradient, extent=[0,0.06,-1,1], cmap=plt.cm.get_cmap("bwr"))

  ax.get_xaxis().set_ticks([])
  ax.yaxis.tick_right()
  plt.tick_params(labelsize=labelsize)

  plt.show()
  #fig.savefig("figures/figure_tmp.pdf", bbox_inches="tight")



if __name__ == "__main__":

  data = pd.DataFrame(
      np.random.rand(100,10),
      columns=["feature-"+str(i) for i in range(10)],
      index=["sample-"+str(i) for i in range(100)])

  plot_correlation(data)

