from functools import partial
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from autoencoders import AE, VAE
import sys

n_components = 2
n_neighbors = 10
n_points = 1000

def plot_s_curve():
  print('make s-curve dataset: (1000,3)')
  X, color = datasets.make_s_curve(n_points, random_state=0)
  
  # setup manifold methods
  methods = dict()
  LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors=n_neighbors, n_components=n_components,
              eigen_solver='auto')
  
  methods['PCA'] = PCA(n_components=n_components)
  methods['LLE'] = LLE(method='standard')
  methods['LTSA'] = LLE(method='ltsa')
  methods['Hessian LLE'] = LLE(method='hessian')
  methods['Modified LLE'] = LLE(method='modified')
  methods['Isomap'] = manifold.Isomap(n_neighbors=n_neighbors,
                                      n_components=n_components)
  methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
  methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                             n_neighbors=n_neighbors)
  methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                   random_state=0)
  methods['AE'] = AE(n_components, X.shape[-1])
  methods['VAE'] = VAE(n_components, X.shape[-1])
  
  fig = plt.figure(figsize=(30, 10))
  fig.suptitle("Manifold Learning with %i points, %i neighbors"
               % (n_points, n_neighbors), fontsize=36)

  # plot X(s-curve)
  ax = fig.add_subplot(271, projection='3d')
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
  ax.view_init(4, -72)
  for i, (label, method) in enumerate(methods.items()):
    t0 = time.time()
    if label == 'AE' or label == 'VAE':
      Y = method.fit_transform(X, shuffle=True, epochs=15, batch_size=100, verbose=0)
    else:
      Y = method.fit_transform(X)
    print("%s: %.2g sec" % (label, time.time() - t0))
    ax = fig.add_subplot(2, 7, 2 + i + (i > 4))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("%s (%.2g sec)" % (label, time.time() - t0))
  fig.tight_layout(pad=3.0)
  plt.savefig('./plots/manifold_' + str(round(time.time())) + '.png')

if __name__ == '__main__':
  if len(sys.args) != 2:
    print('usage: python3 %s <s_curve or mnist>' % (sys.args[0]))
    sys.exit()
  
  print('start plotting manifold of %s' % (sys.args[1]))
  if sys.args[1] == 'mnist':
    plot_mnist()
  else:
    plot_s_curve()
