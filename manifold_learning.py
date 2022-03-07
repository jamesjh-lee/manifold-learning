from functools import partial
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from autoencoders import AE, VAE

n_components = 2
n_neighbors = 10
n_points = 1000

def plot_manifold(X, methods, color):
  fig = plt.figure(figsize=(20, 8))
  fig.suptitle("Manifold Learning with %i points, %i neighbors"
               % (n_points, n_neighbors), fontsize=14)

  # plot X(s-curve)
  ax = fig.add_subplot(251, projection='3d')
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
  ax.view_init(4, -72)
  for i, (label, method) in enumerate(methods.items()):
    t0 = time.time()
    Y = method.fit_transform(X)
    print("%s: %.2g sec" % (label, time.time() - t0))
    ax = fig.add_subplot(2, 7, 2 + i + (i > 4))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    ax.axis('tight')
  plt.savefig('./plots/manifold_' + str(round(time.time())) + '.png')

if __name__ == '__main__':
  print('make s-curve dataset: (1000,3)')
  X, color = datasets.make_s_curve(n_points, random_state=0)
  
  # setup manifold methods
  methods = dict()
  
