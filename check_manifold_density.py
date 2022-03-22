import os, sys
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from functools import partial
from autoencoders import AE, VAE
import argparse
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

def parse_args():
  desc = 'estimate densities of manifold space by manifold methods'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--method', type=str, default='pca', help='manifold method to estimate density, default pca')
  parser.add_argument('--n_iter', type=int, default=50, help='number of iterate to estimate density, default 50')
  parser.add_argument('--n_samples', type=int, default=1000, help='sample size, default 1000')
  parser.add_argument('--n_components', type=int, default=2, help='the size of manifold space, default 2')
  parser.add_argument('--n_neighbors', type=int, default=10, help='the size of neighbors, default 10')
  parser.add_argument('--loss', type=str, default='s_curve', help='the name of dataset, default s_curve')
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='the size of learning step, default 1e-3')
  parser.add_argument('--optimizer', type=str, default='adam', help='the name of optimizer, adam')
  return check_args(parser.parse_args())
    
def check_args(args):
  try:
    args.method = args.method.lower()
    assert args.method in ['pca', 'lle', 'ltsa', 'hessian', 'modified', 'isomap', 'mds', 'se', 't-sne', 'ae', 'vae']
  except:
    args.method = 'pca'
  
  try:
    args.n_iter = int(args.n_iter) 
  except:
    args.n_iter = 50
  
  try:
    args.n_samples = int(args.n_samples) 
  except:
    args.n_samples = 1000
    
  try:
    args.n_components = int(args.n_components) 
  except:
    args.n_components = 2
  
  try:
    args.n_neighbors = int(n_neighbors)
  except:
    args.n_neighbors = 10
    
  try:
    args.loss = args.loss.lower()
    assert args.loss in ['mse', 'mae', 'binary_crossentropy', 'vae_loss']
  except:
    args.loss = 'mse'
  
  try:
    args.learning_rate = float(args.learning_rate)
  except:
    args.learning_rate = 1e-3
    
  try:
    args.optimizer = args.optimizer.lower()
    assert args.optimizer in ['adam', 'SGD', 'RMSProp', 'Nadam']
  except:
    args.optimizer = 'adam'
  
  return args
  

def _make_datasets(n_samples=1000):
  return datasets.make_s_curve(n_samples, random_state=0)

def _get_method(args):
  LLE = partial(manifold.LocallyLinearEmbedding,
                n_neighbors=args.n_neighbors, n_components=args.n_components,
                eigen_solver='auto')
  methods = {}
  methods['pca'] = PCA(args.n_components)
  methods['lle'] = LLE(method='standard')
  methods['ltsa'] = LLE(method='ltsa')
  methods['hessian'] = LLE(method='hessian')
  methods['modified'] = LLE(method='modified')
  methods['isomap'] = manifold.Isomap(n_components=args.n_components, n_neighbors=args.n_neighbors)
  methods['mds'] = manifold.MDS(args.n_components, max_iter=100, n_init=1)
  methods['se'] = manifold.SpectralEmbedding(n_components=args.n_components, n_neighbors=args.n_neighbors)
  methods['t-sne'] = manifold.TSNE(n_components=args.n_components, init='pca', random_state=0)
  methods['ae'] = AE(args.n_components, optimizer=args.optimizer, loss=args.loss, learning_rate=args.learning_rate)
  methods['vae'] = VAE(args.n_components, optimizer=args.optimizer, learnig_rate=args.learning_rate, beta=2)
  return methods[args.method]

def _estimate_density(encoded, kde):
  return kde.score_samples(np.transpose(encoded))  
  
def _iter_estimate(data, args):
  result = {}
  X, color = data
  
  method = manifold.TSNE(n_components=args.n_components, init='pca', random_state=0)
# if args.method == 'ae':
#   encoded = manifold.fit_transform(X, epochs=15, batch_size=1000, shuffle=True, verbose=0)
#   plt.scatter(encoded[:,0], encoded[:,1], c=color, cmap=plt.cm.Spectral)
#   plt.show()
#   plt.clf()
#   x = None
#   print('if contiune y or n:', end=' ')
#   while True:
#     x = input()
#     if x != 'y' and x != 'n':
#       print('if contiune y or n:', end=' ')
#     elif x == 'n': 
#       sys.exit()
#     elif x == 'y':
#       break
# elif args.method == 'vae':
#   encoded = manifold.fit_transform(X, epochs=3, batch_size=1000, shuffle=True, verbose=0)
#   plt.scatter(encoded[:,0], encoded[:,1], c=color, cmap=plt.cm.Spectral)
#   plt.show()
#   plt.clf()
#   x = None
#   print('if contiune y or n:', end=' ')
#   while True:
#     x = input()
#     if x != 'y' and x != 'n':
#       print('if contiune y or n:', end=' ')
#     elif x == 'n': 
#       sys.exit()
#     elif x == 'y':
#       break
# else:
  encoded = method.fit_transform(X)
  encoded = scaler.fit_transform(encoded)
  
  if args.n_iter % 5 == 0:
    rows = int(args.n_iter/5)
  else:
    rows = int(args.n_iter/5) + 1
  
  fig, axs = plt.subplots(nrows=rows, ncols=5, figsize=(15, 20))
  fig.suptitle(args.method.upper(), fontsize=24)
  
  kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
  kde.fit(np.transpose(encoded))
  init_density = _estimate_density(encoded, kde)
  print(init_density)
  
  for i in range(1, args.n_iter+1):
    print(i, end=' ')
    method = _get_method(args)
    if args.method == 'ae':
      encoded = method.fit_transform(X, epochs=15, batch_size=1000, shuffle=True, verbose=0)
    elif args.method == 'vae':
      encoded = method.fit_transform(X, epochs=3, batch_size=1000, shuffle=True, verbose=0)
    else:
      encoded = method.fit_transform(X)
    encoded = scaler.fit_transform(encoded)
    result[i] = _estimate_density(encoded, kde)
    axs[int((i-1)/5), (i-1) % 5].scatter(encoded[:,0], encoded[:,1], c=color, cmap=plt.cm.Spectral)
  
  print()
  fig.tight_layout(pad=3.0)
  plt.savefig('./density/manifolds/%s_%s.png' % (args.method.upper(), str(time.time())))
  plt.clf()
  return init_density, result

def plot_density(init_density, trials, method):
  plt.figure(figsize=(5,8))
  data = [sum(init_density - x) for x in trials.values()]
  print(data)
  plt.barh(range(1,len(data)+1), data)
  plt.title(method.upper(), fontsize=24)
  plt.savefig('./density/%s_barh_%s.png' % (method.upper(), str(time.time())))
  plt.clf()
  plt.figure(figsize=(15,7))
  ax = sns.distplot(data, bins=15, hist_kws={"rwidth":0.7, 'alpha':1.0})
  mids = [round(rect.get_x() + rect.get_width()/2,1) for rect in ax.patches]
  ax.set_xticks(mids)
  plt.title(method.upper(), fontsize=24)
  plt.savefig('./density/%s_hist_%s.png' % (method.upper(), str(time.time())))
  plt.clf()

def main(args):
  print('make a s_curve datasets')
  X, color = _make_datasets(n_samples=args.n_samples)
  print(X.shape)
  
  # estimate
  print('estimate density of %s' % args.method)
  init_density, trials = _iter_estimate((X, color), args)
  
  print('plot difference from original one')
  plot_density(init_density, trials, args.method)

if __name__ == '__main__':
    args = parse_args()
    if args is None:
      sys.exit()
    
    print(args)
    main(args)


      