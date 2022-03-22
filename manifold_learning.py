from functools import partial
from itertools import zip_longest
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from autoencoders import AE, VAE, ConvolutionalVAE
import sys
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import offsetbox

n_components = 2
n_neighbors = 10
n_points = 1000
es = EarlyStopping(monitor='binary_crossentropy', patience=5)
scaler = MinMaxScaler(feature_range=(0,1))

def scheduler(epoch, lr):
    if epoch > 30 and epoch % 10 == 0:
        return 0.8 * lr
    return lr

sched = LearningRateScheduler(scheduler)

def plot_s_curve():
    print('make s-curve dataset: (1000,3)')
    X, color = datasets.make_s_curve(n_points, random_state=0)
    X_train = scaler.fit_transform(X)
  
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
    methods['Isomap'] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    methods['AE'] = AE(n_components, optimizer='adam', learning_rate=1e-4, loss='mse')
    methods['VAE'] = VAE(n_components, optimizer='adam', learning_rate=1e-4, beta=.2)
  
    fig = plt.figure(figsize=(10, 30))
    fig.suptitle("Manifold Learning with %i points, %i neighbors"
               % (n_points, n_neighbors), fontsize=24)

    # plot X(s-curve)
    ax = fig.add_subplot(621, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    for i, (label, method) in enumerate(methods.items()):
        t0 = time.time()
        if label == 'AE':
            Y = method.fit_transform(X, shuffle=True, epochs=15, batch_size=1000, verbose=0)
        elif label == 'VAE':
            Y = method.fit_transform(X, shuffle=True, epochs=3, batch_size=1000, verbose=0)
        else:
            Y = method.fit_transform(X)
        print("%s: %.2g sec" % (label, time.time() - t0))
        ax = fig.add_subplot(6, 2, 2 + i)
        ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        ax.set_title("%s (%.2g sec)" % (label, time.time() - t0))
    fig.tight_layout(pad=3.0)
    plt.savefig('./plots/manifold_s_curve' + str(round(time.time())) + '.png')
#    plt.show()

def plot_mnist():
    print('make digits dataset: (1797,64)')
    digits = datasets.load_digits(n_class=10)
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape
    n_neighbors = 10
#    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
#   for idx, ax in enumerate(axs.ravel()):
#       ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
#       ax.axis("off")
#   _ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
#   plt.show()
    def plot_embedding(X, title, ax):
        X = MinMaxScaler().fit_transform(X)
        for digit in digits.target_names:
            ax.scatter(
              *X[y == digit].T,
              marker=f"${digit}$",
              s=60,
              color=plt.cm.Dark2(digit),
              alpha=0.425,
              zorder=2,
            )
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(X.shape[0]):
            # plot every digit on the embedding
            # show an annotation box for a group of digits
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
              # don't show points that are too close
              continue
            shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
            imagebox = offsetbox.AnnotationBbox(
              offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
            )
            imagebox.set(zorder=1)
            ax.add_artist(imagebox)
          
        ax.set_title(title)
        ax.axis("off")
        
    # setup manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors=n_neighbors, n_components=n_components,
                  eigen_solver='dense')
    embeddings = {}
    embeddings['PCA'] = PCA(n_components=n_components)
    embeddings['LLE'] = LLE(method='standard')
    embeddings['LTSA'] = LLE(method='ltsa')
    embeddings['Hessian LLE'] = LLE(method='hessian')
    embeddings['Modified LLE'] = LLE(method='modified')
    embeddings['Isomap'] = manifold.Isomap(n_neighbors=n_neighbors,
                                       n_components=n_components)
    embeddings['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    embeddings['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors)
    embeddings['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                     random_state=0)
    embeddings['AE'] = AE(n_components, optimizer='adam', learning_rate=1e-4, loss='mae')
    embeddings['VAE'] = VAE(n_components, optimizer='adam', learning_rate=1e-6, beta=0.5)
    embeddings['CVAE'] = ConvolutionalVAE(n_components, optimizer='adam', learning_rate=1e-6, beta=0.5)
    
    # projections
    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        data = X
        start_time = time.time()
        if name == 'AE':
            projections[name] = transformer.fit_transform(data, epochs=30, batch_size=100, shuffle=True, callbacks=[es, sched], verbose=0)
        elif name == 'VAE' or name == 'CVAE':
            projections[name] = transformer.fit_transform(data, epochs=70, batch_size=100, shuffle=True, callbacks=[es, sched], verbose=0)
        else:
            projections[name] = transformer.fit_transform(data)
        timing[name] = time.time() - start_time
        print("%s: %.2g sec" % (name, timing[name]))
      
    # plot
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
    for name, ax in zip_longest(timing, axs.ravel()):
        if name is None:
          ax.axis("off")
          continue
        title = f"{name} (time {timing[name]:.3f}s)"
        plot_embedding(projections[name], title, ax)
    fig.tight_layout(pad=3.0)
    plt.savefig('./plots/manifold_mnist_' + str(round(time.time())) + '.png')
#    plt.show()
  
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 %s <s_curve or mnist>' % (sys.argv[0]))
        sys.exit()
  
    print('start plotting manifold of %s' % (sys.argv[1]))
    if sys.argv[1] == 'mnist':
        plot_mnist()
    else:
        plot_s_curve()
        
    