### What is manifold learning

an approach to non-linear dimensionality reduction

- Nonlinear dimensionality reduction
    
    > High-dimensional
     data, meaning data that requires more than two or three dimensions to represent, can be  difficult to interpret
    . One approach to simplification is to assume that the data of interest lies within lower-dimensional space. If the data of interest is of low enough dimension, the data can be visualised in the low-dimensional space.
    > 

High-dimensional datasets can be very difficult to visualize. While data in two or three dimensions can be plotted to show the inherent structure of the data, equivalent high-dimensional plots are much less intuitive. To aid visualization of the structure of a dataset, the dimension must be reduced in some way.

### Manifold hypothesis

- Natural data in high dimensional spaces concentrates close to lower dimensional manifolds.
- Probability density decreases very rapidly when moving away from the supporting manifold.

### Purpose

- **data compression**
    
    Lossy Image Compression with compressive Autoencoders
    ![data_compression](https://user-images.githubusercontent.com/93747285/156995502-8c22fa96-67bf-4354-ba4d-2f571dc46a72.png)
    
- **data visiualization**
    
    t-distributed stochastic neighbor embedding (t-SNE)
    ![tsne](https://user-images.githubusercontent.com/93747285/156995543-2aa1bb8b-6960-416e-9849-9c102615a7db.png)

- **to avoid curse of dimensionality**
 ![curseofdimensionality](https://user-images.githubusercontent.com/93747285/156995606-692dd367-8c4e-4ef9-a1fe-aff1cdb8ba0b.png)
        
- **to extract important features**
  ![featureextraction](https://user-images.githubusercontent.com/93747285/156995704-f7a5c892-0064-4716-9ae1-467068282139.png)


### Methods
- PCA(Principal Component Analysis)
    - the process of computing the principal components and using them to perform a change of basis on the data
- LLE(Locally Linear Embedding)
    - seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods.
    - three stages: search nearest neighbors → weight matrix construction → partial eigenvalue decomposition
- Hessian LLE
    - to address the regularization problem of LLE
    - a hessian-based quadratic form
- Modified LLE
    - to address the regularization problem of LLE
    - to use multiple weight vectors
- LTSA(Local Tangent Space Alignment)
    - to characterize the local geometry at each neighborhood via its tangent space
    - align local tangent sapces
    - three stages: search nearest neighbors → weight matrix construction → partial eigenvalue decomposition
- Isomap
    - an extension of MDS or Kernel PCA
    - seeks a lower-dimensional embedding which maintains geodesic distances between all points.
    - three stages: search nearest neighbors → search shortest-path graph → partial eigenvalue decomposition
- MDS(Multi-dimensional Scaling)
    - seeks a low-dimensional representation of the data in which the distances respect well the distances in the original high-dimensional space.
    - used to analyse similarity or dissimilarity data
- SE(Spectral Embedding)
    - a spectral decomposition of the graph Laplacian
    - three stages: weighted graph construction → graph Laplacian construction → partial eigenvalue decomposition
- t-SNE(t-distributed Stochastic Neighbor Embedding)
    - converts affinities of data points to probabilities.
    - original space: Gaussian joint probabilities
    - embedded space: Student’s t-distribution
- AE(Autoencoder)
    - Auto-associators, diabolo networks, sandglass-shaped net
    - Make output layer same size as input layer
    - Loss encourages output to be close to input
    - Unsupervised Learning → Supervised Learning
    - encoder: represent latent vector well within training datasets at least.
    - decoder: generate data well within training datasets at least.
- VAE(Variational Autoencoder)
    - Generative model
    - simple distribution like normal distribution
    - to use maximum likelihood estimation
    - to use variational inference to find simple distribution to generate target data

### Example: S-curve dataset
- dataset: S-curve
  ![download-1](https://user-images.githubusercontent.com/93747285/157000862-563d44bb-0939-4fe7-8328-03f014d9f8ec.png)

- Environments
    - neighbors: 10
    - dimensionality reduction: 3d → 2d
 ![manifold_1646643973](https://user-images.githubusercontent.com/93747285/157000893-0e8223cf-68d9-4024-acdb-72036b845ff7.png)

### Example: Mnist dataset
- dataset: mnist


- Environments
    - neighbors: 10
    - dimensionality reduction: 3d → 2d
