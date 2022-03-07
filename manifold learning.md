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
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96adcd10-f59e-4fc2-ac22-781999bccb0c/Untitled.png)
    
- **data visiualization**
    
    t-distributed stochastic neighbor embedding (t-SNE)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bc2fae29-0208-4737-b517-76a5d1ae2349/Untitled.png)
    
- **to avoid curse of dimensionality**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d01affff-e8e0-4460-bc75-6b3534224dd7/Untitled.png)
    
- **to extract important features**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3aaf4986-c0dc-439d-80b9-85bfba19e861/Untitled.png)
    

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
    
    ![download-1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2eea4adf-c8aa-4be4-9b7b-e33c14a49c2a/download-1.png)
    
- Environments
    - neighbors: 10
    - dimensionality reduction: 3d → 2d

![download.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/876a0457-ea68-41b0-8597-e82005bb45a6/download.png)
