import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variances = None

    def fit(self, X):
        n_features = X.shape[1]
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot be greater than "
                f"the number of features ({n_features})"
            )
        
        self.mean = np.mean(X, axis=0)  # nehseb el mean for each feature
        X_centered = X - self.mean  

        cov = np.cov(X_centered.T) 

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # eigenvalues: (D,)
                # eigenvectors: (D, D) 
                # kol col how principal component

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

       
        # 5. Select components
        self.components = eigenvectors[:, :self.n_components]  # shape (D, n_components)
        
        # 6. Explained variance
        total_variance = np.trace(cov)  
#   kan momken nesta3mel sum el eigenvalues badal sum el diagonal beta3 el coavariance matrix 
        self.explained_variances = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        # X_centered: (N, D)
        # self.components: (D, n_components)  
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # X: (N, K)
          #self.components.T: (K, D)
        return np.dot(X, self.components.T) + self.mean