import numpy as np

class KMeans_Scratch:
    def __init__(self, n_clusters=2, max_iter=300,tolerance=1e-4, init_method='kmeans++'):
     
        self.n_clusters = n_clusters 
        self.max_iter = max_iter
        self.tolerance =tolerance 
        self.init_method = init_method
        
        # Attributes to store results
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None       # Final inertia
        self.inertia_history = []  # List to track inertia over iterations

    def _calculate_distances(self, X, centroids):
       
        #Ana 2st5dmt Vectorization 3ashan a7seb el distances
        # X_shape: (n_samples, 1, n_features)
        # Centroids shape: (1, n_clusters, n_features)
        # 3mlt kda 3ashan a5od el difference between each feature of point and each feature of centroid
        # Result shape: (n_samples, n_clusters)
        dist = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)) ## (n_samples, n_clusters)
        return dist #each row contain distances from a point to all centroids, each column contain one centroid
        #axis=2 3ashan a5od el sum 3la kol feature


    def _init_centroids(self, X):
    
        n_samples, n_features = X.shape
        
        if self.init_method == 'random':
            # choose random K points from the dataset
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices]
            
        elif self.init_method == 'kmeans++':
            #25tar first centroid random uniformly
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                # calculate distances from points to nearest centroid
                dist_matrix = self._calculate_distances(X, np.array(centroids))
                
                # Get the minimum distance squared for each point
                min_dist_sq = np.min(dist_matrix, axis=1) ** 2
                
                # 25tar next centroid based on weighted probability
                probs = min_dist_sq / np.sum(min_dist_sq)
                
                #Cumulative_probability for weighted selection
                Cumulative_probability = np.cumsum(probs)
                r = np.random.rand()
                
                
                for i, p in enumerate(Cumulative_probability):
                    if r < p:
                        centroids.append(X[i])
                        break
            
            return np.array(centroids)

    def fit(self, X):
        
        
        ##square distance between each point and its closest centroid inertua=1/n ∑​∥xi​−centroid_closest(i)​∥2
        self.inertia_history = []
        
        self.centroids = self._init_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            distances = self._calculate_distances(X, self.centroids)
            # Assign labels based on closest centroid
            self.labels_ = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros(self.centroids.shape)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = old_centroids[k] 
            
            self.centroids = new_centroids
            
            min_distances = np.min(distances, axis=1)
            current_inertia = np.sum(min_distances ** 2)
            self.inertia_history.append(current_inertia)
            self.inertia_ = current_inertia
            
            centroid_shift = np.sum(np.sqrt(np.sum((self.centroids - old_centroids)**2, axis=1)))
            
            if centroid_shift < self.tolerance:
                break
                
        return self

    def predict(self, X):
  
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

