import numpy as np

class GMM_Scratch:
    def __init__(self, n_components, max_iter=100, tol=1e-4, covariance_type='full'):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        
        self.means = None      
        self.covariances = None
        self.weights = None    
        self.log_likelihood_history = []
        self.reg_covar = 1e-6

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        #calculate initial means by randomly selecting data points
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        #calculate πₖ initial equal weights for each component as 1/K
        self.weights = np.full(self.n_components, 1 / self.n_components)
        
        if self.covariance_type == 'full':
            #each covariance matrix is initialized to identity matrix (n_features, n_features)
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
            
        elif self.covariance_type == 'tied':
            #all components share the same covariance matrix initialized to identity (n_features, n_features)
            self.covariances = np.eye(n_features)
            
        elif self.covariance_type == 'diagonal':
            #each covariance matrix is initialized to ones vector it only stores variances for each feature not covariances between features
            self.covariances = np.ones((self.n_components, n_features))#(n_components, n_features)
            
        elif self.covariance_type == 'spherical':
            #each covariance is represented by a single variance value initialized to one (n_components,)
            self.covariances = np.ones(self.n_components)



    def _estimate_log_prob(self, X):
  
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
         
         # Calculate mean for each component(k)
        for k in range(self.n_components):
            mu = self.means[k]
            
            if self.covariance_type == 'full': 
                # Full covariance matrix
                cov = self.covariances[k] + np.eye(n_features) * self.reg_covar
                try:
                    inv_cov = np.linalg.inv(cov)
                    det_cov = np.linalg.det(cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.inv(cov + np.eye(n_features) * 1e-4)
                    det_cov = np.linalg.det(cov + np.eye(n_features) * 1e-4)
                
                diff = X - mu
                # formula = (x - mu).T @ inv_cov @ (x - mu) for each x
                mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                # constant term = -0.5 * (n_features * log(2π) + log|cov|)
                const = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_cov))
                log_prob[:, k] = const - 0.5 * mahalanobis #subtract mahalanobis distance from constant term
                
                #All Gaussians share the same covariance matrix
            elif self.covariance_type == 'tied':
                cov = self.covariances + np.eye(n_features) * self.reg_covar
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                diff = X - mu
                mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                const = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_cov))
                log_prob[:, k] = const - 0.5 * mahalanobis
             #here only variances are stored no covariances between features
            elif self.covariance_type == 'diagonal':
                var = self.covariances[k] + self.reg_covar
                #calculate determinant of diagonal covariance matrix as product of variances
                det_cov = np.prod(var)
                diff = X - mu
                mahalanobis = np.sum((diff ** 2) / var, axis=1)
                const = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_cov))
                log_prob[:, k] = const - 0.5 * mahalanobis
                
     
            elif self.covariance_type == 'spherical':
                var = self.covariances[k] + self.reg_covar
                det_cov = var ** n_features
                diff = X - mu
                mahalanobis = np.sum(diff ** 2, axis=1) / var
                const = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_cov))
                log_prob[:, k] = const - 0.5 * mahalanobis

        return log_prob

    def _e_step(self, X):
        # log(πk​N(xi​∣μk​,Σk​))
        weighted_log_prob = self._estimate_log_prob(X) + np.log(self.weights + 1e-10)
        #log(∑​π*​N(xi​∣μk​,Σk​))
        log_prob_norm = np.max(weighted_log_prob, axis=1, keepdims=True) + \
                        np.log(np.sum(np.exp(weighted_log_prob - np.max(weighted_log_prob, axis=1, keepdims=True)), axis=1, keepdims=True))
        #logγik​=log(numerator)−log(denominator)
        log_resp = weighted_log_prob - log_prob_norm
        return np.exp(log_resp), log_prob_norm

    def _m_step(self, X, responsibilities):
     
        n_samples, n_features = X.shape
        
        weights_sum = np.sum(responsibilities, axis=0) 
        
        #haw big this Gaussian is
        self.weights = weights_sum / n_samples
        
        #mean=responsibilities.T⋅X/∑​γik​
        self.means = np.dot(responsibilities.T, X) / weights_sum[:, np.newaxis]
        
        if self.covariance_type == 'full':
            
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = responsibilities[:, k, np.newaxis] * diff
                cov = np.dot(weighted_diff.T, diff) / weights_sum[k]
                self.covariances[k] = cov
                
        elif self.covariance_type == 'tied':
            avg_cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = responsibilities[:, k, np.newaxis] * diff
                avg_cov += np.dot(weighted_diff.T, diff)
            self.covariances = avg_cov / n_samples
            
        elif self.covariance_type == 'diagonal':
            for k in range(self.n_components):
                diff = X - self.means[k]
                avg_sq_diff = np.sum(responsibilities[:, k, np.newaxis] * (diff ** 2), axis=0)
                self.covariances[k] = avg_sq_diff / weights_sum[k]
                
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means[k]
                avg_sq_dist = np.sum(responsibilities[:, k] * np.sum(diff ** 2, axis=1))
                self.covariances[k] = avg_sq_dist / (weights_sum[k] * n_features)

    def fit(self, X):
        self._initialize_parameters(X)
        self.log_likelihood_history = []
        
        for i in range(self.max_iter):
            responsibilities, log_prob_norm = self._e_step(X)
            
            current_log_likelihood = np.sum(log_prob_norm)
            self.log_likelihood_history.append(current_log_likelihood)
            
            self._m_step(X, responsibilities)
            
            if i > 0 and abs(current_log_likelihood - self.log_likelihood_history[-2]) < self.tol:
                break
                
        return self

    def predict(self, X):
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        responsibilities, _ = self._e_step(X)
        return responsibilities

