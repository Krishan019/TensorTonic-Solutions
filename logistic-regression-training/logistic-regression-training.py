import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b = 0.0

    for i in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        error_vector = p - y

        l_w = (1/n_samples)*(np.dot((np.transpose(X)), error_vector))
        l_b = np.mean(error_vector)
        
        w = w - lr * l_w
        b = b - lr * l_b
    
    return w, b