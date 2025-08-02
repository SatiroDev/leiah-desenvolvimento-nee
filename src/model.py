import scipy
from sklearn.linear_model import LogisticRegression

def train_model (X_train: scipy.sparse._csr.csr_matrix, y_train: list):
    model = LogisticRegression(C=5)
    model.fit(X_train, y_train)
    return model