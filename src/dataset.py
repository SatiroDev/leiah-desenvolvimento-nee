from neelearn.datasets import load_nee_assessments
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    data = load_nee_assessments()

    vector = TfidfVectorizer()

    X = vector.fit_transform(data.data)

    y = data.target

    return X, y, data.target_names, vector

    

