import sklearn

def predict(vetor: sklearn.feature_extraction.text.TfidfVectorizer, model: sklearn.linear_model._logistic.LogisticRegression, frase: list):
    sentence_vector = vetor.transform(frase)
    index = model.predict(sentence_vector)[0]
    return index
