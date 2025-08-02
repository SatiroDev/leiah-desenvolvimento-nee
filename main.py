from src.dataset import load_data
from src.model import train_model
from src.predict import predict

X, y, target_names, vector = load_data()

model = train_model(X, y)


sentence = ["Quais críticas sociais são feitas à desigualdade econômica na narrativa?"]

# chamada assim pois a função predict retorna um número que usado como indice
index = predict(vector, model, sentence)

print(target_names[index])