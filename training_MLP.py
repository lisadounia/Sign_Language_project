import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



# Charger les données
data_dict = pickle.load(open('./data.p', 'rb'))
data = np.asarray(data_dict['data'])  

#MPClassifier take only int for labels
alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]
for i in range(len(data_dict['labels'])): 
    for j in range(len(alphabet)):
        if data_dict['labels'][i] == alphabet[j]: 
            data_dict['labels'][i] = j
labels = np.asarray(data_dict['labels'])

data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)


# Initialisation 
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Deux couches cachées avec 128 et 64 neurones
    activation='relu',  # Fonction d’activation ReLU (couramment utilisée)
    solver='adam',  # Optimiseur Adam (très efficace pour l’apprentissage)
    max_iter=500,  # Nombre maximal d'itérations pour l'entraînement
    random_state=42,
    early_stopping=True  # Arrête l'entraînement si la validation ne s'améliore pas
)

#training model
model.fit(data_train, labels_train)

# Assesment
labels_train_pred = model.predict(data_train)
labels_test_pred = model.predict(data_test)

train_acc = accuracy_score(labels_train, labels_train_pred)
test_acc = accuracy_score(labels_test, labels_test_pred)

print("Training Accuracy:",train_acc)#0.9855075231039486
print("Test Accuracy:", test_acc) #0.9828152447872909

# Afficher le rapport de classification
print("\n📊 Classification Report (Test Data):")
print(classification_report(labels_test, labels_test_pred))

# Sauvegarder le modèle entraîné
with open('mlp_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

