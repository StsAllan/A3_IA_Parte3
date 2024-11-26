import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregar o arquivo CSV
file_path = 'DadosTreino_Cardiopatas.csv'
data = pd.read_csv(file_path, delimiter=';')

# Converter a coluna '2nd_AtaqueCoracao' para valores binários (Sim = 1, Nao = 0)
data['2nd_AtaqueCoracao'] = data['2nd_AtaqueCoracao'].map({'Sim': 1, 'Nao': 0})

# Separar as variáveis independentes (features) e a variável dependente (target)
X = data.drop(columns=['2nd_AtaqueCoracao'])
y = data['2nd_AtaqueCoracao']

# Normalizar os dados para uso em redes neurais
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir as diferentes arquiteturas a serem simuladas
arquiteturas = [
    {'hidden_layer_sizes': (5,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (5,), 'activation': 'relu', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (5,), 'activation': 'logistic', 'solver': 'adam'},
    {'hidden_layer_sizes': (5,), 'activation': 'logistic', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (9,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (9,), 'activation': 'relu', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (9,), 'activation': 'logistic', 'solver': 'adam'},
    {'hidden_layer_sizes': (9,), 'activation': 'logistic', 'solver': 'lbfgs'}
]

# Inicializar listas para armazenar resultados
resultados = []

# Treinar e avaliar cada arquitetura
for arquitetura in arquiteturas:
    # Criar o modelo de rede neural
    mlp = MLPClassifier(hidden_layer_sizes=arquitetura['hidden_layer_sizes'],
                        activation=arquitetura['activation'],
                        solver=arquitetura['solver'],
                        max_iter=1000,
                        random_state=42)
    
    # Treinar o modelo
    mlp.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = mlp.predict(X_test)
    
    # Calcular acurácia e matriz de confusão
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    # Armazenar resultados
    resultados.append({
        'Arquitetura': arquitetura,
        'Acurácia': acuracia,
        'Matriz de Confusão': matriz_confusao
    })

# Exibir os resultados
for res in resultados:
    print(f"Arquitetura: {res['Arquitetura']}")
    print(f"Acurácia: {res['Acurácia']:.4f}")
    print(f"Matriz de Confusão:\n{res['Matriz de Confusão']}\n")
