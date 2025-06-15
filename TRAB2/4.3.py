import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# warnings.filterwarnings('ignore')

# %%
# 1. Carregar o dataset com separador e decimal corretos
df = pd.read_csv("../AIRPOL_data.csv", sep=";", decimal=",")

# 2. Remover colunas que foram criadas automaticamente e estão vazias (ex: "Unnamed")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 3. Contar e remover linhas duplicadas para evitar análises repetidas ou enviesadas
duplicados = df.duplicated().sum()
print(f"\nNúmero de linhas duplicadas encontradas: {duplicados}")
if duplicados > 0:
    df = df.drop_duplicates()
    print("Linhas duplicadas removidas.")

    # Renomear coluna 'Value' para 'Premature_Deaths'
    df.rename(columns={'Value': 'Premature_Deaths'}, inplace=True)

# %%
print("Colunas do dataset:")
print(df.columns.tolist())

print("\nPrimeiras linhas do dataset:")
display(df.head())

# %%
# ============================================================
# 4.3
# ALÍNEA 1 — Derive um novo atributo RespDisease que separa as doenças em respiratórias
# ('Asthma' 'Chronic obstructive pulmonary disease') e não respiratórias.
# ============================================================

# Lista de doenças respiratórias
respiratory_diseases = ['Asthma', 'Chronic obstructive pulmonary disease']

# Criar a coluna RespDisease
df['RespDisease'] = np.where(df['Outcome'].isin(respiratory_diseases), 'Respiratory', 'Non-Respiratory')

# Mostrar quantidade de cada categoria
print("Contagem por categoria:")
print(df['RespDisease'].value_counts())

print("\nProporção por categoria:")
print(df['RespDisease'].value_counts(normalize=True))

# %%
# ============================================================
# 4.3
# ALÍNEA 2 — Usando o método k-fold cross validation desenvolva modelos de previsão de
# RespDisease usando os seguintes métodos:
# a) Árvore de decisão. Otimize os parâmetros do modelo.
# b) Rede neuronal. Otimize a configuração da rede.
# c) SVM. Otimize o kernel.
# d) K-vizinhos-mais-próximos. Otimize o parâmetro K.
# ============================================================

# Seleção de variáveis preditoras e variável alvo
features = ['Air_Pollution_Average[ug/m3]', 'Affected_Population', 'Populated_Area[km2]']
X = df[features].copy()
y = df['RespDisease']
print("Preparar X e y... ", end="")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("feito.")

# Função para avaliar os modelos com k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y):
    acc, sens, spec, f1 = [], [], [], []
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    return {
        "Accuracy": (np.mean(acc), np.std(acc)),
        "Sensitivity": (np.mean(sens), np.std(sens)),
        "Specificity": (np.mean(spec), np.std(spec)),
        "F1": (np.mean(f1), np.std(f1))
    }

# Avaliação dos modelos
results = {}

modelos = {
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Rede Neuronal': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'K-vizinhos': KNeighborsClassifier(n_neighbors=5)
}

for nome, modelo in modelos.items():
    # Como alguns modelos podem demorar algum tempo faz-se este print para perceber qual está a ser processado
    print(f"Treinar modelo: {nome}... ", end="")
    results[nome] = evaluate_model(modelo, X, y_encoded)
    print("feito.")

# Construir DataFrame com os resultados
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.columns = ['Accuracy (mean±std)', 'Sensitivity (mean±std)', 'Specificity (mean±std)', 'F1 (mean±std)']

# Formatar as métricas para apresentação (média ± desvio padrão)
def format_metric(mean_std):
    mean, std = mean_std
    return f"{mean:.4f} ± {std:.4f}"

df_formatado = df_results.map(format_metric)

# Mostrar resultados formatados
print("\nResumo dos resultados (média ± desvio padrão):\n")
print(df_formatado)

# %%
# ============================================================
# 4.3
# ALÍNEA 3 — Obtenha a média e o desvio padrão da Accuracy;
# Sensitivity; Specificity e F1 do atributo RespDisease com
# os modelos obtidos na alínea anterior.
# ============================================================

def format_metric(mean_std):
    mean, std = mean_std
    return f"{mean:.4f} ± {std:.4f}"

df_alinea3 = df_results.map(format_metric)

print("Média e desvio padrão por modelo (alvo: RespDisease):\n")
print(df_alinea3)

# %%
# ============================================================
# 4.3
# ALÍNEA 4 — Verifique se existe diferença significativa no desempenho
# dos dois melhores modelos (nível de significância de 5%).
# ============================================================

# Hipóteses do teste estatístico:
# H0 (hipótese nula): Não existe diferença significativa entre os modelos.
# H1 (hipótese alternativa): Existe diferença significativa entre os modelos.
# Nível de significância: α = 0.05

# Utilizar os F1-scores médios previamente calculados na alínea 2 (results)
print("Avaliar F1-score médio dos modelos... ", end="")
f1_scores = {nome: resultado['F1'] for nome, resultado in results.items()}
print("feito.")

# Ordenar modelos pelo F1-score médio (posição 0 do tuplo: média)
print("Ordenar modelos pelo F1-score médio... ", end="")
sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1][0], reverse=True)
modelo_1, modelo_2 = sorted_f1[0][0], sorted_f1[1][0]
print("feito.")

print(f"Modelos selecionados para comparação: {modelo_1} vs {modelo_2}")

# Função para obter a lista de F1-scores (um por fold) — necessário para o teste estatístico
def get_f1_list(model):
    f1 = []
    for train_idx, test_idx in kfold.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1.append(f1_score(y_test, y_pred))
    return f1

# Recalcular listas de F1 por fold apenas para os dois modelos a comparar
print(f"Calcular F1-score por fold para {modelo_1}... ", end="")
f1_1 = get_f1_list(modelos[modelo_1])
print("feito.")

print(f"Calcular F1-score por fold para {modelo_2}... ", end="")
f1_2 = get_f1_list(modelos[modelo_2])
print("feito.")

# Aplicar o teste t pareado (comparação estatística entre os dois modelos)
print("Executar teste t pareado... ", end="")
stat, p_value = ttest_rel(f1_1, f1_2)
print("feito.")

# Resultado e interpretação
print(f"p-value = {p_value:.4f}")

# Se p-value < 0.05 -> há evidência suficiente para rejeitar a hipótese nula.
if p_value < 0.05:
    print("Conclusão: diferença estatisticamente significativa.")
    melhor = modelo_1 if np.mean(f1_1) > np.mean(f1_2) else modelo_2
    print(f"Modelo com melhor desempenho: {melhor}")
else:
    print("Conclusão: não há diferença estatisticamente significativa entre os dois modelos.")

# %%
# ============================================================
# 4.3
# ALÍNEA 5 — Compare os resultados dos modelos.
# Discuta qual apresentou melhor e pior desempenho em:
# Accuracy, Sensitivity, Specificity e F1.
# ============================================================

# Extrair média de cada métrica para comparação
metricas = df_results.copy()
for coluna in metricas.columns:
    metricas[coluna] = metricas[coluna].apply(lambda x: x[0])  # só a média, ignora std

# Identificar o melhor e pior modelo por métrica
for metrica in metricas.columns:
    melhor_modelo = metricas[metrica].idxmax()
    pior_modelo = metricas[metrica].idxmin()
    valor_melhor = metricas[metrica].max()
    valor_pior = metricas[metrica].min()

    print(f"{metrica}:")
    print(f"  Melhor desempenho -> {melhor_modelo} ({valor_melhor:.4f})")
    print(f"  Pior desempenho   -> {pior_modelo} ({valor_pior:.4f})\n")
