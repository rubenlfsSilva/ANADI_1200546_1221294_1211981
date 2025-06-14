# ============================================================
# Bibliotecas necessárias para análise de dados e visualização
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # importar o módulo warnings

warnings.filterwarnings('ignore')  # ignorar todos os warnings

# ============================================================
# ALÍNEA 1: Carregar o ficheiro CSV, verificar dimensão e sumário
# ============================================================

# 1. Carregar o dataset com separador e decimal corretos
df = pd.read_csv("C:/Users/E6103/Desktop/Anadi/ANADI_1221294_1211981_1200546/TRAB2/AIRPOL_data.csv", sep=";", decimal=",")

# 2. Remover colunas que foram criadas automaticamente e estão vazias (ex: "Unnamed")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 3. Contar e remover linhas duplicadas para evitar análises repetidas ou enviesadas
duplicados = df.duplicated().sum()
print(f"\nNúmero de linhas duplicadas encontradas: {duplicados}")
if duplicados > 0:
    df = df.drop_duplicates()
    print("Linhas duplicadas removidas.")

# 4. Mostrar o número de linhas e colunas do dataset
print("\nDimensões do dataset:", df.shape)

# 5. Mostrar as primeiras 5 linhas para entender como os dados estão estruturados
print("\nPrimeiras 5 linhas dos dados:")
print(df.head())

# 6. Estatísticas descritivas para todas as colunas (numéricas e categóricas)
print("\nResumo estatístico (antes da conversão):")
print(df.describe(include='all'))

# 7. Verificar tipos de dados e quantidade de valores nulos por coluna
print("\nInformação dos dados:")
print(df.info())

print("\nValores nulos por coluna:")
print(df.isnull().sum())

# ============================================================
# ALÍNEA 2 — Gráficos para análise exploratória dos dados
# ============================================================

# Definir estilo visual para gráficos e tamanho padrão das figuras
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ----------------------------
# Variáveis numéricas
# ----------------------------

# Lista das colunas que possuem dados numéricos para análise
numerical_cols = [
    'Affected_Population',
    'Populated_Area[km2]',
    'Air_Pollution_Average[ug/m3]',
    'Value'  # Coluna ainda com nome original nesta fase
]

# Para cada coluna numérica, plotar histograma + KDE e boxplot
for col in numerical_cols:
    plt.figure(figsize=(14, 5))

    # Histograma + curva de densidade (KDE) para mostrar distribuição dos dados
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f'Histograma de {col}')

    # Boxplot para identificar outliers e ver resumo estatístico visualmente
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color="lightcoral")
    plt.title(f'Boxplot de {col}')

    # Ajustar layout para evitar sobreposição dos gráficos
    plt.tight_layout()
    plt.show()

# ----------------------------
# Variáveis categóricas
# ----------------------------

# Lista das colunas categóricas para análise de frequência
categorical_cols = ['Country', 'NUTS_Code', 'Air_Pollutant', 'Outcome']

# Para cada coluna categórica, plotar gráfico de barras (contagem)
for col in categorical_cols:
    n_uniques = df[col].nunique()  # Quantidade de categorias diferentes

    # Ajustar tamanho da figura conforme número de categorias (mais categorias = figura maior)
    plt.figure(figsize=(12, 6 if n_uniques <= 15 else 10))

    if n_uniques > 20:
        # Se muitas categorias, mostrar somente as 20 mais frequentes para melhor visualização
        top_values = df[col].value_counts().nlargest(20).index
        df_top = df[df[col].isin(top_values)]
        sns.countplot(data=df_top, y=col, order=top_values, palette="viridis")
        plt.title(f"Top 20 categorias em {col}")
    else:
        # Se poucas categorias, mostrar todas
        sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette="viridis")
        plt.title(f"Distribuição de {col}")

    # Legendas dos eixos
    plt.xlabel("Contagem")
    plt.ylabel(col)

    # Ajuste do layout para visualização correta
    plt.tight_layout()
    plt.show()

# ============================================================
# ALÍNEA 4 — Agrupamento dos países por região
# ============================================================

# Renomear coluna 'Value' para 'Premature_Deaths'
df.rename(columns={'Value': 'Premature_Deaths'}, inplace=True)

# Atualizar array de colunas numéricas para refletir o novo nome da coluna
numerical_cols = ['Premature_Deaths' if col == 'Value' else col for col in numerical_cols]

western = ['Austria', 'Belgium', 'France', 'Germany', 'Netherlands', 'Switzerland']
eastern = ['Poland', 'Czechia', 'Hungary']
southern = ['Greece', 'Spain', 'Italy', 'Portugal']
northern = ['Sweden', 'Denmark', 'Finland', 'Northern Europe']

# Função para mapear país à respectiva região
def map_region(country):
    if country in western:
        return 'Western Europe'
    elif country in eastern:
        return 'Eastern Europe'
    elif country in southern:
        return 'Southern Europe'
    elif country in northern:
        return 'Northern Europe'
    else:
        return 'Other'

# Aplicar o mapeamento para criar a coluna 'Region'
df['Region'] = df['Country'].apply(map_region)

# Mostrar a contagem de registros por região
# 'Número de Registos' significa a quantidade de linhas (observações) do dataset para cada região
print("\nDistribuição por Região:")
print(df['Region'].value_counts())

# Gráfico da distribuição por região (quantidade de observações por região)
sns.countplot(data=df, x='Region', order=df['Region'].value_counts().index, palette='muted')
plt.title("Distribuição de Registos por Região")
plt.xlabel("Região")
plt.ylabel("Número de Registos")  # Número de observações/linhas do dataset em cada região
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ============================================================
# 4.2
# ALÍNEA 3A — regressão linear múltipla
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Seleção das features e target
features = ['Affected_Population', 'Populated_Area[km2]', 'Air_Pollution_Average[ug/m3]']
X = df[df['Region'] == 'Southern Europe'][features]
y = df[df['Region'] == 'Southern Europe']['Premature_Deaths']

# Regressão Linear Múltipla com k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()

mae_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_test_fold)

    mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
    r2_scores.append(r2_score(y_test_fold, y_pred_fold))

# Resultados médios
print("Resultados da Regressão Linear Múltipla (k-fold cross-validation):")
print(f"MAE médio: {np.mean(mae_scores):.2f}")
print(f"RMSE médio: {np.mean(rmse_scores):.2f}")
print(f"R2 médio: {np.mean(r2_scores):.2f}")

# Treinar modelo final com todos os dados
model.fit(X, y)
print("\nModelo final treinado em todos os dados:")
print(f"Intercepto: {model.intercept_:.2f}")
print("Coeficientes:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Resultados da Regressão Linear Múltipla com validação cruzada (k-fold):

# O modelo foi avaliado usando k-fold cross-validation para garantir uma avaliação robusta.
# O MAE médio (Mean Absolute Error) foi aproximadamente 466.98, indicando que, em média,
# o erro absoluto nas previsões de mortes prematuras é cerca de 467 casos.

# O RMSE médio (Root Mean Squared Error) foi 1956.01, valor maior que o MAE, o que indica
# que existem alguns erros grandes que impactam a média quadrática do erro.

# O R² médio de 0.22 indica que aproximadamente 22% da variabilidade da variável 'Premature_Deaths'
# é explicada pelas variáveis independentes consideradas: 'Affected_Population',
# 'Populated_Area[km2]' e 'Air_Pollution_Average[ug/m3]'.

# Modelo final treinado em todos os dados:
# Intercepto: 10.51
# Coeficientes:
#   - Affected_Population: 0.0003 (pequeno impacto positivo na previsão)
#   - Populated_Area[km2]: 0.0319 (impacto positivo)
#   - Air_Pollution_Average[ug/m3]: -2.1823 (impacto negativo, pode indicar colinearidade ou outros fatores)

# Conclusão:
# O modelo de regressão linear múltipla apresenta desempenho moderado na previsão de mortes prematuras.
# O valor relativamente baixo do R² sugere que o modelo não explica grande parte da variação,
# indicando a necessidade de incluir variáveis adicionais ou usar métodos mais complexos.
# Assim, a regressão linear múltipla serve como uma linha de base, mas não deve ser o modelo final.

# ============================================================
# 4.2
# ALÍNEA 3C — SVM
# ============================================================
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

def kfold_indices(data, k=5):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        if i == k-1:
            test_idx = indices[i*fold_size:]
        else:
            test_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        folds.append((train_idx, test_idx))
    return folds

kernels = ['linear', 'rbf', 'poly']
k = 5
folds = kfold_indices(X, k)

resultados_svm = {}

for kernel in kernels:
    maes = []
    rmses = []
    for train_idx, test_idx in folds:
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = SVR(kernel=kernel)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        maes.append(mae)
        rmses.append(rmse)

    mae_medio = np.mean(maes)
    rmse_medio = np.mean(rmses)
    resultados_svm[kernel] = (mae_medio, rmse_medio)
    print(f"Kernel: {kernel} | MAE médio: {mae_medio:.2f} | RMSE médio: {rmse_medio:.2f}")

melhor_kernel = min(resultados_svm, key=lambda k: resultados_svm[k][0])  # menor MAE
print(f"\nMelhor kernel: {melhor_kernel} com MAE médio {resultados_svm[melhor_kernel][0]:.2f} e RMSE médio {resultados_svm[melhor_kernel][1]:.2f}")

# ============================================================
# 4.2
# ALÍNEA 3D — Rede Neuronial
# ============================================================

from sklearn.neural_network import MLPRegressor

# Exemplos de configurações para testar
configs = [
    (10,),        # 1 camada oculta, 10 neurônios
    (20,),        # 1 camada oculta, 20 neurônios
    (10, 10),     # 2 camadas ocultas, 10 neurônios cada
    (50,),        # 1 camada oculta, 50 neurônios
]

resultados_nn = {}
k = 5
folds = kfold_indices(X, k)

for config in configs:
    maes = []
    rmses = []
    for train_idx, test_idx in folds:
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = MLPRegressor(hidden_layer_sizes=config, max_iter=500, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        maes.append(mae)
        rmses.append(rmse)

    mae_medio = np.mean(maes)
    rmse_medio = np.mean(rmses)
    resultados_nn[config] = (mae_medio, rmse_medio)
    print(f"Config {config} | MAE médio: {mae_medio:.2f} | RMSE médio: {rmse_medio:.2f}")

melhor_config = min(resultados_nn, key=lambda k: resultados_nn[k][0])  # menor MAE
print(f"\nMelhor configuração: {melhor_config} com MAE médio {resultados_nn[melhor_config][0]:.2f} e RMSE médio {resultados_nn[melhor_config][1]:.2f}")
