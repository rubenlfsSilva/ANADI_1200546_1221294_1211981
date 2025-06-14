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

    # Renomear coluna 'Value' para 'Premature_Deaths'
    df.rename(columns={'Value': 'Premature_Deaths'}, inplace=True)

# ============================================================
# 4.3
# ALÍNEA 1 — Derive um novo atributo RespDisease que separa as doenças em respiratórias
# ('Asthma' 'Chronic obstructive pulmonary disease') e não respiratórias.
# ============================================================
import numpy as np

print(df.columns)

# Lista de doenças respiratórias
respiratory_diseases = ['Asthma', 'Chronic obstructive pulmonary disease']

# Criar a coluna RespDisease
df['RespDisease'] = np.where(df['Outcome'].isin(respiratory_diseases), 'Respiratory', 'Non-Respiratory')

# Mostrar quantidade de cada categoria
print("Contagem por categoria:")
print(df['RespDisease'].value_counts())

print("\nProporção por categoria:")
print(df['RespDisease'].value_counts(normalize=True))

# ============================================================
# 4.3
# ALÍNEA 2.a — Árvore de decisão. Otimize os parâmetros do modelo.
# ============================================================