# %% Importando os pacotes necessários
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler

# %% Importando o banco de dados
dataset = pd.read_csv("dataset/airline_passenger_satisfaction.csv", sep=",")
dataset.head()

# %% Informações gerais sobre o DataFrame
dataset.shape

# %% Analyzing descriptive statistics

dataset.describe()

# %% Checking if that there are no non-null values in the data 

dataset.info()

dataset.isnull().sum()

dataset.isna().sum()

# %% Removendo os valores nulos do dataset
dataset.dropna(inplace=True)

# %% Removendo colunas categóricas do dataset
# Como o objetivo desse projeto é analisar o método PCA, deve-se ter apenas colunas numéricas.

# Mantém apenas colunas numéricas (int ou float)
dataset_pca = dataset.select_dtypes(include=['int64', 'float64'])
dataset_pca.head()


# Deve-se retirar a coluna ID também
dataset_pca.drop(columns=["ID"], inplace=True)
dataset_pca.head()
dataset_pca.dtypes

# %% Analisar correlação entre variáveis
corr = dataset_pca.corr()

# Plotando 

import matplotlib.pyplot as plt
import seaborn as sns

# Cria a figura
plt.figure(figsize=(18,12), dpi=600)  # Tamanho 12x8 polegadas e resolução 600 dpi

# Cria o heatmap
sns.heatmap(
    corr,                  # Matriz de correlação
    cmap='Blues',          # Mapa de cores (tons de azul)
    vmax=1,                # Valor máximo da escala de cores
    vmin=-1,               # Valor mínimo da escala de cores
    center=0,              # Ponto central da escala
    square=True,           # Células quadradas
    linewidths=.5,         # Largura das linhas que separam as células
    annot=True,            # Escreve os valores dentro das células
    fmt='.2f',             # Formato dos números (2 casas decimais)
    annot_kws={'size':8}, # Tamanho da fonte dos números dentro das células
    cbar_kws={"shrink":0.50} # Reduz o tamanho da barra de cores
)

# Adiciona título e ajusta layout
plt.title('Matriz de Correlações', fontsize=14)  # Título do gráfico
plt.tight_layout()                               # Ajusta layout para não cortar nada
plt.tick_params(labelsize=10)                    # Tamanho das labels dos eixos

# Mostra o gráfico
plt.show()


# %% Teste de esfericidade de Bartlett
# %% Teste de Esfericidade de Bartlett
# Hipótese nula (H0): A matriz de correlação é igual à matriz identidade
#                    → as variáveis NÃO estão correlacionadas
# Hipótese alternativa (H1): A matriz de correlação NÃO é a identidade
#                            → as variáveis estão correlacionadas

bartlett, p_value = calculate_bartlett_sphericity(dataset_pca)

print(f'Qui² Bartlett: {round(bartlett,2)}')
print(f'p-valor: {round(p_value,4)}')

# %% Conclusão baseada no p-valor
if p_value < 0.05:
    print("Conclusão: p-valor < 0.05 → rejeita H0 → as variáveis estão correlacionadas. Faz sentido aplicar PCA.")
else:
    print("Conclusão: p-valor ≥ 0.05 → não rejeita H0 → as variáveis NÃO estão correlacionadas. PCA pode não ser apropriado.")

# %% Método PCA - Padronizando as variáveis
# Criar o scaler
scaler = StandardScaler()

# Ajustar e transformar os dados numéricos
dataset_pca_scaled = scaler.fit_transform(dataset_pca)

# Converte para DataFrame novamente, mantendo os nomes das colunas
dataset_pca_scaled = pd.DataFrame(dataset_pca_scaled, columns=dataset_pca.columns)

# Agora dá pra usar head(), describe(), etc.
dataset_pca_scaled.head()

# %%#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)

# Obtendo o número de variáveias para inciar o procedimento 
n_colunas = dataset_pca_scaled.shape[1]

fa = FactorAnalyzer(n_factors=n_colunas, method='principal', rotation=None).fit(dataset_pca_scaled)

#%% Obtendo os eigenvalues (autovalores): resultantes da função FactorAnalyzer

autovalores = fa.get_eigenvalues()[0]

print(autovalores) 

# Soma dos autovalores - é igual ao nº de variáveis

round(autovalores.sum(), 2) 

#%% Autovalores, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(18,12), dpi=600)
ax = sns.barplot(x=tabela_eigen.index, y='Variância', hue=tabela_eigen.index, palette='rocket', data=tabela_eigen)
for container in ax.containers:
    labels = [f"{v*100:.2f}%" for v in container.datavalues]
    ax.bar_label(container, labels=labels)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} fatores que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=8)
plt.ylabel("Variância explicada", fontsize=8)
plt.show()

#%% Determinando as cargas fatoriais

# Note que não há alterações nas cargas fatoriais nos 2 fatores!

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = dataset_pca_scaled.columns

print(tabela_cargas)

#%% Critério de Kaiser - selecionar automaticamente fatores com autovalor > 1

# Obter os autovalores
autovalores = fa.get_eigenvalues()[0]

# Selecionar apenas os fatores com autovalor > 1
fatores_kaiser = np.sum(autovalores > 1)
print(f"Número de fatores a manter pelo Critério de Kaiser: {fatores_kaiser}")

# %% Parametrizando o PCA novamente
fa_kaiser = FactorAnalyzer(n_factors=fatores_kaiser, method='principal', rotation=None).fit(dataset_pca_scaled)

#%% Autovalores, variâncias e variâncias acumuladas de 2 fatores

# Note que não há alterações nos valores, apenas ocorre a seleção dos fatores

autovalores_fatores = fa_kaiser.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

# Note que não há alterações nas cargas fatoriais nos 2 fatores!

cargas_fatoriais = fa_kaiser.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = dataset_pca_scaled.columns

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa_kaiser.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = dataset_pca_scaled.columns

print(tabela_comunalidades)

# Variáveis com comunalidade mais alta são bem representadas pelos fatores extraídos, ou seja, os fatores retêm mais informação dessas variáveis.
#%% Identificando os scores fatoriais

# Não há mudanças nos scores fatoriais!

#%% Scores fatoriais com fatores selecionados pelo Critério de Kaiser

# Usar .transform() para obter os scores das observações
scores_kaiser = fa_kaiser.transform(dataset_pca_scaled)

# Converter para DataFrame
tabela_scores = pd.DataFrame(
    scores_kaiser,
    columns=[f"Fator {i+1}" for i in range(scores_kaiser.shape[1])],
    index=dataset_pca_scaled.index  # mantém índices das observações
)

# Visualizar os primeiros scores
print(tabela_scores.head())


# %% Conclusão 

# Após a aplicação do PCA, as 18 variáveis originais foram reduzidas para 5 fatores, que explicam 64,55% da variância total, mantendo boa parte da informação do dataset.

# %% Fim