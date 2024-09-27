import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Importar os dados do medical_examination.csv e atribuí-los à variável df.
df = pd.read_csv('medical_examination.csv')

# 2 Criar a coluna de sobrepeso na variável df
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)  # Calculando o IMC
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)  # Definindo sobrepeso

# 3 Normalizar os dados
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)  # Define 0 para colesterol bom e 1 para colesterol ruim
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)  # Define 0 para glicose boa e 1 para glicose ruim

# 4 Desenhar o Gráfico Categórico na função draw_cat_plot()
def draw_cat_plot():
    # 5 Criar um DataFrame para o gráfico categórico usando pd.melt() com os valores de cholesterol, gluc, smoke, alco, active, e overweight em df_cat.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])  # Transformando os dados para formato longo

    # 6 Agrupar e reformular os dados em df_cat para separar por cardio e mostrar as contagens de cada recurso.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')  # Contando as ocorrências de cada combinação

    # 7 Criar um gráfico que mostre as contagens dos recursos categóricos usando sns.catplot()
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig  # Alterado de 'count' para 'bar'

    # 8 Obter a figura para a saída e armazená-la na variável fig
    fig.suptitle('Count of categorical features by Cardio')  # Adicionando um título ao gráfico
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustando o layout para que o título não se sobreponha ao gráfico

    # 9
    fig.savefig('catplot.png')  # Salvando o gráfico como um arquivo
    return fig  # Retorna o objeto fig para testes

# 10
def draw_heat_map():
    # 11 Limpar os dados em df_heat
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Garantindo que a pressão diastólica seja menor que a sistólica
        (df['height'] >= df['height'].quantile(0.025)) &  # Filtrando pelos percentis 2.5 e 97.5 para altura
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Filtrando pelos percentis 2.5 e 97.5 para peso
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 Calcular a matriz de correlação e armazená-la na variável corr
    corr = df_heat.corr()

    # 13 Gerar uma máscara para o triângulo superior e armazená-la na variável mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))  # Você pode ajustar o tamanho conforme necessário

    # 15 Plotar a matriz de correlação usando sns.heatmap()
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, cmap='coolwarm', cbar_kws={"shrink": .5}, ax=ax)

    # 16
    fig.savefig('heatmap.png')  # Salvando o gráfico como um arquivo
    return fig  # Retorna o objeto fig para testes
