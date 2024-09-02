import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Leitura das tabelas Excel (supondo que você tenha duas planilhas: 'facebook.xlsx' e 'instagram.xlsx')
facebook_df = pd.read_excel('Pasta.xlsx', sheet_name="Facbook")
instagram_df = pd.read_excel('Pasta.xlsx', sheet_name="Instagram")

# Adicionando uma coluna para identificar a origem dos dados
facebook_df['canal_origem'] = 'Facebook'
instagram_df['canal_origem'] = 'Instagram'

# Combinando os DataFrames
df = pd.concat([facebook_df, instagram_df], ignore_index=True)

# Converter a coluna 'feedback' para numérico, substituindo valores não numéricos por NaN
df['feedback'] = pd.to_numeric(df['feedback'], errors='coerce')

# Análise de canais de comunicação que mais trazem resultados
canais = df.groupby('canal_origem')['Venda'].sum()
print("Vendas por canal de origem:")
print(canais)

# Decisão de onde investir mais em marketing
canal_mais_eficaz = canais.idxmax()
print(f"O canal mais eficaz para investimento em marketing é: {canal_mais_eficaz}")

# Análise de feedbacks
feedbacks = df.groupby('canal_origem')['feedback'].mean()
print("\nMédia de feedback por canal:")
print(feedbacks)

# Pré-processamento para segmentação de clientes
features = df[['Venda', 'feedback']].dropna()  # Remover linhas com NaN
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Segmentação de clientes usando KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['segmento'] = kmeans.fit_predict(features_scaled)

# Análise de segmentos - Filtrando apenas colunas numéricas
segmentos = df.groupby('segmento').agg({
    'feedback': 'mean',
    'Venda': 'mean',
    'canal_origem': 'count'  # Para contar quantos clientes estão em cada segmento
})

# Renomeando a coluna de contagem de clientes
segmentos.rename(columns={'canal_origem': 'N_Clientes'}, inplace=True)

print("\nAnálise dos segmentos:")
print(segmentos)

# Decisão sobre quais segmentos priorizar
segmento_priorizado = segmentos['Venda'].idxmax()
print(f"\nO segmento que deve ser priorizado é: Segmento {segmento_priorizado}")





