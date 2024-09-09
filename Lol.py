import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Leitura das tabelas Excel
facebook_df = pd.read_excel('Pasta.xlsx', sheet_name="Facebook")
instagram_df = pd.read_excel('Pasta.xlsx', sheet_name="Instagram")

# Adicionando uma coluna para identificar a origem dos dados
facebook_df['canal_origem'] = 'Facebook'
instagram_df['canal_origem'] = 'Instagram'

# Combinando os DataFrames
df = pd.concat([facebook_df, instagram_df], ignore_index=True)

# Converter a coluna 'feedback' para numérico, substituindo valores não numéricos por NaN
df['feedback'] = pd.to_numeric(df['feedback'], errors='coerce')

# Cálculo da idade baseado na data de nascimento
df['data_nascimento'] = pd.to_datetime(df['data_nascimento'], errors='coerce')

# Função para calcular a idade
def calcular_idade(data_nascimento):
    hoje = datetime.today()
    return hoje.year - data_nascimento.year - ((hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day))

# Aplicar a função de idade na coluna 'data_nascimento'
df['idade'] = df['data_nascimento'].apply(lambda x: calcular_idade(x) if pd.notnull(x) else np.nan)

# Extraindo o mês da data de venda
df['mes_venda'] = df['data_venda'].dt.to_period('M')

# Contando o número de vendas por mês
vendas_por_mes = df.groupby('mes_venda')['Venda'].sum()

# Identificando o mês com o maior número de vendas
mes_mais_vendas = vendas_por_mes.idxmax()
maior_numero_vendas = vendas_por_mes.max()

print(f"\nO mês com o maior número de vendas foi: {mes_mais_vendas} com {maior_numero_vendas} vendas.\n")

# Cálculo da média das idades
media_idade = int(df['idade'].mean())  # Arredondar para inteiro
print(f"Média de idade dos clientes cadastrados: {media_idade} anos")

# Contagem de clientes por faixa etária
faixas_etarias = pd.cut(df['idade'], bins=[10, 20, 30, 40, 50, 60, 70, 80], right=False)
contagem_faixas = df.groupby(faixas_etarias, observed=True)['idade'].count()

print("\nContagem de clientes por faixa etária:")
print(contagem_faixas)

# Análise de maiores compradores
compradores_total = df.groupby('idade')['Venda'].sum().sort_values(ascending=False)

# Média de idade dos 20% que mais compraram
top_20_percent = int(len(compradores_total) * 0.2)
top_compradores = compradores_total.index[:top_20_percent]
media_idade_top_compradores = int(np.mean(top_compradores.to_numpy()))  # Arredondar para inteiro

print(f"\nMédia de idade dos 20% que mais compraram: {media_idade_top_compradores} anos")

# Análise de gênero
contagem_genero = df['genero'].value_counts()
print("\nContagem de clientes por gênero:")
print(contagem_genero)

# Identificando o gênero que compõe a maioria dos compradores
compradores = df[df['Venda'] > 0]  # Filtrar compradores
contagem_genero_compradores = compradores['genero'].value_counts()
genero_maioria_compradores = contagem_genero_compradores.idxmax()
print(f"\nO gênero que compõe a maioria dos compradores é: {genero_maioria_compradores}")

# Análise de canais de comunicação que mais trazem resultados
canais = df.groupby('canal_origem')['Venda'].sum()
print("\nVendas por canal de origem:")
print(canais)

# Decisão de onde investir mais em marketing
canal_mais_eficaz = canais.idxmax()
print(f"O canal mais eficaz para investimento em marketing é: {canal_mais_eficaz}")

# Análise de feedbacks
feedbacks = df.groupby('canal_origem')['feedback'].mean()
print("\nMédia de feedback por canal:")
print(feedbacks)

# Cálculo do lucro com base no preço fixo
preco_fixo = 12.90

# Criando uma nova coluna 'lucro', que é o número de vendas vezes o preço fixo
df['lucro'] = df['Venda'] * preco_fixo

# Cálculo do lucro total
lucro_total = df['lucro'].sum()
print(f"\nO lucro total obtido com as vendas foi: R$ {lucro_total:.2f}")

# Lucro por canal de origem
lucro_por_canal = df.groupby('canal_origem')['lucro'].sum()
print("\nLucro por canal de origem:")
print(lucro_por_canal)

# Análise de curtidas e compartilhamentos
curtidas_por_canal = df.groupby('canal_origem')['Curtidas'].mean()
compartilhamentos_por_canal = df.groupby('canal_origem')['Compartilhamentos'].mean()

# Exibindo a média de curtidas e compartilhamentos por canal
print("\nMédia de curtidas por canal:")
print(curtidas_por_canal)

print("\nMédia de compartilhamentos por canal:")
print(compartilhamentos_por_canal)

# Comparando as redes sociais com base nas médias
melhor_rede_curtidas = curtidas_por_canal.idxmax()
melhor_rede_compartilhamentos = compartilhamentos_por_canal.idxmax()

print(f"\nA rede com a melhor média de curtidas é: {melhor_rede_curtidas}")
print(f"A rede com a melhor média de compartilhamentos é: {melhor_rede_compartilhamentos}")

# Pré-processamento para segmentação de clientes
# Removendo apenas os valores ausentes de 'Venda' e 'feedback'
features = df[['Venda', 'feedback']].dropna(subset=['Venda', 'feedback'])

# Verificando o número de amostras válidas
print(f"Número de amostras válidas para segmentação: {len(features)}")

# Verifique se há amostras suficientes para o KMeans
if len(features) >= 3:  # Verifica se há ao menos 3 amostras
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Segmentação de clientes usando KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['segmento'] = kmeans.fit_predict(features_scaled)

    # Análise de segmentos - Filtrando apenas colunas numéricas
    segmentos = df.groupby('segmento').agg({
        'feedback': 'mean',
        'Venda': 'mean',
        'canal_origem': 'count',  # Para contar quantos clientes estão em cada segmento
        'lucro': 'mean'  # Média de lucro por segmento
    })

    # Renomeando a coluna de contagem de clientes
    segmentos.rename(columns={'canal_origem': 'N_Clientes'}, inplace=True)

    print("\nAnálise dos segmentos:")
    print(segmentos)

    # Decisão sobre quais segmentos priorizar
    segmento_priorizado = segmentos['Venda'].idxmax()
    print(f"\nO segmento que deve ser priorizado é: Segmento {segmento_priorizado}")
else:
    print("Amostras insuficientes para segmentação.")


