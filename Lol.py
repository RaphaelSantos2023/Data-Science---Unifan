import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

def read_data():
    # Leitura das tabelas Excel
    facebook_df = pd.read_excel('Pasta.xlsx', sheet_name="Facebook")
    instagram_df = pd.read_excel('Pasta.xlsx', sheet_name="Instagram")

    # Adicionando uma coluna para identificar a origem dos dados
    facebook_df['canal_origem'] = 'Facebook'
    instagram_df['canal_origem'] = 'Instagram'

    # Combinando os DataFrames
    return pd.concat([facebook_df, instagram_df], ignore_index=True)

def preprocess_data(df):
    # Converter a coluna 'feedback' para numérico, substituindo valores não numéricos por NaN
    df['feedback'] = pd.to_numeric(df['feedback'], errors='coerce')
    df['data_nascimento'] = pd.to_datetime(df['data_nascimento'], errors='coerce')

    # Calcular a idade
    df['idade'] = df['data_nascimento'].apply(lambda x: calcular_idade(x) if pd.notnull(x) else np.nan)

    # Extraindo o mês da data de venda
    df['mes_venda'] = df['data_venda'].dt.to_period('M')

    return df

def calcular_idade(data_nascimento):
    hoje = datetime.today()
    return hoje.year - data_nascimento.year - ((hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day))

def analyze_sales(df):
    # Contando o número de vendas por mês
    vendas_por_mes = df.groupby('mes_venda')['Venda'].sum()
    mes_mais_vendas = vendas_por_mes.idxmax()
    maior_numero_vendas = vendas_por_mes.max()
    print(f"\nO mês com o maior número de vendas foi: {mes_mais_vendas} com {maior_numero_vendas} vendas.\n")

    # Cálculo da média das idades
    media_idade = int(df['idade'].mean())
    print(f"Média de idade dos clientes cadastrados: {media_idade} anos")

def analyze_age_distribution(df):
    # Contagem de clientes por faixa etária
    faixas_etarias = pd.cut(df['idade'], bins=[10, 20, 30, 40, 50, 60, 70, 80], right=False)
    contagem_faixas = df.groupby(faixas_etarias, observed=True)['idade'].count()
    print("\nContagem de clientes por faixa etária:")
    print(contagem_faixas)

def analyze_buyers(df):
    # Análise de maiores compradores
    compradores_total = df.groupby('idade')['Venda'].sum().sort_values(ascending=False)
    top_20_percent = int(len(compradores_total) * 0.2)
    top_compradores = compradores_total.index[:top_20_percent]
    media_idade_top_compradores = int(np.mean(top_compradores.to_numpy()))
    print(f"\nMédia de idade dos 20% que mais compraram: {media_idade_top_compradores} anos")

def analyze_gender(df):
    # Análise de gênero
    contagem_genero = df['genero'].value_counts()
    print("\nContagem de clientes por gênero:")
    print(contagem_genero)

    # Identificando o gênero que compõe a maioria dos compradores
    compradores = df[df['Venda'] > 0]  # Filtrar compradores
    contagem_genero_compradores = compradores['genero'].value_counts()
    genero_maioria_compradores = contagem_genero_compradores.idxmax()
    print(f"\nO gênero que compõe a maioria dos compradores é: {genero_maioria_compradores}")

def analyze_channels(df):
    # Análise de canais de comunicação que mais trazem resultados
    canais = df.groupby('canal_origem')['Venda'].sum()
    print("\nVendas por canal de origem:")
    print(canais)

    # Decisão de onde investir mais em marketing
    canal_mais_eficaz = canais.idxmax()
    print(f"O canal mais eficaz para investimento em marketing é: {canal_mais_eficaz}")

def analyze_feedbacks(df):
    # Análise de feedbacks
    feedbacks = df.groupby('canal_origem')['feedback'].mean()
    print("\nMédia de feedback por canal:")
    print(feedbacks)

def calculate_profit(df):
    # Cálculo do lucro com base no preço fixo
    preco_fixo = 12.90
    df['lucro'] = df['Venda'] * preco_fixo
    lucro_total = df['lucro'].sum()
    print(f"\nO lucro total obtido com as vendas foi: R$ {lucro_total:.2f}")

    # Lucro por canal de origem
    lucro_por_canal = df.groupby('canal_origem')['lucro'].sum()
    print("\nLucro por canal de origem:")
    print(lucro_por_canal)

def analyze_engagement(df):
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

def segment_customers(df):
    # Pré-processamento para segmentação de clientes
    features = df[['Venda', 'feedback']].dropna(subset=['Venda', 'feedback'])

    # Verificando o número de amostras válidas
    print(f"Número de amostras válidas para segmentação: {len(features)}")

    if len(features) >= 3:  # Verifica se há ao menos 3 amostras
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Segmentação de clientes usando KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['segmento'] = kmeans.fit_predict(features_scaled)

        # Análise de segmentos
        segmentos = df.groupby('segmento').agg({
            'feedback': 'mean',
            'Venda': 'mean',
            'canal_origem': 'count',
            'lucro': 'mean'
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

def prepare_target_variable(df):
    # Criar a variável alvo (se houve venda ou não)
    df['sucesso_campanha'] = np.where(df['Venda'] > 0, 1, 0)
    return df

def train_logistic_regression(df):
    # Verifique a distribuição da variável alvo
    print(df['sucesso_campanha'].value_counts())

    if df['sucesso_campanha'].value_counts().min() > 0:  # Se há pelo menos uma amostra de cada classe
        # Selecionar as features para prever o sucesso da campanha
        features = ['feedback', 'idade', 'Curtidas', 'Compartilhamentos'] + list(df.filter(like='canal_origem_').columns) + list(df.filter(like='genero_').columns)

        # Separar as variáveis independentes (X) e a variável alvo (y)
        X = df[features]
        y = df['sucesso_campanha']

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Padronizar os dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[['feedback', 'idade', 'Curtidas', 'Compartilhamentos']])
        X_test_scaled = scaler.transform(X_test[['feedback', 'idade', 'Curtidas', 'Compartilhamentos']])

        # Manter as colunas categóricas
        X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']), X_train.drop(columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']).reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']), X_test.drop(columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']).reset_index(drop=True)], axis=1)

        # Treinar o modelo de regressão logística
        modelo = LogisticRegression()
        try:
            modelo.fit(X_train_final, y_train)
            
            # Previsões
            y_pred = modelo.predict(X_test_final)

            # Avaliação do modelo
            acuracia = accuracy_score(y_test, y_pred)
            matriz_confusao = confusion_matrix(y_test, y_pred)
            relatorio_classificacao = classification_report(y_test, y_pred)

            print(f"\nAcurácia do modelo: {acuracia:.2f}")
            print("\nMatriz de Confusão:")
            print(matriz_confusao)
            print("\nRelatório de Classificação:")
            print(relatorio_classificacao)
        except ValueError as e:
            print(f"Erro ao treinar o modelo: {e}")
    else:
        print("A variável alvo não contém amostras suficientes de ambas as classes para treinar o modelo.")

def main():
    df = read_data()
    df = preprocess_data(df)
    analyze_sales(df)
    analyze_age_distribution(df)
    analyze_buyers(df)
    analyze_gender(df)
    analyze_channels(df)
    analyze_feedbacks(df)
    calculate_profit(df)
    analyze_engagement(df)
    segment_customers(df)
    df = prepare_target_variable(df)
    train_logistic_regression(df)

if __name__ == "__main__":
    main()
