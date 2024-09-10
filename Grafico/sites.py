import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def read_data():
    facebook_df = pd.read_excel('Pasta.xlsx', sheet_name="Facebook")
    instagram_df = pd.read_excel('Pasta.xlsx', sheet_name="Instagram")
    facebook_df['canal_origem'] = 'Facebook'
    instagram_df['canal_origem'] = 'Instagram'
    return pd.concat([facebook_df, instagram_df], ignore_index=True)

def preprocess_data(df):
    df['feedback'] = pd.to_numeric(df['feedback'], errors='coerce')
    df['data_nascimento'] = pd.to_datetime(df['data_nascimento'], errors='coerce')
    df['idade'] = df['data_nascimento'].apply(lambda x: calcular_idade(x) if pd.notnull(x) else np.nan)
    df['mes_venda'] = df['data_venda'].dt.to_period('M')
    df['feedback'].fillna(df['feedback'].median(), inplace=True)
    df['idade'].fillna(df['idade'].median(), inplace=True)
    return df

def calcular_idade(data_nascimento):
    hoje = datetime.today()
    return hoje.year - data_nascimento.year - ((hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day))

def analyze_sales(df):
    vendas_por_mes = df.groupby('mes_venda')['Venda'].sum()
    mes_mais_vendas = vendas_por_mes.idxmax()
    maior_numero_vendas = vendas_por_mes.max()
    print(f"\nO mês com o maior número de vendas foi: {mes_mais_vendas} com {maior_numero_vendas} vendas.\n")
    media_idade = int(df['idade'].mean())
    print(f"Média de idade dos clientes cadastrados: {media_idade} anos")

def analyze_age_distribution(df):
    faixas_etarias = pd.cut(df['idade'], bins=[10, 20, 30, 40, 50, 60, 70, 80], right=False)
    contagem_faixas = df.groupby(faixas_etarias, observed=True)['idade'].count()
    print("\nContagem de clientes por faixa etária:")
    print(contagem_faixas)
    return contagem_faixas

def analyze_buyers(df):
    compradores_total = df.groupby('idade')['Venda'].sum().sort_values(ascending=False)
    top_20_percent = int(len(compradores_total) * 0.2)
    top_compradores = compradores_total.index[:top_20_percent]
    media_idade_top_compradores = int(np.mean(top_compradores.to_numpy()))
    print(f"\nMédia de idade dos 20% que mais compraram: {media_idade_top_compradores} anos")

def analyze_gender(df):
    contagem_genero = df['genero'].value_counts()
    print("\nContagem de clientes por gênero:")
    print(contagem_genero)
    compradores = df[df['Venda'] > 0]
    contagem_genero_compradores = compradores['genero'].value_counts()
    genero_maioria_compradores = contagem_genero_compradores.idxmax()
    print(f"\nO gênero que compõe a maioria dos compradores é: {genero_maioria_compradores}")

def analyze_channels(df):
    canais = df.groupby('canal_origem')['Venda'].sum()
    print("\nVendas por canal de origem:")
    print(canais)
    canal_mais_eficaz = canais.idxmax()
    print(f"O canal mais eficaz para investimento em marketing é: {canal_mais_eficaz}")
    return canais

def analyze_feedbacks(df):
    feedbacks = df.groupby('canal_origem')['feedback'].mean()
    print("\nMédia de feedback por canal:")
    print(feedbacks)
    return feedbacks

def calculate_profit(df):
    preco_fixo = 12.90
    df['lucro'] = df['Venda'] * preco_fixo
    lucro_total = df['lucro'].sum()
    print(f"\nO lucro total obtido com as vendas foi: R$ {lucro_total:.2f}")
    lucro_por_canal = df.groupby('canal_origem')['lucro'].sum()
    print("\nLucro por canal de origem:")
    print(lucro_por_canal)
    return lucro_por_canal

def analyze_engagement(df):
    curtidas_por_canal = df.groupby('canal_origem')['Curtidas'].mean()
    compartilhamentos_por_canal = df.groupby('canal_origem')['Compartilhamentos'].mean()
    print("\nMédia de curtidas por canal:")
    print(curtidas_por_canal)
    print("\nMédia de compartilhamentos por canal:")
    print(compartilhamentos_por_canal)
    melhor_rede_curtidas = curtidas_por_canal.idxmax()
    melhor_rede_compartilhamentos = compartilhamentos_por_canal.idxmax()
    print(f"\nA rede com a melhor média de curtidas é: {melhor_rede_curtidas}")
    print(f"A rede com a melhor média de compartilhamentos é: {melhor_rede_compartilhamentos}")

def segment_customers(df):
    features = df[['Venda', 'feedback']].dropna(subset=['Venda', 'feedback'])
    print(f"Número de amostras válidas para segmentação: {len(features)}")
    if len(features) >= 3:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['segmento'] = kmeans.fit_predict(features_scaled)
        segmentos = df.groupby('segmento').agg({
            'feedback': 'mean',
            'Venda': 'mean',
            'canal_origem': 'count',
            'lucro': 'mean'
        })
        segmentos.rename(columns={'canal_origem': 'N_Clientes'}, inplace=True)
        print("\nAnálise dos segmentos:")
        print(segmentos)
        segmento_priorizado = segmentos['Venda'].idxmax()
        print(f"\nO segmento que deve ser priorizado é: Segmento {segmento_priorizado}")
        return segmentos
    else:
        print("Amostras insuficientes para segmentação.")
        return pd.DataFrame()

def prepare_target_variable(df):
    df['sucesso_campanha'] = np.where(df['Venda'] > 0, 1, 0)
    return df

def train_logistic_regression(df):
    print(df['sucesso_campanha'].value_counts())
    if df['sucesso_campanha'].nunique() > 1:  # Verifica se há pelo menos 2 classes
        features = ['feedback', 'idade', 'Curtidas', 'Compartilhamentos'] + list(df.filter(like='canal_origem_').columns) + list(df.filter(like='genero_').columns)
        X = df[features]
        y = df['sucesso_campanha']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[['feedback', 'idade', 'Curtidas', 'Compartilhamentos']])
        X_test_scaled = scaler.transform(X_test[['feedback', 'idade', 'Curtidas', 'Compartilhamentos']])
        X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']), X_train.drop(columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']).reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']), X_test.drop(columns=['feedback', 'idade', 'Curtidas', 'Compartilhamentos']).reset_index(drop=True)], axis=1)
        model = LogisticRegression()
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        print("Acurácia:", accuracy_score(y_test, y_pred))
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))
        print("Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
    else:
        print("Não há classes suficientes para treinar o modelo.")

def plot_results(df):
    # Gráfico de Vendas por Mês
    vendas_por_mes = df.groupby('mes_venda')['Venda'].sum()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=vendas_por_mes.index.astype(str),
        y=vendas_por_mes.values,
        name='Vendas por Mês',
        marker_color='skyblue',
        text=vendas_por_mes.values,
        textposition='auto'
    ))

    fig.update_layout(
        title='Vendas por Mês',
        xaxis_title='Mês',
        yaxis_title='Número de Vendas',
        xaxis_tickangle=-45
    )
    fig.show()

    # Gráfico de Distribuição Etária
    contagem_faixas = analyze_age_distribution(df)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=contagem_faixas.index.astype(str),
        y=contagem_faixas.values,
        name='Distribuição Etária',
        marker_color='lightgreen',
        text=contagem_faixas.values,
        textposition='auto'
    ))

    fig.update_layout(
        title='Distribuição Etária',
        xaxis_title='Faixa Etária',
        yaxis_title='Número de Clientes'
    )
    fig.show()

    # Gráfico de Vendas por Canal
    canais = analyze_channels(df)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=canais.index,
        y=canais.values,
        name='Vendas por Canal de Origem',
        marker_color='coral',
        text=canais.values,
        textposition='auto'
    ))

    fig.update_layout(
        title='Vendas por Canal de Origem',
        xaxis_title='Canal de Origem',
        yaxis_title='Vendas'
    )
    fig.show()

    # Gráfico de Feedback por Canal
    feedbacks = analyze_feedbacks(df)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=feedbacks.index,
        y=feedbacks.values,
        name='Feedback Médio por Canal',
        marker_color='gold',
        text=feedbacks.values,
        textposition='auto'
    ))

    fig.update_layout(
        title='Feedback Médio por Canal',
        xaxis_title='Canal de Origem',
        yaxis_title='Feedback Médio'
    )
    fig.show()

def main():
    df = read_data()
    df = preprocess_data(df)
    analyze_sales(df)
    analyze_buyers(df)
    analyze_gender(df)
    calculate_profit(df)
    analyze_engagement(df)
    df = prepare_target_variable(df)
    segment_customers(df)
    train_logistic_regression(df)
    plot_results(df)

if __name__ == "__main__":
    main()
