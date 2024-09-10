import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

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

def analyze_buyers(df):
    # Análise de maiores compradores
    compradores_total = df.groupby('idade')['Venda'].sum().sort_values(ascending=False)
    top_20_percent = int(len(compradores_total) * 0.2)
    top_compradores = compradores_total.index[:top_20_percent]
    media_idade_top_compradores = int(np.mean(top_compradores.to_numpy()))
    menssage = (f"{media_idade_top_compradores}")
    return menssage

def analyze_age_distribution(df):
    # Contagem de clientes por faixa etária
    faixas_etarias = pd.cut(df['idade'], bins=[10, 20, 30, 40, 50, 60, 70, 80], right=False)
    contagem_faixas = df.groupby(faixas_etarias, observed=False)['idade'].count().reset_index(name='Número de Clientes')
    contagem_faixas.rename(columns={'faixa_etaria': 'Faixa Etária'}, inplace=True)
    
    return contagem_faixas, analyze_buyers(df)


def analyze_gender(df):
    # Análise de gênero
    contagem_genero = df['genero'].value_counts().reset_index()
    contagem_genero.columns = ['Gênero', 'Contagem']
    
    compradores = df[df['Venda'] > 0]
    contagem_genero_compradores = compradores['genero'].value_counts()
    genero_maioria_compradores = contagem_genero_compradores.idxmax()
    
    result_message = f"O gênero que compõe a maioria dos compradores é: {genero_maioria_compradores}"
    return contagem_genero, result_message

def analyze_channels(df):
    # Análise de canais de comunicação que mais trazem resultados
    canais = df.groupby('canal_origem')['Venda'].sum().reset_index()
    canais.columns = ['Canal', 'Vendas']
    
    canal_mais_eficaz = canais.loc[canais['Vendas'].idxmax(), 'Canal']
    result_message = f"O canal mais eficaz para investimento em marketing é: {canal_mais_eficaz}"
    return canais, result_message

def analyze_feedbacks(df):
    # Análise de feedbacks
    feedbacks = df.groupby('canal_origem')['feedback'].mean().reset_index()
    feedbacks.columns = ['Canal', 'Média de Feedback']
    return feedbacks, ""

def calculate_profit(df):
    # Cálculo do lucro com base no preço fixo
    preco_fixo = 12.90
    df['lucro'] = df['Venda'] * preco_fixo
    lucro_total = df['lucro'].sum()
    lucro_por_canal = df.groupby('canal_origem')['lucro'].sum().reset_index()
    lucro_por_canal.columns = ['Canal', 'Lucro']
    
    result_message = f"O lucro total obtido com as vendas foi: R$ {lucro_total:.2f}"
    return lucro_por_canal, result_message

def analyze_engagement(df):
    # Análise de curtidas e compartilhamentos
    curtidas_por_canal = df.groupby('canal_origem')['Curtidas'].mean().reset_index()
    compartilhamentos_por_canal = df.groupby('canal_origem')['Compartilhamentos'].mean().reset_index()
    
    curtidas_por_canal.columns = ['Canal', 'Média de Curtidas']
    compartilhamentos_por_canal.columns = ['Canal', 'Média de Compartilhamentos']
    
    result_message = (
        f"A rede com a melhor média de curtidas é: {curtidas_por_canal.loc[curtidas_por_canal['Média de Curtidas'].idxmax(), 'Canal']}\n"
        f"A rede com a melhor média de compartilhamentos é: {compartilhamentos_por_canal.loc[compartilhamentos_por_canal['Média de Compartilhamentos'].idxmax(), 'Canal']}"
    )
    
    return pd.merge(curtidas_por_canal, compartilhamentos_por_canal, on='Canal'), result_message

def segment_customers(df):
   # Pré-processamento para segmentação de clientes
    features = df[['Venda', 'feedback']].dropna(subset=['Venda', 'feedback'])

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
            'canal_origem': 'count'
        })

        # Renomeando a coluna de contagem de clientes
        segmentos.rename(columns={'canal_origem': 'N_Clientes'}, inplace=True)

        # Decisão sobre quais segmentos priorizar
        segmento_priorizado = segmentos['Venda'].idxmax()
        message = (f"{segmento_priorizado}")
        return segmentos, message
    else:
        print("Amostras insuficientes para segmentação.")
def show_table(df):
    for widget in frame_table.winfo_children():
        widget.destroy()
    
    tree = ttk.Treeview(frame_table, columns=list(df.columns), show='headings')
    tree.pack(expand=True, fill='both')

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    for _, row in df.iterrows():
        tree.insert('', 'end', values=list(row))

def show_message(message):
    for widget in frame_message.winfo_children():
        widget.destroy()
    
    text = ScrolledText(frame_message, wrap=tk.WORD, width=60, height=10)
    text.pack(expand=True, fill='both')
    text.insert(tk.END, message)
    text.config(state=tk.DISABLED)

def show_table_with_result(df, analysis_function, result_label):
    table, result_message = analysis_function(df)
    if not table.empty:
        show_table(table)
    show_message(result_label + "\n" + result_message)

def run_analysis(analysis_function, result_label):
    df = read_data()
    df = preprocess_data(df)
    show_table_with_result(df, analysis_function, result_label)

def create_menu(root):
    menu = tk.Menu(root)
    root.config(menu=menu)
    
    analysis_menu = tk.Menu(menu, tearoff=0)
    menu.add_cascade(label="Análises", menu=analysis_menu)
    
    analysis_menu.add_command(label="Gênero", command=lambda: run_analysis(analyze_gender, "Análise de Gênero"))
    analysis_menu.add_command(label="Faixa Etária", command=lambda: run_analysis(analyze_age_distribution, "Média de idade dos 20% que mais compraram: "))
    analysis_menu.add_command(label="Canais de Comunicação", command=lambda: run_analysis(analyze_channels, "Análise de Canais de Comunicação"))
    analysis_menu.add_command(label="Feedbacks", command=lambda: run_analysis(analyze_feedbacks, ""))
    analysis_menu.add_command(label="Lucro", command=lambda: run_analysis(calculate_profit, "Análise de Lucro"))
    analysis_menu.add_command(label="Engajamento", command=lambda: run_analysis(analyze_engagement, "Análise de Engajamento"))
    analysis_menu.add_command(label="Segmentação de Clientes", command=lambda: run_analysis(segment_customers, "O segmento que deve ser priorizado é:"))

root = tk.Tk()
root.title("Análise de Dados")

frame_table = tk.Frame(root)
frame_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_message = tk.Frame(root)
frame_message.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

create_menu(root)

root.mainloop()

