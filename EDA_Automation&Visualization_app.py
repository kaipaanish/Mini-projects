import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda_analysis(file_path):
    df=pd.read_csv(file_path)
    
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace = True)
        
        
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    summary = df.describe(include='all').to_string()
    
    missing_values = df.isnull().sum().to_string()
    
    insights = generate_ai_insights(summary)
    
    plot_path = generate_visualizations(df)
    
    return f"\n Data Loaded Scuccessfully!\n\nSummary:\n{summary}\n\nMissing Values:\n{missing_values}\n\nInsights:\n{insights}", plot_path


import ollama
def generate_ai_insights(df_summary):
    prompt = f"Analyze the following dataset and provide insights:\n\n{df_summary}\n\n"
    response = ollama.chat(model = "mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def generate_visualizations(df):
    plot_paths = []
    
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col}')
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
        
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        path = 'correlation_heatmap.png'
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
    return plot_paths

app = gr.Interface(fn = eda_analysis,
                   inputs = gr.File(type = "filepath"),
                   outputs = [gr.Textbox(label = "EDA Summary"), gr.Gallery(label = "Visualizations", columns = 2)],
                   title = "EDA Automation App",
                   description= "Automate your EDA process with AI insights and visualizations.")

app.launch(share=True)
