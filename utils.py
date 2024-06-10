from langchain_community.llms import CTransformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 1000,
        temperature = 0.5
    )
    return llm



def read_csv(file_path):
    return pd.read_csv(file_path)



def calculate_statistics(data, file_path):
    # Select only Salary and Experience columns for the 2019.csv file
    if file_path == '2019.csv':
        numeric_data = data[['GDP per capita', 'Healthy life expectancy']].select_dtypes(include=[np.number])
    else:
        numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric columns found in the CSV file.")
    
    statistics = {
        "mean": numeric_data.mean(),
        "median": numeric_data.median(),
        "mode": numeric_data.mode().iloc[0],
        "std_dev": numeric_data.std(),
        "correlation": numeric_data.corr()
    }
    return statistics


def generate_plots(data, file_path):
    # Select only Salary and Experience columns for the 2019.csv file
    if file_path == '2019.csv':
        numeric_data = data[['GDP per capita', 'Healthy life expectancy']].select_dtypes(include=[np.number])
    else:
        numeric_data = data.select_dtypes(include=[np.number])
    
    for column in numeric_data.columns:
        plt.figure()
        sns.histplot(numeric_data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    for i, col1 in enumerate(numeric_data.columns):
        for col2 in numeric_data.columns[i+1:]:
            plt.figure()
            sns.scatterplot(x=numeric_data[col1], y=numeric_data[col2])
            plt.title(f'Scatter plot of {col1} vs {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)

    for column in numeric_data.columns:
        plt.figure()
        sns.lineplot(data=numeric_data[column])
        plt.title(f'Line plot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)

    # Show all plots together
    plt.show()




def setup_retrieval_chain(file_path,DB_FAISS_PATH):
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    return chain

def conversational_chat(chain, query, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"], history


