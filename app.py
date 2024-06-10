from utils import *



def main(file_path, question):
    data = read_csv(file_path)
    print("Data Loaded Successfully")
    
    try:
        statistics = calculate_statistics(data,file_path)
        print("Statistics Calculated:\n", statistics)
    except ValueError as e:
        print(e)
        return
    
    columns = data.columns
    generate_plots(data, file_path)
    
    chain = setup_retrieval_chain(file_path,DB_FAISS_PATH)
    
    history = []
    answer, history = conversational_chat(chain, question, history)
    print("Answer from Llama-2:", str(answer))


# Example usage
if __name__ == "__main__":
    file_path= '2019.csv'
    DB_FAISS_PATH = 'vectorstore/db_faiss2'
    question = "what was the name of the countryor region with lowest GDP per capita?"
    main(file_path, question)


