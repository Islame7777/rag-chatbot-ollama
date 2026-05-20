from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    embeddings = OllamaEmbeddings(model="llama3.2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    llm = OllamaLLM(model="llama3.2")
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
Reponds a la question en utilisant uniquement le contexte suivant:

Contexte: {context}

Question: {question}

Reponse:""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Chatbot RAG pret ! Tape 'exit' pour quitter.\n")
    
    while True:
        question = input("Tu : ")
        if question.lower() == "exit":
            break
        reponse = chain.invoke(question)
        print(f"\nBot : {reponse}\n")

main()