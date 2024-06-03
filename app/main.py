import json

from pydantic import BaseModel
from fastapi import FastAPI

from langchain.storage import InMemoryStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Classe usada para as requisições do FastAPI
class APIRequest(BaseModel):
    question: str = 'Insira a pergunta para o Llama3!'

# InMemoryStore é área de armazenamento que manterá os documentos
docstore = InMemoryStore()
id_key = "tabagismo_id"

# Embedding (https://github.com/qdrant/fastembed/)
embedding = FastEmbedEmbeddings()

# Banco chroma que irá manter os vetores de informação
vectorstore = Chroma(collection_name="summaries11",
                     embedding_function=embedding)

# Definição do MultiVectorRetriever que irá organizar os
# dados do chroma e as informações dos documentos
retriever = MultiVectorRetriever(vectorstore=vectorstore,
                                 docstore=docstore,
                                 id_key=id_key)

# Leitura dos documentos do PDF 
docs_id, docs = [], []
with open('data/interim/pdf_to_text.json', 'r') as f:
    data = json.load(f)

    for i, text in enumerate(data['text']):
        docs_id += [str(i)]
        docs += [Document(page_content=text, metadata={id_key: str(i)})]
 
# Adiciona os documentos no MultiVectorRetriever que será utilizado
# para inserir os dados de contexto
retriever.vectorstore.add_documents(docs)
retriever.docstore.mset(list(zip(docs_id, docs)))

# Template usado para o prompt da LLM. (Llama3)
template = """
Context: Você é um assistente especialista no Protocolo Clínico e diretrizes Terapeuticas do Tabagismo.\
Os dados que você tem disponível se trata de um documento disponibilizado pelo INCA (Instituto Nacional do Câncer).\
Todo questionamento respondido deve utilizar como base os dados passados no texto abaixo, caso não seja possível\
solucionar a dúvida, informe que não possuí informações disponíveis. Aqui está o texto disponível: \
{context}
Question: {question}
"""

# Inicialização do modelo
prompt = ChatPromptTemplate.from_template(template)
model = Ollama(model="llama3")

# Langchain contendo o fluxo utilizado
chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt | model | StrOutputParser())

# Inicialização da API
app = FastAPI()

# Chamada da api de Llama3, que recebe a pergunta por requisição
@app.post("/api/v1/llama3")
def ask_llama3(req: APIRequest):
    message = ''

    try:  
        message = chain.invoke(req.question) 
    except Exception as e:
        message = f'error on llama3. {str(e)}'
    
    return {'message': message}