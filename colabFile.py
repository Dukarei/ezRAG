
#NOTE: this is designed to be run bit-by bit in a google colab notebook 
#code to install imports
!pip install langchain
!pip install langchain_core
!pip install langchain_community
!pip install pypdf
!pip install ollama
!pip install chromadb
!pip install -qU langchain-ollama

#code to allow running terminal in a colab notebook to install and serve ollama for usage
"""
1. Install ollama
2. Run ollama pull for llama 3.1:8b and nomic-text-embed:latest, although any llm and text-embed model will do as long as you replace them in the models section of the code.
3. run ollama serve to allow usage in background
"""
!pip install colab-xterm
%load_ext colabxterm
%xterm
"""
docQuery
a locally-run RAG that can be trained on any input files, although intended for
use with coding docs.

an interesting alternative use would be to somehow pipeline together an
internet searching/PDF generating model with this one in order to allow live
"training"

This is a temporary script file.
"""

import os
#ollama allows us to call our models for use in the code
import ollama
#chroma is a vector DB allowing us to store embedded content of provided documents
from langchain_community.vectorstores import Chroma
#Document isn a class for storing a piece of text as well as associated metadata
from langchain.schema import Document
#integration with ollama embedding model
from langchain_community.embeddings import OllamaEmbeddings
#prompt templates
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
#chatOllama allows us to interact directly with ollama
from langchain_community.chat_models import ChatOllama
#used for passing prompts to the model
from langchain_core.runnables import RunnablePassthrough
#MultiQueryRetriever allows a separate LLM to tune any input prompts by generating
#multiple alternative prompts from multiple perspectives for any singular input
from langchain.retrievers.multi_query import MultiQueryRetriever
#the next loads pdf and parses text
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
#text splitter thangy
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""end of imports"""

""""models and parameters"""

#models to embed and process data
embedModel = 'nomic-embed-text:latest'
retrieveModel = 'deepseek-r1:70b'
llmModel = 'deepseek-r1:70b'

#folder where pdf files are stored(params)
sourceDirectory = "/home/"
dbName = "jlex_DB"
"""end of models and params"""

#create list of PDF files by iterating the full working directory
fileNames = [fileName for fileName in os.listdir(sourceDirectory) if fileName.endswith('.pdf')|fileName.endswith('.cl')|fileName.endswith('.txt')] # could add '.pdf'/'.txt' in this case using .cl for code
#hihglight down to this line and click run selection to check if documents are properly retrieved
#then type whos and fileNames in spyder terminal to check your documents loaded correctly


pageList = []

#load PDF files, covnert to text, create list of text contents
for file in fileNames:
    filePath = os.path.join(sourceDirectory,file)
    if file.endswith('.cl') == True | file.endswith('.txt') == True:
        loader = TextLoader(file_path = filePath)
    else:
      #loader = PyPDFLoader(file_path = filePath)  #alternately use PyPDFLoader/TextLoader
      loader = TextLoader(file_path = filePath)
    pages = loader.load()
    pageList.extend(pages)

#next, split the text into overlapping chunks for embedding/vector storage

text_Splitter = RecursiveCharacterTextSplitter(chunk_size = 200,
                                               chunk_overlap = 20,
                                               add_start_index = True)
#list to store textChunks
textSplits=[]
#another list to store metadata related to text chunks
textSplitsMetaData=[]
for page in pageList:
    split = text_Splitter.split_text(page.page_content)
    textSplits.extend(split)
    PM = page.metadata
    for i in range(len(split)):
        textSplitsMetaData.append(PM)

#embed every text chunk
embeddings = []
for split in textSplits:
    embedding = ollama.embeddings(model = embedModel, prompt = split)
    embeddings.append(embedding)
#textSplits are the text, textMeta is data about where the chunk came from,
#and embeddings are the number-ified vector representations

#a list to store document objects
DocumentObjectList = [Document(page_content = data[0], metadata = data[1]) for data in zip(textSplits, textSplitsMetaData)]

#add embedding to 7the database

vectorDataBase = Chroma.from_documents(documents = DocumentObjectList, embedding = OllamaEmbeddings(model = embedModel
                                                                                                    ,show_progress = True),
                                                                                                    persist_directory=f"/home/databases/{dbName}")
vectorDataBase.persist()
"""
docQuery
a locally-run RAG that can be trained on any input files, although intended for
use with coding docs.

an interesting alternative use would be to somehow pipeline together an
internet searching/PDF generating model with this one in order to allow live
"training"

This is a temporary script file.
"""

import os
#ollama allows us to call our models for use in the code
import ollama
#chroma is a vector DB allowing us to store embedded content of provided documents
from langchain_community.vectorstores import Chroma
#Document isn a class for storing a piece of text as well as associated metadata
from langchain.schema import Document
#integration with ollama embedding model
from langchain_community.embeddings import OllamaEmbeddings
#prompt templates
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
#chatOllama allows us to interact directly with ollama
from langchain_community.chat_models import ChatOllama
#used for passing prompts to the model
from langchain_core.runnables import RunnablePassthrough
#MultiQueryRetriever allows a separate LLM to tune any input prompts by generating
#multiple alternative prompts from multiple perspectives for any singular input
from langchain.retrievers.multi_query import MultiQueryRetriever
#the next loads pdf and parses text
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
#text splitter thangy
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""end of imports"""

""""models and parameters"""

#models to embed and process data
embedModel = 'nomic-embed-text:latest'
retrieveModel = 'deepseek-r1:70b'
llmModel = 'deepseek-r1:70b'

#folder where pdf files are stored(params)
sourceDirectory = "/home/"
dbName = "jlex_DB"
"""end of models and params"""

#create list of PDF files by iterating the full working directory
fileNames = [fileName for fileName in os.listdir(sourceDirectory) if fileName.endswith('.pdf')|fileName.endswith('.cl')|fileName.endswith('.txt')] # could add '.pdf'/'.txt' in this case using .cl for code
#hihglight down to this line and click run selection to check if documents are properly retrieved
#then type whos and fileNames in spyder terminal to check your documents loaded correctly


pageList = []

#load PDF files, covnert to text, create list of text contents
for file in fileNames:
    filePath = os.path.join(sourceDirectory,file)
    if file.endswith('.cl') == True | file.endswith('.txt') == True:
        loader = TextLoader(file_path = filePath)
    else:
      #loader = PyPDFLoader(file_path = filePath)  #alternately use PyPDFLoader/TextLoader
      loader = TextLoader(file_path = filePath)
    pages = loader.load()
    pageList.extend(pages)

#next, split the text into overlapping chunks for embedding/vector storage

text_Splitter = RecursiveCharacterTextSplitter(chunk_size = 200,
                                               chunk_overlap = 20,
                                               add_start_index = True)
#list to store textChunks
textSplits=[]
#another list to store metadata related to text chunks
textSplitsMetaData=[]
for page in pageList:
    split = text_Splitter.split_text(page.page_content)
    textSplits.extend(split)
    PM = page.metadata
    for i in range(len(split)):
        textSplitsMetaData.append(PM)

#embed every text chunk
embeddings = []
for split in textSplits:
    embedding = ollama.embeddings(model = embedModel, prompt = split)
    embeddings.append(embedding)
#textSplits are the text, textMeta is data about where the chunk came from,
#and embeddings are the number-ified vector representations

#a list to store document objects
DocumentObjectList = [Document(page_content = data[0], metadata = data[1]) for data in zip(textSplits, textSplitsMetaData)]

#add embedding to 7the database

vectorDataBase = Chroma.from_documents(documents = DocumentObjectList, embedding = OllamaEmbeddings(model = embedModel
                                                                                                    ,show_progress = True),
                                                                                                    persist_directory=f"/home/databases/{dbName}")
vectorDataBase.persist()
#above represents steps 1-3, all that is left is retrieval and generation run the above to save
#run the below code to recursively prompt model
#vectorDataBase = Chroma(persist_directory=f"/home/databases/{dbName}", embedding_function = OllamaEmbeddings(model = embedModel, show_progress = True))

model = ChatOllama(model = llmModel)

#below we use multi-query retrieval
queryPrompt = PromptTemplate(input_variables=["question"],
                             template = """You are an AI language model assistant. Your task is
to generate different versions of the given user question to retrieve relevant documents from a
vector database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. Provide these
alternative questions separated by newLines. Original question: {question}""",)

retriever = MultiQueryRetriever.from_llm(vectorDataBase.as_retriever(),
                                         ChatOllama(model = retrieveModel),
                                         prompt = queryPrompt)

#RAG prompt - idea is to use the DB as primary source
templateRAG = """First try to answer the question based ONLY on the following context:
    {context} Question: {question} and if you cannot answer then use LLM knowledge to help"""

#prompt object
prompt = ChatPromptTemplate.from_template(templateRAG)
#create chain
chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    )

#next we finally ask a question

question = '''Question: blah'''

#pass question to the RAG
response = chain.invoke(question)
print(response.content)

"""
#save answer in a file
with open('output.txt','w',encoding = "utf-8") as text_file:
    text_file.write(response.content)
"""

