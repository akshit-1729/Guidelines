# Explanation: This script integrates a language chain approach with Graph RAG.
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
# from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
# from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver

from dotenv import load_dotenv

load_dotenv()

# graph = Neo4jGraph()

loader = TextLoader(file_path="dummytext.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
documents = text_splitter.split_documents(documents=docs)
llm = OllamaFunctions(model="llama3.1", temperature=0, format="json")

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)

graph_documents[0]