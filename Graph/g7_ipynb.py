# Explanation: This script uses notebook-style code to illustrate graph querying techniques.
# %%
# 1. Preparation

## 1.1 Prepare for LLM

import os
import csv
from llama_index.core import Document, KnowledgeGraphIndex, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from typing import List
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine





# %%

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Set up token counter
token_counter = TokenCountingHandler(
    tokenizer=None,  # Use default tokenizer
    verbose=True  # Set to False if you don't want per-query prints
)

callback_manager = CallbackManager([token_counter])

# Define LLM with the callback manager
llm = OpenAI(temperature=0, model="gpt-4o", callback_manager=callback_manager)

# Configure settings
Settings.llm = llm
Settings.chunk_size = 1024
Settings.callback_manager = callback_manager

## 1.2. Prepare for SimpleGraphStore as Graph Store

graph_store = SimpleGraphStore()


# %%
# Function to print token usage and cost
def print_token_usage():
    print(f"Prompt Tokens: {token_counter.prompt_llm_token_count}")
    print(f"Completion Tokens: {token_counter.completion_llm_token_count}")
    print(f"Total Tokens: {token_counter.total_llm_token_count}")
    # Assuming a cost of $0.002 per 1K tokens for gpt-3.5-turbo-instruct
    cost = (token_counter.total_llm_token_count / 1000) * 0.002
    print(f"Estimated Cost: ${cost:.4f}")


# %%
# 2. Build or Load the Knowledge Graph and Vector Index

token_counter.reset_counts()

# Function to load documents
def load_documents():
    documents = []
    with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/NCCN/ei_4_2024.csv', 'r') as f:
        csv_reader = csv.DictReader(f)
        print("CSV Column names:", csv_reader.fieldnames)
        for row in csv_reader:
            content = f"Node ID: {row['Node ID']}\n"
            content += f"Page Key: {row['Page Key']}\n"
            content += f"Page No: {row['Page No']}\n"
            content += f"Information: {row['Information']}\n"
            content += f"Footnotes: {row['Footnotes']}\n"
            documents.append(Document(text=content))
    return documents


# %%
# Check and load/create VectorStoreIndex
if os.path.exists('./new/storage_vector'):
    print("Loading existing VectorStoreIndex...")
    storage_context_vector = StorageContext.from_defaults(persist_dir='./new/storage_vector')
    vector_index = load_index_from_storage(storage_context=storage_context_vector)
else:
    print("Creating new VectorStoreIndex...")
    if 'documents' not in locals():
        documents = load_documents()
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_index.storage_context.persist(persist_dir='./new/storage_vector')

print("\nToken usage after index creation/loading:")
print_token_usage()

# %%
## 5.3 Vector RAG query engine
vector_rag_query_engine = vector_index.as_query_engine()

# %%
# 6. Query with all the Engines

query = "What are the preconditions for giving docetaxel?"

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

# token_counter.reset_counts()
# print("\nGraph RAG result:")
# response_graph_rag = kg_rag_query_engine.query(query)
# print(response_graph_rag)
# print_token_usage()

token_counter.reset_counts()
print("\nVector RAG result:")
response_vector_rag = vector_rag_query_engine.query(query)
print(response_vector_rag)
print_token_usage()

# token_counter.reset_counts()
# print("\nGraph + Vector RAG result:")
# response_graph_vector_rag = graph_vector_rag_query_engine.query(query)
# print(response_graph_vector_rag)
# print_token_usage()

# %%
# 6. Query with all the Engines

query = "When is docetaxel suggested? give all instances."

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

# token_counter.reset_counts()
# print("\nGraph RAG result:")
# response_graph_rag = kg_rag_query_engine.query(query)
# print(response_graph_rag)
# print_token_usage()

token_counter.reset_counts()
print("\nVector RAG result:")
response_vector_rag = vector_rag_query_engine.query(query)
print(response_vector_rag)
print_token_usage()

# token_counter.reset_counts()
# print("\nGraph + Vector RAG result:")
# response_graph_vector_rag = graph_vector_rag_query_engine.query(query)
# print(response_graph_vector_rag)
# print_token_usage()

# %%
# 6. Query with all the Engines

query = "What steps should be taken after docetaxel?"

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

# token_counter.reset_counts()
# print("\nGraph RAG result:")
# response_graph_rag = kg_rag_query_engine.query(query)
# print(response_graph_rag)
# print_token_usage()

token_counter.reset_counts()
print("\nVector RAG result:")
response_vector_rag = vector_rag_query_engine.query(query)
print(response_vector_rag)
print_token_usage()

# token_counter.reset_counts()
# print("\nGraph + Vector RAG result:")
# response_graph_vector_rag = graph_vector_rag_query_engine.query(query)
# print(response_graph_vector_rag)
# print_token_usage()

# %%
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON data
with open('./storage_graph/graph_store.json', 'r') as file:
    data = json.load(file)

# Create a new graph
G = nx.Graph()

# Process the graph data
for node, relationships in data['graph_dict'].items():
    for relationship in relationships:
        if relationship[0] == "Is":
            G.add_edge(node, relationship[1])

# Set up the plot
plt.figure(figsize=(20, 20))

# Draw the graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1000, font_size=8, font_weight='bold')

# Add edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show the plot
plt.title("Graph Visualization", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON data
with open('./storage_graph/graph_store.json', 'r') as file:
    data = json.load(file)

# Create a new graph
G = nx.DiGraph()

# Process the graph data
for node, relationships in data['graph_dict'].items():
    for relationship in relationships:
        G.add_edge(node, relationship[1], relationship=relationship[0])

# Set up the plot
plt.figure(figsize=(30, 30))

# Draw the graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=8, font_weight='bold', 
        arrows=True, arrowsize=20)

# Add edge labels (relationships)
edge_labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

# Add node labels
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

# Show the plot
plt.title("Graph Visualization with Relationships", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON data
with open('./storage_graph/graph_store.json', 'r') as file:
    data = json.load(file)

# Create a new graph
G = nx.DiGraph()

# Process the graph data
for node, relationships in data['graph_dict'].items():
    for relationship in relationships:
        G.add_edge(node, relationship[1], relationship=relationship[0])

# Set up the plot with a much larger figure size and higher DPI
plt.figure(figsize=(60, 60), dpi=300)

# Draw the graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=False, node_color='lightblue', 
        node_size=3000, arrows=True, arrowsize=20)

# Add edge labels (relationships)
edge_labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Add node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Show the plot
plt.title("Graph Visualization with Relationships", fontsize=40)
plt.axis('off')
plt.tight_layout()

# Save the plot as a high-resolution PNG file
plt.savefig('graph_visualization.png', format='png', dpi=300, bbox_inches='tight')
print("Graph visualization saved as 'graph_visualization.png'")

# Optionally, you can still display the plot if you want
# plt.show()

# %%
query = "What steps should be taken after docetaxel?"
answer = "After docetaxel, several options can be considered depending on the patient's specific condition and treatment response.\
    Recommended regimens include cabazitaxel, which is a category 1 option, and a rechallenge with docetaxel. In certain circumstances,\
    cabazitaxel combined with carboplatin may be useful, especially for patients with aggressive variant prostate cancer or unfavorable\
    genomics. Other treatments such as lutetium Lu 177 vipivotide tetraxetan for PSMA-positive metastases, mitoxantrone for palliation,\
    olaparib for HRR mutation, pembrolizumab for specific genetic markers, radium-223 for symptomatic bone metastases, and rucaparib for\
    BRCA mutation may also be considered based on the patient's specific clinical scenario."


# %%
query = "When is docetaxel suggested? give all instances."
answer1 = "The context does not provide specific preconditions for administering docetaxel."
answer2 = "Docetaxel is suggested in the following instances: as a rechallenge\
    regimen and in combination with carboplatin."


