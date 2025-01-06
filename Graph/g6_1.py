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
    with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/NCCN/half_ei_4_2024.csv', 'r') as f:
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
# Check and load/create KnowledgeGraphIndex
if os.path.exists('./storage_graph'):
    print("Loading existing KnowledgeGraphIndex...")
    storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
    kg_index = load_index_from_storage(storage_context=storage_context)
else:
    print("Creating new KnowledgeGraphIndex...")
    documents = load_documents()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=20,
        include_embeddings=True,
    )
    kg_index.storage_context.persist(persist_dir='./storage_graph')


# %%
# Check and load/create VectorStoreIndex
if os.path.exists('./storage_vector'):
    print("Loading existing VectorStoreIndex...")
    storage_context_vector = StorageContext.from_defaults(persist_dir='./storage_vector')
    vector_index = load_index_from_storage(storage_context=storage_context_vector)
else:
    print("Creating new VectorStoreIndex...")
    if 'documents' not in locals():
        documents = load_documents()
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_index.storage_context.persist(persist_dir='./storage_vector')

print("\nToken usage after index creation/loading:")
print_token_usage()

# %%
# 5. Prepare for different query approaches

## 5.1 text-to-GraphQuery
# from llama_index.core.query_engine import KnowledgeGraphQueryEngine

# nl2kg_query_engine = KnowledgeGraphQueryEngine(
#     storage_context=kg_index.storage_context,
#     llm=llm,
#     verbose=True,
# )

## 5.2 Graph RAG query engine
kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

## 5.3 Vector RAG query engine
vector_rag_query_engine = vector_index.as_query_engine()

# %%
## 5.4 Graph+Vector RAG query engine


class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, kg_retriever, mode="OR"):
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._mode = mode


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(index=kg_index, retriever_mode="keyword", include_text=False)
custom_retriever = CustomRetriever(vector_retriever, kg_retriever)


# %%
# from llama_index import get_response_synthesizer
# from llama_index.query_engine import RetrieverQueryEngine

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

graph_vector_rag_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# %%
# 6. Query with all the Engines

query = "What are the initial steps for prostate cancer diagnosis?"

token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
print_token_usage()

token_counter.reset_counts()
print("\nGraph RAG result:")
response_graph_rag = kg_rag_query_engine.query(query)
print(response_graph_rag)
print_token_usage()

token_counter.reset_counts()
print("\nVector RAG result:")
response_vector_rag = vector_rag_query_engine.query(query)
print(response_vector_rag)
print_token_usage()

token_counter.reset_counts()
print("\nGraph + Vector RAG result:")
response_graph_vector_rag = graph_vector_rag_query_engine.query(query)
print(response_graph_vector_rag)
print_token_usage()

# %%
# 7. Conclusion

# Print total token usage and cost
print("\nTotal Token Usage:")
print_token_usage()

# Compare results and analyze performance


