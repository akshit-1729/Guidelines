# %% [markdown]
# install required packages

# %%
# %pip install --upgrade --quiet  langchain langchain-community langchain-ollama langchain-experimental neo4j tiktoken yfiles_jupyter_graphs python-dotenv json-repair langchain-openai langchain_core

# %%
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver

from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# setup neo4j

# %%
# NEO4J_URI="neo4j+s://ff9635e0.databases.neo4j.io"
# NEO4J_USERNAME="neo4j"
# NEO4J_PASSWORD="FlpkaG2M_jHx2yxE_HkdHFAn7oU2d-ldDxM2rrlvHC8"
# AURA_INSTANCEID="ff9635e0"
# AURA_INSTANCENAME="Instance01"

# %%
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="00000000"
# NEO4J_DATABASE="langchain"


# %%
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# %%
os.environ["OPENAI_API_KEY"] = ""

# %%
''' use gpt '''
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

llm_transformer = LLMGraphTransformer(llm=llm)

# %%
'''creating graph documents by using a LLM'''
# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# %% [markdown]
# create graph documents from jsonld file

# %%
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

# %%
import json
with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {}
count = 0
for item in data.get('@graph', []):
    count += 1
    # node_id = item.get('@id')
    # if node_id and node_id not in nodes:
    #     nodes[node_id] = Node(id=node_id, properties={})
print(count)

# %%
import json

# Load the JSON-LD file
with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Relationships to extract
relationship_keys = [
    # 'nccn:page-key',  ??? -> keep in properties
    'nccn:labels',
    # 'nccn:content',  ???    where to keep?
    'nccn:next',
    'nccn:prev',
    'nccn:parent',
    'nccn:reference',
    'nccn:contains'
    # 'nccn:nodelink'.  ??? -> not required
]
### PROCEED WITH THIS AND CREATE FEW GRAPH DOCUMENTS AND SEE
nodes = {}
relationships = []
graph_documents = []

# Create Node instances
for item in data.get('@graph', []):
    node_id = item.get('@id')
    metadata = { "nccn:page-key": item.get("nccn:page-key"),
                 "nccn:page-no": item.get("nccn:page-no") }
    if node_id and node_id not in nodes:
        nodes[node_id] = Node(id=node_id, properties=metadata)

# Create Relationship instances
for item in data.get('@graph', []):
    source_id = item.get('@id')
    source_node = nodes.get(source_id)
    if not source_node:
        continue
    per_node_rels = []
    for rel_type in relationship_keys:
        if rel_type in item:
            targets = item[rel_type]
            if not isinstance(targets, list):
                targets = [targets]
            for target in targets:
                # Determine target ID
                if isinstance(target, dict) and '@id' in target:
                    target_id = target['@id']
                else:
                    target_value = str(target)
                    if target_value.startswith('http://'):
                        target_id = target_value
                    else:
                        target_id = target_value
                # Create target node if it doesn't exist
                if target_id not in nodes:
                    nodes[target_id] = Node(id=target_id, properties={})
                target_node = nodes[target_id]
                # Create Relationship
                per_node_rels.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel_type,
                    properties={}
                ))
                relationships.extend(per_node_rels)

    # create a graph document for the node
    source_doc = Document(metadata={'source': 'NCCN_prostate_4.2024_Graph_12_33.json'}, page_content=item.get('nccn:content'))
    graph_documents.append(GraphDocument(
        nodes=[source_node],
        relationships=per_node_rels,
        source=source_doc,
    ))

# Output Nodes
node_counter = 1
node_name_map = {}
for node_id, node in nodes.items():
    node_var_name = f'node{node_counter}'
    node_name_map[node_id] = node_var_name
    print(f'{node_var_name} = {node}')
    node_counter += 1

# Output Relationships
rel_counter = 1
for rel in relationships:
    source_name = node_name_map[rel.source.id]
    target_name = node_name_map[rel.target.id]
    print(f'rel{rel_counter} = Relationship(source={source_name}, target={target_name}, type="{rel.type}", properties={{}})')
    rel_counter += 1

# %%
graph.add_graph_documents(graph_documents)

# %%
graph.refresh_schema()

# %%
from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(
#     model="mxbai-embed-large",
# )

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)


from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    
    
)
vector_retriever = vector_index.as_retriever()

# %%
print(vector_index.similarity_search("http://nccn-guideline.org/nsclc/24"))

# %%
# vector_index.add_documents([Document(page_content="foo")])
# docs_with_score = vector_index.similarity_search_with_score("foo")
# docs_with_score[0]
dummy = Node(id="asdasc", properties={})
target_node = Node(id="rfwewd", properties={})
dummy_rel = Relationship(source=dummy,
                    target=target_node,
                    type='label',
                    properties={}
                ) 
source_doc = Document(metadata={'source': 'NCCN_prostate_4.2024_Graph_12_33.json'}, page_content="Clinically localized prostate cancer (Any T, N0, M0 or Any T, NX, MX)")
graph_documents.append(GraphDocument(
        nodes=[dummy],
        relationships=[dummy_rel],
        source=source_doc,
    ))

# %%
graph.add_graph_documents([graph_documents[-1]])

# %%
graph.refresh_schema()

# %%
vector_data = vector_retriever.invoke("Clinically localized prostate cancer (Any T, N0, M0 or Any T, NX, MX)")

# %%
print(vector_data)

# %%
driver = GraphDatabase.driver(
        uri = NEO4J_URI,
        auth = (NEO4J_USERNAME,
                NEO4J_PASSWORD))

def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Function to execute the query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

# Call the function to create the index
try:
    create_index()
except:
    print("not created index")
    pass

# Close the driver connection
driver.close()

# %%
class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


entity_chain = llm.with_structured_output(Entities)

# %%
entity_chain.invoke("List all the ways in which bone imaging is performed.")

# %%
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result


# %%
vector_data = vector_retriever.invoke("http://nccn-guideline.org/nsclc/1")

# %%
print(graph_retriever("bone imaging"))

# %%
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# %%
graph_documents[0]

# %%


# %%



