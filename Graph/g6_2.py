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


triplets = [
    ("NCCN Guidelines", "version", "4.2024"),
    ("NCCN Guidelines", "topic", "Prostate Cancer"),
    ("Clinically localized prostate cancer", "defined as", "Any T, N0, M0 or Any T, NX, MX"),
    ("Clinically localized prostate cancer", "requires workup", "Perform physical exam"),
    ("Clinically localized prostate cancer", "requires workup", "Perform digital rectal exam (DRE)"),
    ("Digital rectal exam (DRE)", "purpose", "Confirm clinical stage"),
    ("Clinically localized prostate cancer", "requires workup", "Perform and/or collect prostate-specific antigen (PSA)"),
    ("Prostate-specific antigen (PSA)", "requires", "Calculate PSA density"),
    ("Clinically localized prostate cancer", "requires workup", "Obtain and review diagnostic prostate biopsies"),
    ("Clinically localized prostate cancer", "requires workup", "Estimate life expectancy"),
    ("Life expectancy estimation", "refers to", "Principles of Life Expectancy Estimation [PROS-A]"),
    ("Clinically localized prostate cancer", "requires workup", "Inquire about known high-risk germline mutations"),
    ("Clinically localized prostate cancer", "requires workup", "Inquire about family history"),
    ("Clinically localized prostate cancer", "requires workup", "Perform somatic and/or germline testing"),
    ("Somatic and/or germline testing", "condition", "As appropriate"),
    ("Clinically localized prostate cancer", "requires workup", "Assess quality-of-life measures"),
    ("Clinically localized prostate cancer", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease [PROS-2]"),
    ("Regional prostate cancer", "defined as", "Any T, N1, M0"),
    ("Regional prostate cancer", "requires workup", "Perform physical exam"),
    ("Regional prostate cancer", "requires workup", "Perform imaging for staging"),
    ("Regional prostate cancer", "requires workup", "Perform DRE to confirm clinical stage"),
    ("Regional prostate cancer", "requires workup", "Perform and/or collect PSA"),
    ("PSA", "calculation", "Calculate PSA doubling time (PSADT)"),
    ("Regional prostate cancer", "requires workup", "Estimate life expectancy"),
    ("Regional prostate cancer", "requires workup", "Inquire about known high-risk germline mutations"),
    ("Regional prostate cancer", "requires workup", "Inquire about family history"),
    ("Regional prostate cancer", "requires workup", "Perform somatic and/or germline testing"),
    ("Regional prostate cancer", "requires workup", "Assess quality-of-life measures"),
    ("Regional prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]"),
    ("Metastatic prostate cancer", "defined as", "Any T, Any N, M1"),
    ("Metastatic prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]"),
    ("Metastatic prostate cancer", "next step", "Systemic Therapy for M1 Castration-Sensitive Prostate Cancer (CSPC) [PROS-13]"),
    ("Bone imaging", "can be achieved by", "Conventional technetium-99m-MDP bone scan"),
    ("Bone imaging", "can be achieved by", "CT"),
    ("Bone imaging", "can be achieved by", "MRI"),
    ("Bone imaging", "can be achieved by", "PSMA-PET/CT"),
    ("Bone imaging", "can be achieved by", "PSMA-PET/MRI"),
    ("Bone imaging", "can be achieved by", "PET/CT"),
    ("Bone imaging", "can be achieved by", "PET/MRI with F-18 sodium fluoride"),
    ("Bone imaging", "can be achieved by", "C-11 choline"),
    ("Bone imaging", "can be achieved by", "F-18 fluciclovine"),
    ("Equivocal results", "require", "Soft tissue imaging of the pelvis"),
    ("Equivocal results", "require", "Abdomen and chest imaging"),
    ("Multiparametric MRI (mpMRI)", "preferred over", "CT for pelvic staging"),
    ("PSMA-PET/CT", "can be considered for", "Bone and soft tissue (full body) imaging"),
    ("PSMA-PET/MRI", "can be considered for", "Bone and soft tissue (full body) imaging"),
    ("PSMA-PET tracers", "have", "Increased sensitivity and specificity"),
    ("PSMA-PET tracers", "compared to", "Conventional imaging"),
    ("Conventional imaging", "includes", "CT"),
    ("Conventional imaging", "includes", "Bone scan"),
    ("PSMA-PET", "use", "Not a necessary prerequisite"),
    ("PSMA-PET/CT", "can serve as", "Equally effective frontline imaging tool"),
    ("PSMA-PET/MRI", "can serve as", "Equally effective frontline imaging tool"),
    ("Initial Risk Stratification", "for", "Clinically Localized Disease"),
    ("Very low risk group", "has all of", "cT1c"),
    ("Very low risk group", "has all of", "Grade Group 1"),
    ("Very low risk group", "has all of", "PSA <10 ng/mL"),
    ("Very low risk group", "has all of", "<3 prostate biopsy fragments/cores positive"),
    ("Very low risk group", "has all of", "≤50% cancer in each fragment/core"),
    ("Very low risk group", "has all of", "PSA density <0.15 ng/mL/g"),
    ("Low risk group", "has all of", "cT1–cT2a"),
    ("Low risk group", "has all of", "Grade Group 1"),
    ("Low risk group", "has all of", "PSA <10 ng/mL"),
    ("Low risk group", "does not qualify for", "Very low risk"),
    ("Intermediate risk group", "has", "Favorable intermediate"),
    ("Intermediate risk group", "has", "Unfavorable intermediate"),
    ("Favorable intermediate", "has all of", "1 IRF"),
    ("Favorable intermediate", "has all of", "Grade Group 1 or 2"),
    ("Favorable intermediate", "has all of", "<50% biopsy cores positive"),
    ("Unfavorable intermediate", "has", "2 or 3 IRFs"),
    ("Unfavorable intermediate", "has", "Grade Group 3"),
    ("Unfavorable intermediate", "has", "≥ 50% biopsy cores positive"),
    ("High risk group", "has", "cT3a OR"),
    ("High risk group", "has", "Grade Group 4 or Grade Group 5 OR"),
    ("High risk group", "has", "PSA >20 ng/mL"),
    ("Very high risk group", "has at least one of", "cT3b–cT4"),
    ("Very high risk group", "has at least one of", "Primary Gleason pattern 5"),
    ("Very high risk group", "has at least one of", "2 or 3 high-risk features"),
    ("Very high risk group", "has at least one of", ">4 cores with Grade Group 4 or 5"),
    ("Very low risk group", "additional evaluation", "Confirmatory testing"),
    ("Very low risk group", "initial therapy", "PROS-3"),
    ("Low risk group", "additional evaluation", "Confirmatory testing"),
    ("Low risk group", "initial therapy", "PROS-4"),
    ("Favorable intermediate risk group", "additional evaluation", "Confirmatory testing"),
    ("Favorable intermediate risk group", "initial therapy", "PROS-5"),
    ("Unfavorable intermediate risk group", "additional evaluation", "Bone and soft tissue imaging"),
    ("Unfavorable intermediate risk group", "initial therapy", "PROS-6"),
    ("High risk group", "additional evaluation", "Bone and soft tissue imaging"),
    ("High risk group", "initial therapy", "PROS-7"),
    ("Very high risk group", "additional evaluation", "Bone and soft tissue imaging"),
    ("Very high risk group", "initial therapy", "PROS-7"),
    ("Confirmatory testing", "purpose", "Assess appropriateness of active surveillance"),
    ("Bone and soft tissue imaging", "condition", "If regional or distant metastases are found"),
    ("Bone and soft tissue imaging", "next step if metastases found", "See PROS-8 or PROS-13"),
    ("Very-low-risk group", "expected patient survival ≥10 y", "Active surveillance"),
    ("Very-low-risk group", "expected patient survival <10 y", "Observation"),
    ("Active surveillance", "refers to", "See Active Surveillance Program (PROS-F 2 of 5)"),
    ("Progressive disease", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease (PROS-2)"),
    ("Observation", "next step", "See Monitoring (PROS-9)"),
    ("Footnote a", "refers to", "See NCCN Guidelines for Older Adult Oncology"),
    ("Footnote b", "refers to", "NCCN Guidelines for Prostate Cancer Early Detection"),
    ("Footnote c", "refers to", "Principles of Bone Health in Prostate Cancer (PROS-K)"),
    ("Footnote d", "refers to", "Principles of Genetics and Molecular/Biomarker Analysis (PROS-C)"),
    ("Footnote e", "refers to", "Principles of Quality-of-Life and Shared Decision-Making (PROS-D)"),
    ("Footnote f", "refers to", "Principles of Imaging (PROS-E)"),
    ("Footnote g", "refers to", "Bone imaging"),
    ("Footnote h", "refers to", "PSMA-PET tracers"),
    ("Footnote i", "refers to", "Principles of Imaging (PROS-E)"),
    ("Footnote j", "refers to", "For patients who are asymptomatic"),
    ("Footnote k", "refers to", "An ultrasound- or MRI- or DRE-targeted lesion"),
    ("Footnote l", "refers to", "Percentage of positive cores in the intermediate-risk group"),
    ("Footnote m", "refers to", "Bone imaging for symptomatic patients"),
    ("Footnote n", "refers to", "Expected Patient Survival"),
    ("Footnote o", "refers to", "Expected Patient Survival ≥10 y"),
    ("Footnote p", "refers to", "Active surveillance"),
    ("Footnote q", "refers to", "See Active Surveillance Program"),
    ("Footnote r", "refers to", "Observation"),
    ("Footnote s", "refers to", "Progressive disease"),
]


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
# print(kg_index.docstore.)
# print(kg_index.docstore.docs['15b3c807-94d5-43f2-82d9-ff03ecc8dfb9'].relationships.keys())




# %%
documents


# %%


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

# 5.1 text-to-GraphQuery
# from llama_index.core.query_engine import KnowledgeGraphQueryEngine

# nl2kg_query_engine = KnowledgeGraphQueryEngine(
#     storage_context=kg_index.storage_context,
#     llm=llm,
#     verbose=True,
# )

# %%


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

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

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

# %%
# 6. Query with all the Engines

query = "At what times are we referencing the footnote a? a footnote is referenced when it is written in curly braces."

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

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
# 6. Query with all the Engines

query = "when do we Perform imaging for staging?"

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

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
# 6. Query with all the Engines

query = "How does the treatment for metastatic prostate cancer (M1) differ in its use of androgen deprivation therapy (ADT) compared to castration-sensitive prostate cancer (CSPC)?"

# token_counter.reset_counts()
# print("Text-to-GraphQuery result:")
# response_nl2kg = nl2kg_query_engine.query(query)
# print(response_nl2kg)
# print_token_usage()

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



