import os
import json
import re
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from graphviz import Digraph

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = ""
api_key = os.getenv("OPENAI_API_KEY")

# Global variables for tracking
total_tokens = 0
total_cost = 0
api_calls = 0

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(texts: List[str]) -> FAISS:
    global total_tokens, total_cost, api_calls
    embeddings = OpenAIEmbeddings()
    
    with get_openai_callback() as cb:
        vector_store = FAISS.from_texts(texts, embeddings)
        total_tokens += cb.total_tokens
        total_cost += cb.total_cost
        api_calls += cb.successful_requests
    
    return vector_store

# def generate_decision_tree(vector_store: FAISS, model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
#     global total_tokens, total_cost, api_calls
    
#     llm = ChatOpenAI(model_name=model_name, temperature=0.7)
    
#     prompt = ChatPromptTemplate.from_template("""
#         Based on the following medical information, create a decision tree node for treatment and management of advanced prostate cancer. The node should include:

#         1. Preconditions that patient must meet before being eligible for a clinical decision
#         2. Clinical decision to be made
#         3. Follow-ups required

#         Medical Information:
#         {context}

#         Output the node in the following JSON format:
#         {{
#             "condition": "Condition or symptom to check",
#             "outcomes": [
#                 {{
#                     "result": "Outcome or next step",
#                     "next_node": "Unique identifier for the next node"
#                 }}
#             ],
#             "description": "Brief description with additional details"
#         }}
#         """)
    
#     chain = (
#         {"context": RunnablePassthrough()} 
#         | prompt 
#         | llm 
#         | StrOutputParser()
#     )
    
#     tree = {}
#     processed_nodes = set()
    
#     # Define sections based on the text content
#     sections = [
#         "introduction_and_purpose",  # Overview and justification
#         "biochemical_recurrence",     # Rising PSA without metastases
#         "metastatic_hormone_sensitive",# Treatment for mHSPC
#         "non_metastatic_crpc",        # nmCRPC prognosis and treatment
#         "metastatic_crpc",            # mCRPC prognosis and treatment
#         "performance_status",          # Importance of performance status
#         "clinical_trial_enrollment",   # Discussing clinical trials
#         "methodology"                  # Guideline development process
#     ]

#     # Initialize nodes_to_process with the sections
#     nodes_to_process = sections.copy()

#     while nodes_to_process:
#         current_node = nodes_to_process.pop(0)
#         if current_node in processed_nodes:
#             continue

#         context = vector_store.similarity_search(current_node, k=1)[0].page_content
        
#         with get_openai_callback() as cb:
#             result = chain.invoke(context)
#             total_tokens += cb.total_tokens
#             total_cost += cb.total_cost
#             api_calls += cb.successful_requests
        
#         try:
#             node_data = json.loads(result)
#             tree[current_node] = node_data
#             processed_nodes.add(current_node)
            
#             for outcome in node_data["outcomes"]:
#                 if outcome["next_node"] not in processed_nodes:
#                     nodes_to_process.append(outcome["next_node"])
#         except json.JSONDecodeError:
#             print(f"Error processing node: {current_node}. Invalid JSON.")
#             print(result)
#         except KeyError as e:
#             print(f"Error processing node: {current_node}. Missing key: {e}")
#             print(result)
#         except Exception as e:
#             print(f"Error processing node: {current_node}. Error: {e}")
#             print(result)

#     return tree



def generate_decision_tree(vector_store: FAISS, model_name: str = "gpt-4o") -> Dict[str, Any]:
    global total_tokens, total_cost, api_calls
    
    llm = ChatOpenAI(model_name=model_name, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("""
        Based on the following medical information, create a decision tree node for treatment and management of advanced prostate cancer. The node should include:

        1. Preconditions that patient must meet before being eligible for a clinical decision
        2. Clinical decision to be made
        3. Follow-ups required

        Medical Information:
        {context}

        Output ONLY the node in the following JSON format, with no additional text before or after:
        {{
            "condition": "Condition or symptom to check",
            "outcomes": [
                {{
                    "result": "Outcome or next step",
                    "next_node": "Unique identifier for the next node"
                }}
            ],
            "description": "Brief description with additional details"
        }}
        """)
    
    chain = (
        {"context": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    tree = {}
    processed_nodes = set()
    
    # Define sections based on the text content
    sections = [
        "introduction_and_purpose",  # Overview and justification
        "biochemical_recurrence",     # Rising PSA without metastases
        "metastatic_hormone_sensitive",# Treatment for mHSPC
        "non_metastatic_crpc",        # nmCRPC prognosis and treatment
        "metastatic_crpc",            # mCRPC prognosis and treatment
        "performance_status",          # Importance of performance status
        "clinical_trial_enrollment",   # Discussing clinical trials
        "methodology"                  # Guideline development process
    ]

    # Initialize nodes_to_process with the sections
    nodes_to_process = sections.copy()

    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        if current_node in processed_nodes:
            continue

        print(f"Processing node: {current_node}")  # Print the current node being processed

        context = vector_store.similarity_search(current_node, k=1)[0].page_content
        
        with get_openai_callback() as cb:
            result = chain.invoke(context)
            total_tokens += cb.total_tokens
            total_cost += cb.total_cost
            api_calls += cb.successful_requests
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                node_data = json.loads(json_str)
                tree[current_node] = node_data
                processed_nodes.add(current_node)
                print(f"Node added: {current_node}")
                
                for outcome in node_data["outcomes"]:
                    if outcome["next_node"] not in processed_nodes:
                        nodes_to_process.append(outcome["next_node"])
            else:
                raise ValueError("No JSON object found in the result")
        except json.JSONDecodeError as e:
            print(f"Error processing node: {current_node}. Invalid JSON. Error: {e}")
            print("--"*50)
            print(result)
            print("-X"*50)
        except KeyError as e:
            print(f"Error processing node: {current_node}. Missing key: {e}")
            print("--"*50)  
            print(result)
            print("-X"*50)
        except Exception as e:
            print(f"Error processing node: {current_node}. Error: {e}")
            print("--"*50)
            print(result)
            print("-X"*50)

    return tree
def create_dot_graph(tree: Dict[str, Any]) -> str:
    dot = Digraph(comment='Medical Decision Tree')
    dot.attr(rankdir='TB', size='8,8')

    for node_id, node_data in tree.items():
        label = f"{node_data['condition']}\\n\\n{node_data['description']}"
        dot.node(node_id, label)

        for outcome in node_data['outcomes']:
            dot.edge(node_id, outcome['next_node'], label=outcome['result'])

    return dot.source

def main(pdf_path: str, output_path: str, model_name: str = "gpt-4o"):
    global total_tokens, total_cost, api_calls
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    text_chunks = split_text(text)

    # Create vector store
    vector_store = create_vector_store(text_chunks)

    # Generate decision tree
    tree = generate_decision_tree(vector_store, model_name)

    # Create DOT graph
    dot_graph = create_dot_graph(tree)

    # Save DOT graph to file
    with open(output_path, 'w') as f:
        f.write(dot_graph)

    print(f"DOT graph saved to {output_path}")
    
    # Print usage statistics
    print(f"\nAPI Usage Statistics:")
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total cost: ${total_cost:.4f}")

if __name__ == "__main__":
    pdf_path = "/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/APC Unabridged FINAL 081023.pdf"
    output_path = "medical_decision_tree_2.dot"
    main(pdf_path, output_path, model_name="gpt-4o")