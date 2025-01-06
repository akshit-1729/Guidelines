# Explanation: This script runs a graph-related LLM workflow for data extraction.
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# LangChain & community imports
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

######################
# BEGIN TRIPLET EXTRACTION CODE
######################
def get_node_content(graph, node_id):
    for node in graph:
        if node["@id"] == node_id:
            return node.get("nccn:content", "not_available")
    return "not_available"

def get_label_content(graph, label_id):
    """Similar to get_node_content but specifically for labels."""
    for node in graph:
        if node["@id"] == label_id and node.get("@type") == "nccn:Labels":
            return node.get("nccn:content", "not_available")
    return "not_available"

def get_reference_content(graph, ref_id):
    """Similar to get_label_content but for references (footnotes)."""
    for node in graph:
        if node["@id"] == ref_id and node.get("@type") == "nccn:Footnote":
            return node.get("nccn:content", "not_available")
    return "not_available"

def extract_triplets_from_json(json_file):
    """
    Extracts triplets of (source, relation, target, empty, idd) 
    from the provided JSON-LD file.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    graph = data.get("@graph", [])
    triplets = []
    
    for node in graph:
        source = node.get("nccn:content", "")
        
        # Skip if source is empty and no parent
        if not source and not node.get("nccn:parent"):
            continue
            
        # If source is empty, get content from parent's prev nodes
        if not source:
            parent_id = node.get("nccn:parent")
            for parent_node in graph:
                if parent_node["@id"] == parent_id:
                    prev_ids = parent_node.get("nccn:prev", [])
                    source = " | ".join([get_node_content(graph, prev_id) for prev_id in prev_ids])
                    if not source:
                        source = "not_available"
                    break
        
        # Page key relationship
        if "nccn:page-key" in node:
            triplets.append((source, "page_key", node["nccn:page-key"], "", node.get("@id", "")))
            
        # Page number relationship
        if "nccn:page-no" in node:
            triplets.append((source, "page_no", str(node["nccn:page-no"]), "", node.get("@id", "")))
            
        # Label relationships
        if "nccn:labels" in node:
            for label_id in node["nccn:labels"]:
                label_content = get_label_content(graph, label_id)
                triplets.append((source, "label", label_content, "", node.get("@id", "")))
                
        # Reference relationships
        if "nccn:reference" in node:
            for ref_id in node["nccn:reference"]:
                ref_content = get_reference_content(graph, ref_id)
                triplets.append((source, "refers_to", ref_content, "", node.get("@id", "")))
                
        # Previous node relationships
        if "nccn:prev" in node:
            for prev_id in node["nccn:prev"]:
                prev_content = get_node_content(graph, prev_id)
                triplets.append((source, "previous_node", prev_content, "", node.get("@id", "")))
                
        # Next node relationships
        if "nccn:next" in node:
            next_ids = node["nccn:next"]
            for next_id in next_ids:
                next_content = get_node_content(graph, next_id)
                if next_content == "":  # If next node is empty
                    # Find contained nodes
                    for next_node in graph:
                        if next_node["@id"] == next_id:
                            if "nccn:contains" in next_node:
                                contained_contents = [get_node_content(graph, contained_id) 
                                                      for contained_id in next_node["nccn:contains"]]
                                for content in contained_contents:
                                    triplets.append((source, "next_contained_nodes", content, "", node.get("@id", "")))
                            break
                else:
                    triplets.append((source, "next_node", next_content, "", node.get("@id", "")))
    
    return triplets
######################
# END TRIPLET EXTRACTION CODE
######################


class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class GraphRAGWithLangchain:
    def __init__(
        self,
        openai_api_key,
        neo4j_uri,
        neo4j_username,
        neo4j_password,
        neo4j_database
    ):
        print("Initializing GraphRAGWithLangchain...")
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
            database=neo4j_database
        )
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )

        # LLM initialization
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",  # or "gpt-4o" if appropriate
            openai_api_key=openai_api_key
        )

        self.vector_index = self.setup_vector_index(neo4j_uri, neo4j_username, neo4j_password)
        self.entity_chain = self.llm.with_structured_output(Entities)
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)
        print("GraphRAGWithLangchain initialized.")

    def setup_vector_index(self, neo4j_uri, neo4j_username, neo4j_password):
        """Sets up the Neo4j vector index for semantic search on label `Node`."""
        print("Setting up vector index for label `Node`...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="nccn_vector",
            search_type="hybrid",
            node_label="Node",  # use "Node"
            text_node_properties=["nccn_content"],
            embedding_node_property="embedding",
        )
        print("Vector index setup complete.")
        return vector_index

    def close(self):
        print("Closing Neo4j driver...")
        self.driver.close()
        print("Neo4j driver closed.")

    def create_fulltext_index(self):
        """
        Creates a full-text index on `Node` nodes for faster keyword-based search.
        """
        print("Creating fulltext index on label `Node`...")
        with self.driver.session() as session:
            try:
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        CREATE FULLTEXT INDEX fulltext_node_id IF NOT EXISTS
                        FOR (n:Node) ON EACH [n.id, n.nccn_content]
                        """
                    )
                )
                print("Fulltext index on label `Node` created successfully.")
            except Exception as e:
                print(f"Error creating fulltext index: {e}")

    def populate_graph_from_triplets(self, triplets):
        """
        Inserts nodes and relationships into Neo4j from the extracted triplets.

        Each triplet is of the form:
          (source, relation, target, empty, idd)

        We'll store:
          - A 'Node' labeled node for the source with property 'nccn_content' and 'id' 
          - A 'Node' labeled node for the target with property 'nccn_content'
          - A relationship :TRIPLET_REL with a 'type' property for the 'relation'
        """
        print("Populating graph from extracted triplets...")

        with self.driver.session() as session:
            for (source, relation, target, _, idd) in triplets:
                # We store the relationship name as a property so we do not have dynamic relationship types.
                # If you want dynamic relationship types, you must build a dynamic Cypher string, 
                # but we keep it simpler here.
                query = """
                MERGE (s:Node {id: $idd})
                ON CREATE SET s.nccn_content = $source
                SET s.nccn_content = COALESCE(s.nccn_content, $source)
                MERGE (t:Node {nccn_content: $target})
                MERGE (s)-[r:TRIPLET_REL {relation_type: $relation}]->(t)
                """
                session.run(
                    query,
                    idd=idd or "unknown_id",
                    source=source,
                    target=target,
                    relation=relation
                )

        print("Finished populating graph from triplets.")

    def populate_graph_from_jsonld(self, json_file_path):
        """
        Populates the Neo4j graph using GraphDocument objects created from a JSON-LD file
        (the original approach). Then also uses the extracted triplets approach.
        """
        print(f"Populating graph from {json_file_path} using JSON-LD approach...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # First, do the standard JSON-LD loading approach:
        graph_documents = self.create_graph_documents_from_jsonld(data)
        self.graph.add_graph_documents(graph_documents)
        print(f"Populated graph with data from {json_file_path} via JSON-LD approach.")

        # Second, do the triplet approach:
        print("Now extracting triplets from JSON and populating them as well.")
        triplets = extract_triplets_from_json(json_file_path)
        print(f"Extracted {len(triplets)} triplets.")
        self.populate_graph_from_triplets(triplets)
        print("Populated graph with extracted triplets.")

    def create_graph_documents_from_jsonld(self, data):
        """
        Creates a list of GraphDocument objects from JSON-LD data.
        We label them as `Node` to match what's actually stored in the DB.
        """
        relationship_keys = [
            'nccn:labels',
            'nccn:next',
            'nccn:prev',
            'nccn:parent',
            'nccn:reference',
            'nccn:contains'
        ]
        nodes = {}
        graph_documents = []

        for item in data.get('@graph', []):
            node_id = item.get('@id')
            metadata = {
                "nccn:page-key": item.get("nccn:page-key"),
                "nccn:page-no": item.get("nccn:page-no")
            }
            # Simple log
            print(f"Processing item with @id: {node_id}")
            if node_id and node_id not in nodes:
                nodes[node_id] = Node(
                    id=node_id,
                    type="Node",  # labeled as "Node"
                    properties={
                        **metadata,
                        "nccn_content": item.get('nccn:content', '')
                    }
                )
            elif not node_id:
                print(f"Warning: Item with missing @id: {item}")

        # Build relationships + GraphDocuments
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
                        if isinstance(target, dict):
                            target_id = target.get('@id')
                        else:
                            target_id = str(target)

                        if target_id not in nodes:
                            # Create node if it doesn't exist
                            nodes[target_id] = Node(
                                id=target_id,
                                type="Node",  # keep consistent labeling
                                properties={
                                    "nccn_content": item.get('nccn:content', '')
                                }
                            )
                        target_node = nodes[target_id]

                        # Replace colons in relationship type
                        safe_rel_type = rel_type.replace(":", "_")
                        per_node_rels.append(
                            Relationship(
                                source=source_node,
                                target=target_node,
                                type=safe_rel_type,
                                properties={}
                            )
                        )

            source_doc = Document(
                metadata={
                    'source': 'NCCN_prostate_4.2024_Graph_12_33.json',
                    'node_id': source_id
                },
                page_content=item.get('nccn:content', '')
            )
            graph_documents.append(
                GraphDocument(
                    nodes=[source_node],
                    relationships=per_node_rels,
                    source=source_doc
                )
            )

        return graph_documents

    def vector_search(self, query, k=5):
        """Performs a similarity search using the Neo4j vector index."""
        print(f"Performing vector search for query: {query}")
        try:
            results = self.vector_index.similarity_search(query, k=k)
            print(f"Vector search results: {results}")
            return results
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []

    def extract_entities(self, text):
        """Extracts entities from a given text using the LLM."""
        print(f"Extracting entities from text: {text}")
        try:
            entities = self.entity_chain.invoke(text)
            print(f"Extracted entities: {entities}")
            return entities.names
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def fulltext_search(self, entity, limit=5):
        """
        Performs a full-text search on `Node` labeled nodes based on the extracted entity.
        We fully consume the query results inside the transaction to avoid scope errors.
        """
        print(f"Performing full-text search for entity: {entity}")
        escaped_entity = entity.replace("'", "\\'").replace('"', '\\"')
        search_query_str = f'nccn_content:"{escaped_entity}"~2 OR id:"{escaped_entity}"~2'
        print(f"Full-text search query string: {search_query_str}")

        try:
            with self.driver.session() as session:
                def do_query(tx):
                    query = """
                    CALL db.index.fulltext.queryNodes(
                        'fulltext_node_id',
                        $search_query,
                        {limit: $search_limit}
                    )
                    YIELD node, score
                    RETURN node, score
                    """
                    cursor = tx.run(
                        query,
                        search_query=search_query_str,
                        search_limit=limit
                    )
                    return list(cursor)  # fully consume cursor here

                results_list = session.read_transaction(do_query)

            nodes = []
            for record in results_list:
                node = record["node"]
                score = record["score"]
                nodes.append(node)
                print(f"Full-text search match: node={node}, score={score}")

            return nodes

        except Exception as e:
            print("Exception in fulltext_search()!")
            print(f"Full-text search query: {search_query_str}")
            print(f"Limit: {limit}")
            print(f"Error during full-text search: {e}")
            return []

    def get_neighbors(self, node_ids, limit=50):
        """Retrieves the neighbors of the given node IDs from the graph labeled `Node`."""
        print(f"Retrieving neighbors for node IDs: {node_ids}")
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    UNWIND $node_ids AS node_id
                    MATCH (node:Node) WHERE node.id = node_id
                    CALL {
                        WITH node
                        MATCH (node)-[r]->(neighbor:Node)
                        RETURN {source: node.id, type: type(r), target: neighbor.id} AS rel
                        UNION
                        WITH node
                        MATCH (node)<-[r]-(neighbor:Node)
                        RETURN {source: neighbor.id, type: type(r), target: node.id} AS rel
                    }
                    RETURN rel
                    LIMIT $limit
                    """,
                    node_ids=node_ids,
                    limit=limit
                )

                neighbors = []
                for record in result:
                    rel = record["rel"]
                    source = rel["source"]
                    type_ = rel["type"]
                    target = rel["target"]
                    if source and type_ and target:
                        neighbors.append(f"{source} -[{type_}]-> {target}")
                print(f"Retrieved neighbors: {neighbors}")
                return neighbors

        except Exception as e:
            print(f"Error retrieving neighbors: {e}")
            return []

    def retrieve_context(self, question, k=5, fulltext_limit=2, neighbor_limit=50):
        """
        Retrieves context using a combination of vector search, full-text search, and graph traversal.
        """
        print(f"Retrieving context for question: {question}")
        vector_results = self.vector_search(question, k=k)
        vector_node_ids = [
            doc.metadata['node_id']
            for doc in vector_results if 'node_id' in doc.metadata
        ]
        print(f"Vector search node IDs: {vector_node_ids}")

        entities = self.extract_entities(question)
        print(f"Extracted entities for full-text search: {entities}")

        fulltext_node_ids = []
        for entity in entities:
            fulltext_results = self.fulltext_search(entity, limit=fulltext_limit)
            fulltext_node_ids.extend([
                node['id'] for node in fulltext_results if 'id' in node
            ])
        print(f"Full-text search node IDs: {fulltext_node_ids}")

        all_node_ids = list(set(vector_node_ids + fulltext_node_ids))
        print(f"All relevant node IDs: {all_node_ids}")

        neighbor_info = self.get_neighbors(all_node_ids, limit=neighbor_limit)

        context = ""
        if vector_results:
            context += "Relevant documents from vector search:\n"
            for doc in vector_results:
                context += f"- {doc.page_content}\n"

        if neighbor_info:
            context += "\nRelevant relationships from graph:\n"
            for neighbor in neighbor_info:
                context += f"- {neighbor}\n"

        print(f"Retrieved context:\n{context}")
        return context

    def generate_cypher_query(self, question, context, schema_string):
        """
        Generates a Cypher query using the LLM. We instruct it to use (n:Node)
        and only known properties (n.id, n.nccn_content). Then we do some
        post-processing to handle any LLM hallucinations (like d_Document).
        """
        print(f"Generating Cypher query for question: {question}")
        if not schema_string or schema_string == '{"node_labels": {}, "relationship_types": []}':
            return "// The provided schema does not contain any node labels or relationship types, cannot generate Cypher query."

        prompt_template = """
        Task: Generate a Cypher query to answer the question based on the provided context and schema.
        - Use only (n:Node) or (m:Node) for nodes. 
        - DO NOT use variables like d_Document or e_Document.
        - The only textual properties are n.id, n.nccn_content. 
        - Relationship types must be replaced with underscores, e.g. (n)-[:NCCN_CONTAINS]->(m).
        - Return only a valid Cypher query, no extra text or explanation.

        Context:
        {context}

        Graph Schema:
        ```json
        {schema_string}
        ```

        Question: {question}
        Cypher query:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "schema_string", "question"]
        )

        cypher_chain = (
            {
                "context": lambda input: input["context"],
                "schema_string": lambda input: input["schema_string"],
                "question": lambda input: input["question"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            raw_query = cypher_chain.invoke({
                "context": context,
                "schema_string": schema_string,
                "question": question
            })

            # Remove triple backticks if present
            start_index = raw_query.find("```cypher")
            end_index = raw_query.find("```", start_index + 1)
            if start_index != -1 and end_index != -1:
                raw_query = raw_query[start_index + len("```cypher") : end_index].strip()

            # Relationship colons -> underscores
            raw_query = raw_query.replace(":", "_")

            # Minimal post-processing
            post_processed = (raw_query
                              .replace("d_Document", "n")
                              .replace("e_Document", "m")
                              .replace("d.title", "n.nccn_content")
                              .replace("e.title", "m.nccn_content")
                              .replace(":Document", ":Node")
                              .replace(":Entity", ":Node"))

            # Example: fix if the LLM tries 'WHERE n.title = "Bone Imaging"'
            if 'WHERE n.title = "Bone Imaging"' in post_processed:
                post_processed = post_processed.replace(
                    'WHERE n.title = "Bone Imaging"',
                    'WHERE n.nccn_content CONTAINS "bone imaging"'
                )

            print(f"Post-processed Cypher query:\n{post_processed}")
            return post_processed
        except Exception as e:
            print(f"Error generating Cypher query: {e}")
            return None

    def get_graph_schema(self):
        """Retrieves basic schema information from the graph."""
        print("Retrieving graph schema...")
        node_labels = set()
        relationship_types = set()
        try:
            with self.driver.session() as session:
                node_label_result = session.run("CALL db.labels()")
                for record in node_label_result:
                    node_labels.add(record["label"])

                relationship_type_result = session.run("CALL db.relationshipTypes()")
                for record in relationship_type_result:
                    relationship_types.add(record["relationshipType"])

            schema_info = {
                "node_labels": sorted(list(node_labels)),
                "relationship_types": sorted(list(relationship_types)),
            }
            print(f"Graph schema: {schema_info}")
            return schema_info
        except Exception as e:
            print(f"Error retrieving graph schema: {e}")
            return {"node_labels": [], "relationship_types": []}

    def answer_question_with_retrieval(self, question):
        """
        Answers a question using a combination of retrieval (context + schema) and LLM generation (Cypher + final answer).
        """
        print(f"Answering question: {question}")
        context = self.retrieve_context(question)
        schema_info = self.get_graph_schema()
        schema_string = json.dumps(schema_info, indent=2)

        cypher_query = self.generate_cypher_query(question, context, schema_string)
        if cypher_query and not cypher_query.startswith("//"):
            with self.driver.session() as session:
                try:
                    print(f"Executing Cypher query:\n{cypher_query}")
                    result = session.run(cypher_query)
                    graph_data = [record.data() for record in result]
                    print(f"Graph data from Cypher query: {graph_data}")
                except Exception as e:
                    print("Exception in answer_question_with_retrieval()!")
                    print(f"Cypher query: {cypher_query}")
                    print(f"Error executing Cypher query: {e}")
                    return "I couldn't execute the generated Cypher query."

            # Now we ask LLM to produce a final answer using the retrieved context and the query results
            prompt_template = """
            Task: Answer the question based on the provided context and graph data.

            Context:
            {context}

            Graph Data:
            {graph_data}

            Question: {question}
            Answer:
            """
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "graph_data", "question"]
            )

            answer_chain = (
                {
                    "context": lambda input: input["context"],
                    "graph_data": lambda input: input["graph_data"],
                    "question": lambda input: input["question"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            try:
                answer = answer_chain.invoke({
                    "context": context,
                    "graph_data": graph_data,
                    "question": question
                })
                print(f"Full Prompt:\n{prompt.format(context=context, graph_data=graph_data, question=question)}")
                print(f"Answer:\n{answer}")
                return answer
            except Exception as e:
                print(f"Error generating answer: {e}")
                return "I couldn't generate an answer at the moment."
        else:
            return "I couldn't generate a Cypher query for your question."

    def debug_graph_store(self, node_limit=30, rel_limit=30):
        """
        Prints information about up to `node_limit` nodes and `rel_limit` relationships
        in the Neo4j database to help debug what's actually stored.
        """
        print("=== DEBUG: Printing up to", node_limit, "nodes and", rel_limit, "relationships ===")
        with self.driver.session() as session:
            # Print node info
            nodes_query = f"""
            MATCH (n)
            RETURN DISTINCT
              labels(n) AS labels,
              properties(n) AS props
            LIMIT {node_limit}
            """
            node_records = session.run(nodes_query)
            print(f"--- Showing up to {node_limit} nodes ---")
            for i, record in enumerate(node_records, start=1):
                labels = record["labels"]
                props = record["props"]
                print(f"[Node {i}] Labels: {labels}, Properties: {props}")

            # Print relationship info
            rels_query = f"""
            MATCH ()-[r]->()
            RETURN DISTINCT
              type(r) AS rel_type,
              properties(r) AS rel_props,
              startNode(r).id AS start_id,
              endNode(r).id AS end_id
            LIMIT {rel_limit}
            """
            rel_records = session.run(rels_query)
            print(f"\n--- Showing up to {rel_limit} relationships ---")
            for i, record in enumerate(rel_records, start=1):
                rel_type = record["rel_type"]
                rel_props = record["rel_props"]
                start_id = record["start_id"]
                end_id = record["end_id"]
                print(f"[Rel {i}] Type: {rel_type}, Properties: {rel_props}, "
                      f"StartNodeID: {start_id}, EndNodeID: {end_id}")

        print("=== END of Graph Store Debug ===")

def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-key")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your-password")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    json_file_path = os.getenv("JSON_FILE_PATH")

    graph_rag_agent = GraphRAGWithLangchain(
        openai_api_key,
        neo4j_uri,
        neo4j_username,
        neo4j_password,
        neo4j_database
    )

    # Create the full-text index on `Node`
    graph_rag_agent.create_fulltext_index()

    # Populate the graph if JSON file path is provided
    if json_file_path:
        graph_rag_agent.populate_graph_from_jsonld(json_file_path)

        # Debug: print up to 30 nodes and relationships
        graph_rag_agent.debug_graph_store()
    else:
        print("JSON_FILE_PATH not set in environment. Skipping graph population.")

    while True:
        user_question = input("Ask a question (or type 'exit'): ")
        if user_question.lower() == 'exit':
            break
        answer = graph_rag_agent.answer_question_with_retrieval(user_question)
        print(answer)

    graph_rag_agent.close()

if __name__ == "__main__":
    main()
