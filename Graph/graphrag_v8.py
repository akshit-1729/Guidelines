import networkx as nx
from graspologic.partition import hierarchical_leiden
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os
import matplotlib.pyplot as plt 
import re
import colorsys

class Document:
    def __init__(self, text: str, doc_id: str = None):
        self.text = text
        self.doc_id = doc_id

class GraphRAGStore:
    def __init__(self):
        self.graph = nx.Graph()
        self.communities = {}
        self.max_cluster_size = 5
        self.documents = {}  # New attribute to store documents

    def add_document(self, document: Document):
        """Add a document to the store."""
        if document.doc_id is None:
            document.doc_id = str(len(self.documents))
        self.documents[document.doc_id] = document

    def add_triplet(self, subject: str, predicate: str, object: str, description: str, doc_id: str = None):
        """Add a single triplet to the graph."""
        self.graph.add_edge(subject, object, relationship=predicate, description=description, doc_id=doc_id)

    def add_triplets(self, triplets: List[tuple]):
        """Add multiple triplets to the graph."""
        for triplet in triplets:
            if len(triplet) == 5:
                subject, predicate, object, description, doc_id = triplet
            else:
                subject, predicate, object, description = triplet
                doc_id = None
            self.add_triplet(subject, predicate, object, description, doc_id)

    def build_communities(self):
        """Builds communities from the graph."""
        community_hierarchical_clusters = hierarchical_leiden(
            self.graph, max_cluster_size=self.max_cluster_size
        )
        self.communities = self._collect_community_info(
            self.graph, community_hierarchical_clusters
        )

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        if 'doc_id' in edge_data:
                            detail += f" (Document ID: {edge_data['doc_id']})"
                        community_info[cluster_id].append(detail)
        return community_info

    def get_communities(self) -> Dict[Any, List[str]]:
        """Returns the communities, building them if not already done."""
        if not self.communities:
            self.build_communities()
        return self.communities

    def print_graph_info(self):
        """Print basic information about the graph."""
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Number of communities: {len(self.communities)}")
        print(f"Number of documents: {len(self.documents)}")

    def print_communities(self):
        """Print the contents of each community."""
        for community_id, details in self.communities.items():
            print(f"\nCommunity {community_id}:")
            for detail in details:
                print(f"  {detail}")

    def save(self, filename='graph_store.pkl'):
        """Save the GraphRAGStore object to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename='graph_store.pkl'):
        """Load a GraphRAGStore object from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, object: Optional[str] = None, 
              fuzzy_match: bool = True, case_sensitive: bool = False) -> List[Tuple]:
        """
        Query the graph for matching triplets with enhanced flexibility.
        """
        results = []

        def match_string(query: Optional[str], target: str) -> bool:
            if query is None:
                return True
            if not case_sensitive:
                query = query.lower()
                target = target.lower()
            if fuzzy_match:
                return query in target
            else:
                return query == target

        for s, o, data in self.graph.edges(data=True):
            p = data['relationship']
            d = data['description']
            doc_id = data.get('doc_id')

            if match_string(subject, s) and match_string(predicate, p) and match_string(object, o):
                result = (s, p, o, d)
                if doc_id:
                    result += (doc_id,)
                results.append(result)

        return results

    def advanced_query(self, query: str) -> List[Tuple]:
        """
        Perform an advanced query using natural language processing techniques.
        """
        # Tokenize the query
        tokens = re.findall(r'\w+', query.lower())
        
        # Remove stop words (you can expand this list)
        stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by', 'is', 'are', 'was', 'were'])
        tokens = [token for token in tokens if token not in stop_words]
        
        # Perform the query for each token
        all_results = []
        for token in tokens:
            results = self.query(subject=token, fuzzy_match=True, case_sensitive=False)
            results += self.query(predicate=token, fuzzy_match=True, case_sensitive=False)
            results += self.query(object=token, fuzzy_match=True, case_sensitive=False)
            all_results.extend(results)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in all_results:
            result_key = (result[0], result[1], result[2])  # Use (subject, predicate, object) as the key
            if result_key not in seen:
                seen.add(result_key)
                unique_results.append(result)
        
        # Sort results by relevance (number of matching tokens)
        sorted_results = sorted(unique_results, key=lambda x: sum(token in ' '.join(x).lower() for token in tokens), reverse=True)
        
        # Return top 10 most relevant results
        return sorted_results[:10]

    def extract_subgraph(self, query_results):
        """Extract a subgraph based on query results."""
        subgraph = nx.Graph()
        for result in query_results:
            s, p, o, d = result[:4]
            edge_data = {'relationship': p, 'description': d}
            if len(result) == 5:
                edge_data['doc_id'] = result[4]
            subgraph.add_edge(s, o, **edge_data)
        return subgraph
    
    def visualize_graph(self, subgraph=None, figsize=(12, 8)):
        """
        Visualize the graph or a subgraph using networkx and matplotlib.
        
        Args:
            subgraph (nx.Graph, optional): A subgraph to visualize. If None, visualize the entire graph.
            figsize (tuple, optional): Figure size. Defaults to (12, 8).
        """
        plt.figure(figsize=figsize)
        graph_to_draw = subgraph if subgraph is not None else self.graph
        pos = nx.spring_layout(graph_to_draw)
        nx.draw(graph_to_draw, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(graph_to_draw, 'relationship')
        nx.draw_networkx_edge_labels(graph_to_draw, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Graph Visualization" if subgraph is None else "Subgraph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def subgraph_to_dot(self, subgraph):
        """
        Convert a subgraph to DOT format for visualization with Graphviz.

        Args:
            subgraph (nx.Graph): The subgraph to convert.

        Returns:
            str: The DOT representation of the subgraph.
        """
        dot_string = "digraph G {\n"
        dot_string += "  rankdir=LR;\n"  # Left to right layout
        dot_string += "  node [style=filled];\n"  # Use filled style for nodes

        # Generate unique colors for relationships
        relationships = set(nx.get_edge_attributes(subgraph, 'relationship').values())
        colors = self._generate_colors(len(relationships))
        color_map = dict(zip(relationships, colors))

        # Add nodes
        subjects = set()
        objects = set()
        for s, o, _ in subgraph.edges(data=True):
            subjects.add(s)
            objects.add(o)

        for node in subgraph.nodes():
            if node in subjects and node in objects:
                color = "#FFA500"  # Orange for nodes that are both subject and object
            elif node in subjects:
                color = "#ADD8E6"  # Light blue for subjects
            else:
                color = "#90EE90"  # Light green for objects
            dot_string += f'  "{node}" [label="{node}", fillcolor="{color}"];\n'

        # Add edges
        for edge in subgraph.edges(data=True):
            source, target, data = edge
            relationship = data.get('relationship', '')
            description = data.get('description', '')
            doc_id = data.get('doc_id', 'N/A')
            edge_label = f"{relationship}\\n{description}\\n(Doc: {doc_id})"
            edge_color = color_map[relationship]
            dot_string += f'  "{source}" -> "{target}" [label="{edge_label}", color="{edge_color}"];\n'

        # Add legend
        dot_string += "  subgraph cluster_legend {\n"
        dot_string += "    label = \"Legend\";\n"
        dot_string += "    node [shape=box];\n"
        dot_string += '    "Subject" [fillcolor="#ADD8E6"];\n'
        dot_string += '    "Object" [fillcolor="#90EE90"];\n'
        dot_string += '    "Both" [fillcolor="#FFA500"];\n'
        for relationship, color in color_map.items():
            dot_string += f'    "{relationship}" [shape=plaintext, fillcolor="white"];\n'
            dot_string += f'    "dummy_{relationship}" [shape=point, style=invis];\n'
            dot_string += f'    "dummy_{relationship}" -> "{relationship}" [color="{color}"];\n'
        dot_string += "  }\n"

        dot_string += "}"
        return dot_string

    def _generate_colors(self, n):
        """Generate n distinct colors."""
        HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
        return ['#%02x%02x%02x' % tuple(int(x * 255) for x in colorsys.hsv_to_rgb(*hsv)) for hsv in HSV_tuples]

def get_or_create_graph_store(filename='graph_store.pkl'):
    """Get an existing GraphRAGStore or create a new one if it doesn't exist."""
    return GraphRAGStore()
    if os.path.exists(filename):
        return GraphRAGStore.load(filename)
    else:
        return GraphRAGStore()
    
    

# Example usage
if __name__ == "__main__":
    # Get or create GraphRAGStore
    graph_store = get_or_create_graph_store()

    # If the graph is empty, add sample documents and triplets
    if graph_store.graph.number_of_edges() == 0:
        # Add sample documents
        documents = [
            Document("Alice and Bob work together on software projects.", "doc1"),
            Document("Bob manages Charlie and receives weekly reports.", "doc2"),
            Document("Alice mentors David in programming skills.", "doc3"),
            Document("Eve leads Project X with Alice and Bob's involvement.", "doc4"),
        ]
        for doc in documents:
            graph_store.add_document(doc)

#         triplets = [
#     ("NCCN Guidelines", "version", "4.2024", "describes", "doc1"),
#     ("NCCN Guidelines", "topic", "Prostate Cancer", "covers", "doc1"),
#     ("Clinically localized prostate cancer", "defined as", "Any T, N0, M0 or Any T, NX, MX", "is defined as", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Perform physical exam", "requires", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Perform digital rectal exam (DRE)", "requires", "doc1"),
#     ("Digital rectal exam (DRE)", "purpose", "Confirm clinical stage", "is used to", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Perform and/or collect prostate-specific antigen (PSA)", "requires", "doc1"),
#     ("Prostate-specific antigen (PSA)", "requires", "Calculate PSA density", "involves", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Obtain and review diagnostic prostate biopsies", "requires", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Estimate life expectancy", "requires", "doc1"),
#     ("Life expectancy estimation", "refers to", "Principles of Life Expectancy Estimation [PROS-A]", "references", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Inquire about known high-risk germline mutations", "requires", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Inquire about family history", "requires", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Perform somatic and/or germline testing", "requires", "doc1"),
#     ("Somatic and/or germline testing", "condition", "As appropriate", "is performed", "doc1"),
#     ("Clinically localized prostate cancer", "requires workup", "Assess quality-of-life measures", "requires", "doc1"),
#     ("Clinically localized prostate cancer", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease [PROS-2]", "proceeds to", "doc1"),
#     ("Regional prostate cancer", "defined as", "Any T, N1, M0", "is defined as", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Perform physical exam", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Perform imaging for staging", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Perform DRE to confirm clinical stage", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Perform and/or collect PSA", "requires", "doc1"),
#     ("PSA", "calculation", "Calculate PSA doubling time (PSADT)", "involves", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Estimate life expectancy", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Inquire about known high-risk germline mutations", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Inquire about family history", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Perform somatic and/or germline testing", "requires", "doc1"),
#     ("Regional prostate cancer", "requires workup", "Assess quality-of-life measures", "requires", "doc1"),
#     ("Regional prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]", "proceeds to", "doc1"),
#     ("Metastatic prostate cancer", "defined as", "Any T, Any N, M1", "is defined as", "doc1"),
#     ("Metastatic prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]", "proceeds to", "doc1"),
#     ("Metastatic prostate cancer", "next step", "Systemic Therapy for M1 Castration-Sensitive Prostate Cancer (CSPC) [PROS-13]", "proceeds to", "doc1"),
#     ("Bone imaging", "can be achieved by", "Conventional technetium-99m-MDP bone scan", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "CT", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "MRI", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "PSMA-PET/CT", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "PSMA-PET/MRI", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "PET/CT", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "PET/MRI with F-18 sodium fluoride", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "C-11 choline", "uses", "doc1"),
#     ("Bone imaging", "can be achieved by", "F-18 fluciclovine", "uses", "doc1"),
#     ("Equivocal results", "require", "Soft tissue imaging of the pelvis", "necessitates", "doc1"),
#     ("Equivocal results", "require", "Abdomen and chest imaging", "necessitates", "doc1"),
#     ("Multiparametric MRI (mpMRI)", "preferred over", "CT for pelvic staging", "is preferred to", "doc1"),
#     ("PSMA-PET/CT", "can be considered for", "Bone and soft tissue (full body) imaging", "is used for", "doc1"),
#     ("PSMA-PET/MRI", "can be considered for", "Bone and soft tissue (full body) imaging", "is used for", "doc1"),
#     ("PSMA-PET tracers", "have", "Increased sensitivity and specificity", "possess", "doc1"),
#     ("PSMA-PET tracers", "compared to", "Conventional imaging", "are compared with", "doc1"),
#     ("Conventional imaging", "includes", "CT", "encompasses", "doc1"),
#     ("Conventional imaging", "includes", "Bone scan", "encompasses", "doc1"),
#     ("PSMA-PET", "use", "Not a necessary prerequisite", "is considered", "doc1"),
#     ("PSMA-PET/CT", "can serve as", "Equally effective frontline imaging tool", "functions as", "doc1"),
#     ("PSMA-PET/MRI", "can serve as", "Equally effective frontline imaging tool", "functions as", "doc1"),
#     ("Initial Risk Stratification", "for", "Clinically Localized Disease", "is used for", "doc1"),
#     ("Very low risk group", "has all of", "cT1c", "includes", "doc1"),
#     ("Very low risk group", "has all of", "Grade Group 1", "includes", "doc1"),
#     ("Very low risk group", "has all of", "PSA <10 ng/mL", "includes", "doc1"),
#     ("Very low risk group", "has all of", "<3 prostate biopsy fragments/cores positive", "includes", "doc1"),
#     ("Very low risk group", "has all of", "≤50% cancer in each fragment/core", "includes", "doc1"),
#     ("Very low risk group", "has all of", "PSA density <0.15 ng/mL/g", "includes", "doc1"),
#     ("Low risk group", "has all of", "cT1–cT2a", "includes", "doc1"),
#     ("Low risk group", "has all of", "Grade Group 1", "includes", "doc1"),
#     ("Low risk group", "has all of", "PSA <10 ng/mL", "includes", "doc1"),
#     ("Low risk group", "does not qualify for", "Very low risk", "is distinct from", "doc1"),
#     ("Intermediate risk group", "has", "Favorable intermediate", "includes", "doc1"),
#     ("Intermediate risk group", "has", "Unfavorable intermediate", "includes", "doc1"),
#     ("Favorable intermediate", "has all of", "1 IRF", "includes", "doc1"),
#     ("Favorable intermediate", "has all of", "Grade Group 1 or 2", "includes", "doc1"),
#     ("Favorable intermediate", "has all of", "<50% biopsy cores positive", "includes", "doc1"),
#     ("Unfavorable intermediate", "has", "2 or 3 IRFs", "includes", "doc1"),
#     ("Unfavorable intermediate", "has", "Grade Group 3", "includes", "doc1"),
#     ("Unfavorable intermediate", "has", "≥ 50% biopsy cores positive", "includes", "doc1"),
#     ("High risk group", "has", "cT3a OR", "includes", "doc1"),
#     ("High risk group", "has", "Grade Group 4 or Grade Group 5 OR", "includes", "doc1"),
#     ("High risk group", "has", "PSA >20 ng/mL", "includes", "doc1"),
#     ("Very high risk group", "has at least one of", "cT3b–cT4", "includes", "doc1"),
#     ("Very high risk group", "has at least one of", "Primary Gleason pattern 5", "includes", "doc1"),
#     ("Very high risk group", "has at least one of", "2 or 3 high-risk features", "includes", "doc1"),
#     ("Very high risk group", "has at least one of", ">4 cores with Grade Group 4 or 5", "includes", "doc1"),
#     ("Very low risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
#     ("Very low risk group", "initial therapy", "PROS-3", "refers to", "doc1"),
#     ("Low risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
#     ("Low risk group", "initial therapy", "PROS-4", "refers to", "doc1"),
#     ("Favorable intermediate risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
#     ("Favorable intermediate risk group", "initial therapy", "PROS-5", "refers to", "doc1"),
#     ("Unfavorable intermediate risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
#     ("Unfavorable intermediate risk group", "initial therapy", "PROS-6", "refers to", "doc1"),
#     ("High risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
#     ("High risk group", "initial therapy", "PROS-7", "refers to", "doc1"),
#     ("Very high risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
#     ("Very high risk group", "initial therapy", "PROS-7", "refers to", "doc1"),
#     ("Confirmatory testing", "purpose", "Assess appropriateness of active surveillance", "is used to", "doc1"),
#     ("Bone and soft tissue imaging", "condition", "If regional or distant metastases are found", "is performed", "doc1"),
#     ("Bone and soft tissue imaging", "next step if metastases found", "See PROS-8 or PROS-13", "leads to", "doc1"),
#     ("Very-low-risk group", "expected patient survival ≥10 y", "Active surveillance", "recommends", "doc1"),
#     ("Very-low-risk group", "expected patient survival <10 y", "Observation", "recommends", "doc1"),
#     ("Active surveillance", "refers to", "See Active Surveillance Program (PROS-F 2 of 5)", "is detailed in", "doc1"),
#     ("Progressive disease", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease (PROS-2)", "proceeds to", "doc1"),
#     ("Observation", "next step", "See Monitoring (PROS-9)", "proceeds to", "doc1"),
#     ("Footnote a", "refers to", "See NCCN Guidelines for Older Adult Oncology", "references", "doc1"),
#     ("Footnote b", "refers to", "NCCN Guidelines for Prostate Cancer Early Detection", "references", "doc1"),
#     ("Footnote c", "refers to", "Principles of Bone Health in Prostate Cancer (PROS-K)", "references", "doc1"),
#     ("Footnote d", "refers to", "Principles of Genetics and Molecular/Biomarker Analysis (PROS-C)", "references", "doc1"),
#     ("Footnote e", "refers to", "Principles of Quality-of-Life and Shared Decision-Making (PROS-D)", "references", "doc1"),
#     ("Footnote f", "refers to", "Principles of Imaging (PROS-E)", "references", "doc1"),
#     ("Footnote g", "refers to", "Bone imaging", "references", "doc1"),
#     ("Footnote h", "refers to", "PSMA-PET tracers", "references", "doc1"),
#     ("Footnote i", "refers to", "Principles of Imaging (PROS-E)", "references", "doc1"),
#     ("Footnote j", "refers to", "For patients who are asymptomatic", "applies to", "doc1"),
#     ("Footnote k", "refers to", "An ultrasound- or MRI- or DRE-targeted lesion", "describes", "doc1"),
#     ("Footnote l", "refers to", "Percentage of positive cores in the intermediate-risk group", "describes", "doc1"),
#     ("Footnote m", "refers to", "Bone imaging for symptomatic patients", "applies to", "doc1"),
#     ("Footnote n", "refers to", "Expected Patient Survival", "describes", "doc1"),
#     ("Footnote o", "refers to", "Expected Patient Survival ≥10 y", "describes", "doc1"),
#     ("Footnote p", "refers to", "Active surveillance", "describes", "doc1"),
#     ("Footnote q", "refers to", "See Active Surveillance Program", "references", "doc1"),
#     ("Footnote r", "refers to", "Observation", "describes", "doc1"),
#     ("Footnote s", "refers to", "Progressive disease", "describes", "doc1"),
# ]
        
    #     triplets = [
    # ("NCCN Guidelines", "version", "4.2024"),
    # ("NCCN Guidelines", "topic", "Prostate Cancer"),
    # ("Clinically localized prostate cancer", "defined as", "Any T, N0, M0 or Any T, NX, MX"),
    # ("Clinically localized prostate cancer", "requires workup", "Perform physical exam"),
    # ("Clinically localized prostate cancer", "requires workup", "Perform digital rectal exam (DRE)"),
    # ("Digital rectal exam (DRE)", "purpose", "Confirm clinical stage"),
    # ("Clinically localized prostate cancer", "requires workup", "Perform and/or collect prostate-specific antigen (PSA)"),
    # ("Prostate-specific antigen (PSA)", "requires", "Calculate PSA density"),
    # ("Clinically localized prostate cancer", "requires workup", "Obtain and review diagnostic prostate biopsies"),
    # ("Clinically localized prostate cancer", "requires workup", "Estimate life expectancy"),
    # ("Life expectancy estimation", "refers to", "Principles of Life Expectancy Estimation [PROS-A]"),
    # ("Clinically localized prostate cancer", "requires workup", "Inquire about known high-risk germline mutations"),
    # ("Clinically localized prostate cancer", "requires workup", "Inquire about family history"),
    # ("Clinically localized prostate cancer", "requires workup", "Perform somatic and/or germline testing"),
    # ("Somatic and/or germline testing", "condition", "As appropriate"),
    # ("Clinically localized prostate cancer", "requires workup", "Assess quality-of-life measures"),
    # ("Clinically localized prostate cancer", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease [PROS-2]"),
    # ("Regional prostate cancer", "defined as", "Any T, N1, M0"),
    # ("Regional prostate cancer", "requires workup", "Perform physical exam"),
    # ("Regional prostate cancer", "requires workup", "Perform imaging for staging"),
    # ("Regional prostate cancer", "requires workup", "Perform DRE to confirm clinical stage"),
    # ("Regional prostate cancer", "requires workup", "Perform and/or collect PSA"),
    # ("PSA", "calculation", "Calculate PSA doubling time (PSADT)"),
    # ("Regional prostate cancer", "requires workup", "Estimate life expectancy"),
    # ("Regional prostate cancer", "requires workup", "Inquire about known high-risk germline mutations"),
    # ("Regional prostate cancer", "requires workup", "Inquire about family history"),
    # ("Regional prostate cancer", "requires workup", "Perform somatic and/or germline testing"),
    # ("Regional prostate cancer", "requires workup", "Assess quality-of-life measures"),
    # ("Regional prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]"),
    # ("Metastatic prostate cancer", "defined as", "Any T, Any N, M1"),
    # ("Metastatic prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]"),
    # ("Metastatic prostate cancer", "next step", "Systemic Therapy for M1 Castration-Sensitive Prostate Cancer (CSPC) [PROS-13]"),
    # ("Bone imaging", "can be achieved by", "Conventional technetium-99m-MDP bone scan"),
    # ("Bone imaging", "can be achieved by", "CT"),
    # ("Bone imaging", "can be achieved by", "MRI"),
    # ("Bone imaging", "can be achieved by", "PSMA-PET/CT"),
    # ("Bone imaging", "can be achieved by", "PSMA-PET/MRI"),
    # ("Bone imaging", "can be achieved by", "PET/CT"),
    # ("Bone imaging", "can be achieved by", "PET/MRI with F-18 sodium fluoride"),
    # ("Bone imaging", "can be achieved by", "C-11 choline"),
    # ("Bone imaging", "can be achieved by", "F-18 fluciclovine"),
    # ("Equivocal results", "require", "Soft tissue imaging of the pelvis"),
    # ("Equivocal results", "require", "Abdomen and chest imaging"),
    # ("Multiparametric MRI (mpMRI)", "preferred over", "CT for pelvic staging"),
    # ("PSMA-PET/CT", "can be considered for", "Bone and soft tissue (full body) imaging"),
    # ("PSMA-PET/MRI", "can be considered for", "Bone and soft tissue (full body) imaging"),
    # ("PSMA-PET tracers", "have", "Increased sensitivity and specificity"),
    # ("PSMA-PET tracers", "compared to", "Conventional imaging"),
    # ("Conventional imaging", "includes", "CT"),
    # ("Conventional imaging", "includes", "Bone scan"),
    # ("PSMA-PET", "use", "Not a necessary prerequisite"),
    # ("PSMA-PET/CT", "can serve as", "Equally effective frontline imaging tool"),
    # ("PSMA-PET/MRI", "can serve as", "Equally effective frontline imaging tool"),
    # ("Initial Risk Stratification", "for", "Clinically Localized Disease"),
    # ("Very low risk group", "has all of", "cT1c"),
    # ("Very low risk group", "has all of", "Grade Group 1"),
    # ("Very low risk group", "has all of", "PSA <10 ng/mL"),
    # ("Very low risk group", "has all of", "<3 prostate biopsy fragments/cores positive"),
    # ("Very low risk group", "has all of", "≤50% cancer in each fragment/core"),
    # ("Very low risk group", "has all of", "PSA density <0.15 ng/mL/g"),
    # ("Low risk group", "has all of", "cT1–cT2a"),
    # ("Low risk group", "has all of", "Grade Group 1"),
    # ("Low risk group", "has all of", "PSA <10 ng/mL"),
    # ("Low risk group", "does not qualify for", "Very low risk"),
    # ("Intermediate risk group", "has", "Favorable intermediate"),
    # ("Intermediate risk group", "has", "Unfavorable intermediate"),
    # ("Favorable intermediate", "has all of", "1 IRF"),
    # ("Favorable intermediate", "has all of", "Grade Group 1 or 2"),
    # ("Favorable intermediate", "has all of", "<50% biopsy cores positive"),
    # ("Unfavorable intermediate", "has", "2 or 3 IRFs"),
    # ("Unfavorable intermediate", "has", "Grade Group 3"),
    # ("Unfavorable intermediate", "has", "≥ 50% biopsy cores positive"),
    # ("High risk group", "has", "cT3a OR"),
    # ("High risk group", "has", "Grade Group 4 or Grade Group 5 OR"),
    # ("High risk group", "has", "PSA >20 ng/mL"),
    # ("Very high risk group", "has at least one of", "cT3b–cT4"),
    # ("Very high risk group", "has at least one of", "Primary Gleason pattern 5"),
    # ("Very high risk group", "has at least one of", "2 or 3 high-risk features"),
    # ("Very high risk group", "has at least one of", ">4 cores with Grade Group 4 or 5"),
    # ("Very low risk group", "additional evaluation", "Confirmatory testing"),
    # ("Very low risk group", "initial therapy", "PROS-3"),
    # ("Low risk group", "additional evaluation", "Confirmatory testing"),
    # ("Low risk group", "initial therapy", "PROS-4"),
    # ("Favorable intermediate risk group", "additional evaluation", "Confirmatory testing"),
    # ("Favorable intermediate risk group", "initial therapy", "PROS-5"),
    # ("Unfavorable intermediate risk group", "additional evaluation", "Bone and soft tissue imaging"),
    # ("Unfavorable intermediate risk group", "initial therapy", "PROS-6"),
    # ("High risk group", "additional evaluation", "Bone and soft tissue imaging"),
    # ("High risk group", "initial therapy", "PROS-7"),
    # ("Very high risk group", "additional evaluation", "Bone and soft tissue imaging"),
    # ("Very high risk group", "initial therapy", "PROS-7"),
    # ("Confirmatory testing", "purpose", "Assess appropriateness of active surveillance"),
    # ("Bone and soft tissue imaging", "condition", "If regional or distant metastases are found"),
    # ("Bone and soft tissue imaging", "next step if metastases found", "See PROS-8 or PROS-13"),
    # ("Very-low-risk group", "expected patient survival ≥10 y", "Active surveillance"),
    # ("Very-low-risk group", "expected patient survival <10 y", "Observation"),
    # ("Active surveillance", "refers to", "See Active Surveillance Program (PROS-F 2 of 5)"),
    # ("Progressive disease", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease (PROS-2)"),
    # ("Observation", "next step", "See Monitoring (PROS-9)"),
    # ("Footnote a", "refers to", "See NCCN Guidelines for Older Adult Oncology"),
    # ("Footnote b", "refers to", "NCCN Guidelines for Prostate Cancer Early Detection"),
    # ("Footnote c", "refers to", "Principles of Bone Health in Prostate Cancer (PROS-K)"),
    # ("Footnote d", "refers to", "Principles of Genetics and Molecular/Biomarker Analysis (PROS-C)"),
    # ("Footnote e", "refers to", "Principles of Quality-of-Life and Shared Decision-Making (PROS-D)"),
    # ("Footnote f", "refers to", "Principles of Imaging (PROS-E)"),
    # ("Footnote g", "refers to", "Bone imaging"),
    # ("Footnote h", "refers to", "PSMA-PET tracers"),
    # ("Footnote i", "refers to", "Principles of Imaging (PROS-E)"),
    # ("Footnote j", "refers to", "For patients who are asymptomatic"),
    # ("Footnote k", "refers to", "An ultrasound- or MRI- or DRE-targeted lesion"),
    # ("Footnote l", "refers to", "Percentage of positive cores in the intermediate-risk group"),
    # ("Footnote m", "refers to", "Bone imaging for symptomatic patients"),
    # ("Footnote n", "refers to", "Expected Patient Survival"),
    # ("Footnote o", "refers to", "Expected Patient Survival ≥10 y"),
    # ("Footnote p", "refers to", "Active surveillance"),
    # ("Footnote q", "refers to", "See Active Surveillance Program"),
    # ("Footnote r", "refers to", "Observation"),
    # ("Footnote s", "refers to", "Progressive disease"),
    # ]
    
        triplets = [
        ("NCCN Guidelines", "version", "4.2024", "describes", "doc1"),
        ("NCCN Guidelines", "topic", "Prostate Cancer", "covers", "doc1"),
        ("Clinically localized prostate cancer", "defined as", "Any T, N0, M0 or Any T, NX, MX", "is defined as", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Perform physical exam", "requires", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Perform digital rectal exam (DRE)", "requires", "doc1"),
        ("Digital rectal exam (DRE)", "purpose", "Confirm clinical stage", "is used to", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Perform and/or collect prostate-specific antigen (PSA)", "requires", "doc1"),
        ("Prostate-specific antigen (PSA)", "requires", "Calculate PSA density", "involves", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Obtain and review diagnostic prostate biopsies", "requires", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Estimate life expectancy", "requires", "doc1"),
        ("Life expectancy estimation", "refers to", "Principles of Life Expectancy Estimation [PROS-A]", "references", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Inquire about known high-risk germline mutations", "requires", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Inquire about family history", "requires", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Perform somatic and/or germline testing", "requires", "doc1"),
        ("Somatic and/or germline testing", "condition", "As appropriate", "is performed", "doc1"),
        ("Clinically localized prostate cancer", "requires workup", "Assess quality-of-life measures", "requires", "doc1"),
        ("Clinically localized prostate cancer", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease [PROS-2]", "proceeds to", "doc1"),
        ("Regional prostate cancer", "defined as", "Any T, N1, M0", "is defined as", "doc1"),
        ("Regional prostate cancer", "requires workup", "Perform physical exam", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Perform imaging for staging", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Perform DRE to confirm clinical stage", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Perform and/or collect PSA", "requires", "doc1"),
        ("PSA", "calculation", "Calculate PSA doubling time (PSADT)", "involves", "doc1"),
        ("Regional prostate cancer", "requires workup", "Estimate life expectancy", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Inquire about known high-risk germline mutations", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Inquire about family history", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Perform somatic and/or germline testing", "requires", "doc1"),
        ("Regional prostate cancer", "requires workup", "Assess quality-of-life measures", "requires", "doc1"),
        ("Regional prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]", "proceeds to", "doc1"),
        ("Metastatic prostate cancer", "defined as", "Any T, Any N, M1", "is defined as", "doc1"),
        ("Metastatic prostate cancer", "next step", "See Regional Prostate Cancer [PROS-8]", "proceeds to", "doc1"),
        ("Metastatic prostate cancer", "next step", "Systemic Therapy for M1 Castration-Sensitive Prostate Cancer (CSPC) [PROS-13]", "proceeds to", "doc1"),
        ("Bone imaging", "can be achieved by", "Conventional technetium-99m-MDP bone scan", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "CT", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "MRI", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "PSMA-PET/CT", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "PSMA-PET/MRI", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "PET/CT", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "PET/MRI with F-18 sodium fluoride", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "C-11 choline", "uses", "doc1"),
        ("Bone imaging", "can be achieved by", "F-18 fluciclovine", "uses", "doc1"),
        ("Equivocal results", "require", "Soft tissue imaging of the pelvis", "necessitates", "doc1"),
        ("Equivocal results", "require", "Abdomen and chest imaging", "necessitates", "doc1"),
        ("Multiparametric MRI (mpMRI)", "preferred over", "CT for pelvic staging", "is preferred to", "doc1"),
        ("PSMA-PET/CT", "can be considered for", "Bone and soft tissue (full body) imaging", "is used for", "doc1"),
        ("PSMA-PET/MRI", "can be considered for", "Bone and soft tissue (full body) imaging", "is used for", "doc1"),
        ("PSMA-PET tracers", "have", "Increased sensitivity and specificity", "possess", "doc1"),
        ("PSMA-PET tracers", "compared to", "Conventional imaging", "are compared with", "doc1"),
        ("Conventional imaging", "includes", "CT", "encompasses", "doc1"),
        ("Conventional imaging", "includes", "Bone scan", "encompasses", "doc1"),
        ("PSMA-PET", "use", "Not a necessary prerequisite", "is considered", "doc1"),
        ("PSMA-PET/CT", "can serve as", "Equally effective frontline imaging tool", "functions as", "doc1"),
        ("PSMA-PET/MRI", "can serve as", "Equally effective frontline imaging tool", "functions as", "doc1"),
        ("Initial Risk Stratification", "for", "Clinically Localized Disease", "is used for", "doc1"),
        ("Very low risk group", "has all of", "cT1c", "includes", "doc1"),
        ("Very low risk group", "has all of", "Grade Group 1", "includes", "doc1"),
        ("Very low risk group", "has all of", "PSA <10 ng/mL", "includes", "doc1"),
        ("Very low risk group", "has all of", "<3 prostate biopsy fragments/cores positive", "includes", "doc1"),
        ("Very low risk group", "has all of", "≤50% cancer in each fragment/core", "includes", "doc1"),
        ("Very low risk group", "has all of", "PSA density <0.15 ng/mL/g", "includes", "doc1"),
        ("Low risk group", "has all of", "cT1–cT2a", "includes", "doc1"),
        ("Low risk group", "has all of", "Grade Group 1", "includes", "doc1"),
        ("Low risk group", "has all of", "PSA <10 ng/mL", "includes", "doc1"),
        ("Low risk group", "does not qualify for", "Very low risk", "is distinct from", "doc1"),
        ("Intermediate risk group", "has", "Favorable intermediate", "includes", "doc1"),
        ("Intermediate risk group", "has", "Unfavorable intermediate", "includes", "doc1"),
        ("Favorable intermediate", "has all of", "1 IRF", "includes", "doc1"),
        ("Favorable intermediate", "has all of", "Grade Group 1 or 2", "includes", "doc1"),
        ("Favorable intermediate", "has all of", "<50% biopsy cores positive", "includes", "doc1"),
        ("Unfavorable intermediate", "has", "2 or 3 IRFs", "includes", "doc1"),
        ("Unfavorable intermediate", "has", "Grade Group 3", "includes", "doc1"),
        ("Unfavorable intermediate", "has", "≥ 50% biopsy cores positive", "includes", "doc1"),
        ("High risk group", "has", "cT3a OR", "includes", "doc1"),
        ("High risk group", "has", "Grade Group 4 or Grade Group 5 OR", "includes", "doc1"),
        ("High risk group", "has", "PSA >20 ng/mL", "includes", "doc1"),
        ("Very high risk group", "has at least one of", "cT3b–cT4", "includes", "doc1"),
        ("Very high risk group", "has at least one of", "Primary Gleason pattern 5", "includes", "doc1"),
        ("Very high risk group", "has at least one of", "2 or 3 high-risk features", "includes", "doc1"),
        ("Very high risk group", "has at least one of", ">4 cores with Grade Group 4 or 5", "includes", "doc1"),
        ("Very low risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
        ("Very low risk group", "initial therapy", "PROS-3", "refers to", "doc1"),
        ("Low risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
        ("Low risk group", "initial therapy", "PROS-4", "refers to", "doc1"),
        ("Favorable intermediate risk group", "additional evaluation", "Confirmatory testing", "requires", "doc1"),
        ("Favorable intermediate risk group", "initial therapy", "PROS-5", "refers to", "doc1"),
        ("Unfavorable intermediate risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
        ("Unfavorable intermediate risk group", "initial therapy", "PROS-6", "refers to", "doc1"),
        ("High risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
        ("High risk group", "initial therapy", "PROS-7", "refers to", "doc1"),
        ("Very high risk group", "additional evaluation", "Bone and soft tissue imaging", "requires", "doc1"),
        ("Very high risk group", "initial therapy", "PROS-7", "refers to", "doc1"),
        ("Confirmatory testing", "purpose", "Assess appropriateness of active surveillance", "is used to", "doc1"),
        ("Bone and soft tissue imaging", "condition", "If regional or distant metastases are found", "is performed", "doc1"),
        ("Bone and soft tissue imaging", "next step if metastases found", "See PROS-8 or PROS-13", "leads to", "doc1"),
        ("Very-low-risk group", "expected patient survival ≥10 y", "Active surveillance", "recommends", "doc1"),
        ("Very-low-risk group", "expected patient survival <10 y", "Observation", "recommends", "doc1"),
        ("Active surveillance", "refers to", "See Active Surveillance Program (PROS-F 2 of 5)", "is detailed in", "doc1"),
        ("Progressive disease", "next step", "See Initial Risk Stratification and Staging Workup for Clinically Localized Disease (PROS-2)", "proceeds to", "doc1"),
        ("Observation", "next step", "See Monitoring (PROS-9)", "proceeds to", "doc1"),
        ("Footnote a", "refers to", "See NCCN Guidelines for Older Adult Oncology", "references", "doc1"),
        ("Footnote b", "refers to", "NCCN Guidelines for Prostate Cancer Early Detection", "references", "doc1"),
        ("Footnote c", "refers to", "Principles of Bone Health in Prostate Cancer (PROS-K)", "references", "doc1"),
        ("Footnote d", "refers to", "Principles of Genetics and Molecular/Biomarker Analysis (PROS-C)", "references", "doc1"),
        ("Footnote e", "refers to", "Principles of Quality-of-Life and Shared Decision-Making (PROS-D)", "references", "doc1"),
        ("Footnote f", "refers to", "Principles of Imaging (PROS-E)", "references", "doc1"),
        ("Footnote g", "refers to", "Bone imaging", "references", "doc1"),
        ("Footnote h", "refers to", "PSMA-PET tracers", "references", "doc1"),
        ("Footnote i", "refers to", "Principles of Imaging (PROS-E)", "references", "doc1"),
        ("Footnote j", "refers to", "For patients who are asymptomatic", "applies to", "doc1"),
        ("Footnote k", "refers to", "An ultrasound- or MRI- or DRE-targeted lesion", "describes", "doc1"),
        ("Footnote l", "refers to", "Percentage of positive cores in the intermediate-risk group", "describes", "doc1"),
        ("Footnote m", "refers to", "Bone imaging for symptomatic patients", "applies to", "doc1"),
        ("Footnote n", "refers to", "Expected Patient Survival", "describes", "doc1"),
        ("Footnote o", "refers to", "Expected Patient Survival ≥10 y", "describes", "doc1"),
        ("Footnote p", "refers to", "Active surveillance", "describes", "doc1"),
        ("Footnote q", "refers to", "See Active Surveillance Program", "references", "doc1"),
        ("Footnote r", "refers to", "Observation", "describes", "doc1"),
        ("Footnote s", "refers to", "Progressive disease", "describes", "doc1"),
]

        graph_store.add_triplets(triplets)
        graph_store.build_communities()
        graph_store.save()  # Save the graph store after adding triplets and documents

    # Print graph information
    graph_store.print_graph_info()

    # Perform queries with the new methods
    print("\nBasic query results for 'Metastatic':")
    basic_results = graph_store.query(subject="Metastatic", fuzzy_match=True, case_sensitive=False)
    for result in basic_results:
        print(f"  {result}")

    print("\nAdvanced query results for 'What is required for Metastatic prostate cancer?':")
    advanced_results = graph_store.advanced_query("What is required for Metastatic prostate cancer?")
    for result in advanced_results:
        print(f"  {result}")

    # Extract subgraph from advanced query results
    subgraph = graph_store.extract_subgraph(advanced_results)
    print(f"\nSubgraph information:")
    print(f"  Number of nodes: {subgraph.number_of_nodes()}")
    print(f"  Number of edges: {subgraph.number_of_edges()}")

    # Print subgraph edges
    print("\nSubgraph edges:")
    for s, o, data in subgraph.edges(data=True):
        print(f"  {s} -> {o} -> {data['relationship']} -> {data['description']} (Document ID: {data.get('doc_id', 'N/A')})")

    # Visualize the subgraph
    # graph_store.visualize_graph(subgraph)
    
    # Convert subgraph to DOT format
    dot_representation = graph_store.subgraph_to_dot(subgraph)
    print("\nDOT representation of the subgraph:")
    print(dot_representation)

    # Save the DOT representation to a file
    with open("subgraph2.dot", "w") as f:
        f.write(dot_representation)

    print("\nDOT representation has been saved to 'subgraph2.dot'")
