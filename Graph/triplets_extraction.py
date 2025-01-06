import json

def get_node_content(graph, node_id):
    for node in graph:
        if node["@id"] == node_id:
            return node.get("nccn:content", "not_available")
    return "not_available"


def get_label_content(graph, label_id):
    # Similar to get_node_content but specifically for labels
    for node in graph:
        if node["@id"] == label_id and node.get("@type") == "nccn:Labels":
            return node.get("nccn:content", "not_available")
    return "not_available"


def get_reference_content(graph, ref_id):
    # Similar to get_label_content but for references (footnotes)
    for node in graph:
        if node["@id"] == ref_id and node.get("@type") == "nccn:Footnote":
            return node.get("nccn:content", "not_available")
    return "not_available"


def extract_triplets_from_json(json_file):
    with open(json_file, 'r') as f:
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
            triplets.append((source, "page_key", node["nccn:page-key"],"", node.get("@id", "")))
            
        # Page number relationship
        if "nccn:page-no" in node:
            triplets.append((source, "page_no", str(node["nccn:page-no"]),"", node.get("@id", "")))
            
        # Label relationships
        if "nccn:labels" in node:
            for label_id in node["nccn:labels"]:
                label_content = get_label_content(graph, label_id)
                triplets.append((source, "label", label_content,"", node.get("@id", "")))
                
        # Reference relationships
        if "nccn:reference" in node:
            for ref_id in node["nccn:reference"]:
                ref_content = get_reference_content(graph, ref_id)
                triplets.append((source, "refers_to", ref_content,"", node.get("@id", "")))
                
        # Previous node relationships
        if "nccn:prev" in node:
            for prev_id in node["nccn:prev"]:
                prev_content = get_node_content(graph, prev_id)
                triplets.append((source, "previous_node", prev_content,"", node.get("@id", "")))
                
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
                                    triplets.append((source, "next_contained_nodes", content,"", node.get("@id", "")))
                            break
                else:
                    triplets.append((source, "next_node", next_content,"", node.get("@id", "")))
    
    # print("triplets in function : ", triplets)
    return triplets

# Example usage
json_file = '/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json'
triplets = extract_triplets_from_json(json_file)

# print(triplets)

# Print triplets
for source, relation, target, empty, idd in triplets:
    print(f"Source: {source}")
    print("-" * 50)

    print(f"Relation: {relation}")
    print("-" * 50)

    print(f"Target: {target}")
    print("-" * 50)
    
    print(f"empty: {empty}")
    print("-" * 50)
    
    print(f"ID: {idd}")
    print("-" * 50)

    
    
