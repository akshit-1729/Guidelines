# import json

# # Load the JSON-LD file
# with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Relationships to extract
# relationships = [
#     'nccn:page-key', 'nccn:labels', 'nccn:content', 'nccn:next', 'nccn:prev',
#     'nccn:parent', 'nccn:reference', 'nccn:contains'
# ]

# # Iterate over each node in the @graph
# for node in data.get('@graph', []):
#     source = node.get('@id')
#     if not source:
#         continue
#     for relationship in relationships:
#         if relationship in node:
#             targets = node[relationship]
#             # Ensure targets is a list
#             if not isinstance(targets, list):
#                 targets = [targets]
#             for target in targets:
#                 # Get the target @id or value
#                 if isinstance(target, dict) and '@id' in target:
#                     target_value = target['@id']
#                 else:
#                     target_value = target
#                 print(f"Source: {source}, Relationship: {relationship}, Target: {target_value}")

import json

# Define Node and Relationship classes
class Node:
    def __init__(self, id, type="Node", properties=None):
        self.id = id
        self.type = type
        self.properties = properties or {}

class Relationship:
    def __init__(self, source, target, type, properties=None):
        self.source = source
        self.target = target
        self.type = type
        self.properties = properties or {}

# Load the JSON-LD file
with open('/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Relationships to extract
relationship_keys = [
    'nccn:page-key', 'nccn:labels', 'nccn:content', 'nccn:next', 'nccn:prev',
    'nccn:parent', 'nccn:reference', 'nccn:contains', 'nccn:nodelink'
]

nodes = {}
relationships = []

# Create Node instances
for item in data.get('@graph', []):
    node_id = item.get('@id')
    if node_id and node_id not in nodes:
        nodes[node_id] = Node(id=node_id, properties={})

# Create Relationship instances
for item in data.get('@graph', []):
    source_id = item.get('@id')
    source_node = nodes.get(source_id)
    if not source_node:
        continue
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
                relationships.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel_type,
                    properties={}
                ))

# Output Nodes
node_counter = 1
node_name_map = {}
for node_id, node in nodes.items():
    node_var_name = f'node{node_counter}'
    node_name_map[node_id] = node_var_name
    print(f'{node_var_name} = Node(id="{node.id}", properties={{}})')
    node_counter += 1

# Output Relationships
rel_counter = 1
for rel in relationships:
    source_name = node_name_map[rel.source.id]
    target_name = node_name_map[rel.target.id]
    print(f'rel{rel_counter} = Relationship(source={source_name}, target={target_name}, type="{rel.type}", properties={{}})')
    rel_counter += 1