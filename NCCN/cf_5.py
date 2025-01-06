import json
import re

def extract_footnotes(content):
    return re.findall(r'\{([a-z]+)\}', content)

def extract_footnotes_from_references(node, node_dict):
    footnotes = []
    
    # Extract footnotes from nccn:reference
    if 'nccn:reference' in node:
        for ref in node['nccn:reference']:
            if ref in node_dict:
                footnote_node = node_dict[ref]
                if 'nccn:content' in footnote_node:
                    footnotes.append({
                        '@id': ref,
                        '@type': 'nccn:Footnote',
                        'nccn:content': footnote_node['nccn:content']
                    })
    
    # Extract footnotes from labels
    if 'nccn:labels' in node:
        for label_id in node['nccn:labels']:
            if label_id in node_dict:
                label_node = node_dict[label_id]
                if 'nccn:reference' in label_node:
                    for ref in label_node['nccn:reference']:
                        if ref in node_dict:
                            footnote_node = node_dict[ref]
                            if 'nccn:content' in footnote_node:
                                footnotes.append({
                                    '@id': ref,
                                    '@type': 'nccn:Footnote',
                                    'nccn:content': footnote_node['nccn:content']
                                })
    
    return footnotes

def extract_label_content(node, node_dict):
    label_content = []
    if 'nccn:labels' in node:
        for label_id in node['nccn:labels']:
            if label_id in node_dict and 'nccn:content' in node_dict[label_id]:
                label_content.append(node_dict[label_id]['nccn:content'])
    return label_content

def process_pros_reference(node, node_dict):
    if 'nccn:content' in node and re.search(r'PROS-[0-9A-Za-z]', node['nccn:content']):
        referenced_content = node['nccn:content'] 
        referenced_content2 = ""
        if 'nccn:next' in node:
            for ref_next_id in node['nccn:next']:
                if ref_next_id in node_dict:
                    ref_next_node = node_dict[ref_next_id]
                    if 'nccn:content' in ref_next_node:
                        referenced_content2 += " " + ref_next_node['nccn:content']

            referenced_content2 =  "Referenced content from PROS" + referenced_content2 
        node['nccn:content'] = referenced_content + referenced_content2
    return node


def expand_node(node, node_dict):
    expanded_node = node.copy()
    
    # Process PROS reference for the current node
    expanded_node = process_pros_reference(expanded_node, node_dict)
    
    # Extract footnotes for the current node
    expanded_node['footnotes'] = extract_footnotes_from_references(node, node_dict)

    # Extract label content for the current node
    expanded_node['label_content'] = extract_label_content(node, node_dict)

    # Add next nodes information
    expanded_node['next_nodes'] = []
    if 'nccn:next' in node:
        for next_id in node['nccn:next']:
            if next_id in node_dict:
                next_node = node_dict[next_id].copy()
                
                # Process PROS reference for the next node
                next_node = process_pros_reference(next_node, node_dict)
                
                # Extract label content for the next node
                next_node['label_content'] = extract_label_content(next_node, node_dict)
                
                # If next node's content is empty, process its contained nodes
                if 'nccn:content' in next_node and next_node['nccn:content'] == "":
                    next_node['contained_content'] = []
                    if 'nccn:contains' in next_node:
                        for contained_id in next_node['nccn:contains']:
                            if contained_id in node_dict:
                                contained_node = node_dict[contained_id].copy()
                                
                                # Process PROS reference for the contained node
                                contained_node = process_pros_reference(contained_node, node_dict)
                                
                                if 'nccn:content' in contained_node:
                                    next_node['contained_content'].append(contained_node['nccn:content'])
                                expanded_node['footnotes'].extend(extract_footnotes_from_references(contained_node, node_dict))
                
                # Extract footnotes for the next node and add to current node's footnotes
                expanded_node['footnotes'].extend(extract_footnotes_from_references(next_node, node_dict))
                expanded_node['next_nodes'].append(next_node)

    # Remove duplicate footnotes
    expanded_node['footnotes'] = [dict(t) for t in {tuple(d.items()) for d in expanded_node['footnotes']}]

    return expanded_node


# def expand_node(node, node_dict):
#     expanded_node = node.copy()
    
#     # Process PROS reference for the current node
#     expanded_node = process_pros_reference(expanded_node, node_dict)
    
#     # Extract footnotes for the current node
#     expanded_node['footnotes'] = extract_footnotes_from_references(node, node_dict)

#     # Extract label content for the current node
#     expanded_node['label_content'] = extract_label_content(node, node_dict)

#     # Add next nodes information
#     expanded_node['next_nodes'] = []
#     if 'nccn:next' in node:
#         for next_id in node['nccn:next']:
#             if next_id in node_dict:
#                 next_node = node_dict[next_id].copy()
                
#                 # Process PROS reference for the next node
#                 next_node = process_pros_reference(next_node, node_dict)
                
#                 # Extract label content for the next node
#                 next_node['label_content'] = extract_label_content(next_node, node_dict)
                
#                 # If next node's content is empty, process its contained nodes
#                 if 'nccn:content' in next_node and next_node['nccn:content'] == "":
#                     next_node['contained_content'] = []
#                     if 'nccn:contains' in next_node:
#                         for contained_id in next_node['nccn:contains']:
#                             if contained_id in node_dict:
#                                 contained_node = node_dict[contained_id].copy()
                                
#                                 # Process PROS reference for the contained node
#                                 contained_node = process_pros_reference(contained_node, node_dict)
                                
#                                 if 'nccn:content' in contained_node:
#                                     next_node['contained_content'].append(contained_node['nccn:content'])
#                                 expanded_node['footnotes'].extend(extract_footnotes_from_references(contained_node, node_dict))
                
#                 # Extract footnotes for the next node and add to current node's footnotes
#                 expanded_node['footnotes'].extend(extract_footnotes_from_references(next_node, node_dict))
#                 expanded_node['next_nodes'].append(next_node)

#     # Remove duplicate footnotes
#     expanded_node['footnotes'] = [dict(t) for t in {tuple(d.items()) for d in expanded_node['footnotes']}]

#     return expanded_node


def process_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or '@graph' not in data:
        print(f"Data structure is not as expected. Root type: {type(data)}")
        return
    
    graph = data['@graph']
    node_dict = {node['@id']: node for node in graph if '@id' in node}

    new_data = [expand_node(node, node_dict) for node in graph if '@id' in node]

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)

# Usage
input_file = '/Users/akshit/Documents/Projects/Python-all/information-extraction/Guidelines/Data/NCCN_prostate_4.2024_Graph_12_33.json'
output_file = 'cf_5_2024.json'
process_json(input_file, output_file)