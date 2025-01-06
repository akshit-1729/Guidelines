import json
import csv
import re

def extract_content(node):
    content = node.get('nccn:content', '')
    if not content and 'contained_content' in node:
        content = '\n'.join(node['contained_content'])
    return content.strip()

def extract_footnote_name(footnote_id):
    match = re.search(r'/footnote/([a-zA-Z0-9]+)$', footnote_id)
    return match.group(1) if match else ''

def gather_footnotes(node):
    footnotes = []
    
    # Current node footnotes
    for footnote in node.get('footnotes', []):
        footnote_name = extract_footnote_name(footnote['@id'])
        footnote_content = footnote.get('nccn:content', '')
        if footnote_name and footnote_content:
            footnotes.append(f"{footnote_name}: {footnote_content}")
    
    # Next nodes footnotes
    for next_node in node.get('next_nodes', []):
        for footnote in next_node.get('footnotes', []):
            footnote_name = extract_footnote_name(footnote['@id'])
            footnote_content = footnote.get('nccn:content', '')
            if footnote_name and footnote_content:
                footnotes.append(f"{footnote_name}: {footnote_content}")
    
    return list(set(footnotes))  # Remove duplicates

def process_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node ID', 'Page Key', 'Page No', 'Information', 'Footnotes'])

        for node in data:
            node_id = node['@id']
            page_key = node.get('nccn:page-key', '')
            page_no = node.get('nccn:page-no', '')

            label_content = ' '.join(node.get('label_content', []))
            content = extract_content(node)

            # Include contained content of next nodes in context
            next_node_content = []
            for next_node in node.get('next_nodes', []):
                next_content = extract_content(next_node)
                if next_content:
                    next_node_content.append(next_content)
                elif 'contained_content' in next_node:
                    next_node_content.extend(next_node['contained_content'])

            context = f"Context:\n{label_content}\n{content}\n"
            # if next_node_content:
            #     context += "Next node content:\n" + "\n".join(next_node_content) + "\n"

            next_steps = []
            for next_node in node.get('next_nodes', []):
                next_label = ' '.join(next_node.get('label_content', []))
                next_content = extract_content(next_node)
                next_steps.append(f"{next_label}\n{next_content}")

            information = context + '\nNext steps:\n' + '\n\n'.join(next_steps)

            # Gather footnotes from current node and next nodes
            footnotes = gather_footnotes(node)
            footnotes_str = '\n'.join(footnotes)

            csvwriter.writerow([node_id, page_key, page_no, information, footnotes_str])

# Usage
input_file = 'cf_5_2024.json'
output_file = 'ei_4_2024.csv'
process_json(input_file, output_file)