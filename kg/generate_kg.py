import pickle
from collections import defaultdict, Counter
from contextlib import redirect_stdout
from pathlib import Path
import json
import argparse
import os
import openai
import time
import numpy as np

import networkx as nx
from pyvis.network import Network
from tqdm import tqdm
from contextlib import redirect_stdout



from .knowledge_graph import generate_knowledge_graph
from .openai_api import load_response_text
from .save_triples import get_response_save_path
from .utils import set_up_logging

logger = set_up_logging('generate-knowledge-graphs-books.log')
KNOWLEDGE_GRAPHS_DIRECTORY_PATH = Path('../knowledge-graphs_new')


def gpt_inference(system_instruction, prompt, retries=10, delay=5):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    messages = [{"role": "system", "content": system_instruction}, 
                {"role": "user", "content": prompt}]
    
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o-mini-2024-07-18',
                messages=messages,
                temperature=0.0,
                max_tokens=128,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0
            )
            result = response['choices'][0]['message']['content']
            return result
        except openai.error.APIError as e:
            
            time.sleep(delay)
            continue


def generate_knowledge_graph_for_scripts(book, idx, save_path):
    """Use the responses from the OpenAI API to generate a knowledge graph for a book."""
    response_texts = defaultdict(list)
    project_gutenberg_id = book['id']
    for chapter in book['chapters']:
        chapter_index = chapter['index']
        chapter_responses_directory = get_response_save_path(
            idx, save_path, project_gutenberg_id, chapter_index)
        for response_path in chapter_responses_directory.glob('*.json'):
            response_text = load_response_text(response_path)
            response_texts[chapter_index].append(response_text)
    knowledge_graph = generate_knowledge_graph(response_texts, project_gutenberg_id)
    return knowledge_graph


def save_knowledge_graph(knowledge_graph,
                         project_gutenberg_id, save_path):
    """Save a knowledge graph to a `pickle` file."""
    save_path = save_path / 'kg.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as knowledge_graph_file:
        pickle.dump(knowledge_graph, knowledge_graph_file)


def load_knowledge_graph(project_gutenberg_id, save_path):
    """Load a knowledge graph from a `pickle` file."""
    save_path = save_path / 'kg.pkl'
    with open(save_path, 'rb') as knowledge_graph_file:
        knowledge_graph = pickle.load(knowledge_graph_file)
    return knowledge_graph


def display_knowledge_graph(knowledge_graph, save_path):
    """Display a knowledge graph using pyvis."""
    # Convert the knowledge graph into a format that can be displayed by pyvis.
    # Merge all edges with the same subject and object into a single edge.
    pyvis_graph = nx.MultiDiGraph()
    for node in knowledge_graph.nodes:
        pyvis_graph.add_node(str(node), label='\n'.join(node.names),
                             shape='box')
    for edge in knowledge_graph.edges(data=True):
        subject = str(edge[0])
        object_ = str(edge[1])
        predicate = edge[2]['predicate']
        chapter_index = edge[2]['chapter_index']
        if pyvis_graph.has_edge(subject, object_):
            pyvis_graph[subject][object_][0].update(
                title=(f'{pyvis_graph[subject][object_][0]["title"]}\n'
                       f'{predicate}')) # f'{predicate} ({chapter_index})'))
        else:
            pyvis_graph.add_edge(subject, object_,
                                 title=f'{predicate}') # title=f'{predicate} ({chapter_index})')
    network = Network(height='99vh', directed=True, bgcolor='#262626',
                      cdn_resources='remote')
    network.set_options('''
    const options = {
        "interaction": {
            "tooltipDelay": 0
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 0
            },
            "solver": "forceAtlas2Based"
        }
    }''')
    network.from_nx(pyvis_graph)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # `show()` tries to print the name of the HTML file to the console, so suppress it.
    with redirect_stdout(None):
        network.show(str(save_path), notebook=False)
    logger.info(f'Saved pyvis knowledge graph to {save_path}.')

def fuse_subject(subjects):
    subject_list = subjects.split('/')
    if len(subject_list) == 1:
        return subject_list[0]
    flag = 0
    striped_subject_list = []
    len_list = []
    for subject in subject_list:
        striped_subject_list.append(subject.strip())
        len_list.append(len(subject))
    idx = np.argmin(len_list)
    for subject in striped_subject_list:
        if striped_subject_list[idx] in subject:
            flag += 1

    if flag == len(striped_subject_list):
        return striped_subject_list[idx]
    else:
        return subjects

def init_kg(idx, save_root, kg_path):
    """
    Generate knowledge graphs for book in the books dataset using saved responses from the OpenAI API.
    """
    # 1) load data
    input_path = f'{save_root}/1_preprocessed/script.json'
    with open(input_path, 'r') as f:
        script = json.load(f)

    # 2) generate character relation graph
    response_path = f'{save_root}/2_responses'
    knowledge_graph = generate_knowledge_graph_for_scripts(script, idx, response_path )
    project_gutenberg_id = script['id']

    # 3) save character relation graph
    os.makedirs(kg_path, exist_ok=True)
    save_knowledge_graph(knowledge_graph, project_gutenberg_id, kg_path)

def refine_kg(idx, kg_path, topk, refine='ner'):
    # 1) load data
    input_path = f'{kg_path}/3_knowledge_graphs/kg.pkl'
    with open(input_path, 'rb') as f:
        knowledge_graph = pickle.load(f)
    save_path = f'{kg_path}/3_knowledge_graphs/refined_kg.html'
    
    # 2) refine kg
    # Calculate the number of edges for each node.
    edge_count = Counter()
    for edge in knowledge_graph.edges(data=True):
        subject = str(edge[0])
        object_ = str(edge[1])
        edge_count[subject] += 1
        edge_count[object_] += 1

    # Select the top k nodes with the most edges.
    top_k_nodes = [node for node, count in edge_count.most_common(topk)]

    # Collect all relationships between the top k nodes.
    rel_dict = {}
    for edge in knowledge_graph.edges(data=True):
        subject = str(edge[0])
        object_ = str(edge[1])
        if subject in top_k_nodes and object_ in top_k_nodes:
            predicate = edge[2]['predicate']
            chapter_index = edge[2]['chapter_index']
            count = edge[2]['count']
            key = f"{subject}\t{object_}"
            if key not in rel_dict:
                rel_dict[key] = []
            rel_dict[key].append((predicate, chapter_index, count))

    # Visualization code
    pyvis_graph = nx.MultiDiGraph()
    for node in top_k_nodes:
        pyvis_graph.add_node(node, label=node, shape='box')

    for key, relations in rel_dict.items():
        subject, object_ = key.split('\t')
        for relation in relations:
            predicate, chapter_index, count = relation
            if 'output' in predicate:
                continue
            if count >= 2:
                if pyvis_graph.has_edge(subject, object_):
                    pyvis_graph[subject][object_][0]['title'] += f', {predicate}'
                else:
                    pyvis_graph.add_edge(subject, object_, title=f'{predicate}')

    network = Network(height='99vh', directed=True, bgcolor='#262626', cdn_resources='remote')
    network.from_nx(pyvis_graph)

    with redirect_stdout(None):
        network.show(str(save_path), notebook=False)

    # Save relationships and nodes to a jsonl file
    root_path = Path(f'{kg_path}/3_knowledge_graphs')
    with open(root_path / 'final_kg.jsonl', 'w', encoding='utf-8') as f:
        for key, relations in rel_dict.items():
            subject, object_ = key.split('\t')
            for relation in relations:
                predicate, chapter_index, count = relation
                if 'output' in predicate:
                    continue

                subject = fuse_subject(subject)
                object_ = fuse_subject(object_)

                relationship = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_,
                    'chapter_index': chapter_index,
                    'count': count,
                    'subject_node_count': edge_count[subject],  # Add the number of edges for the subject node
                    'object_node_count': edge_count[object_]    # Add the number of edges for the object node
                }
                if count >= 2:
                    f.write(json.dumps(relationship, ensure_ascii=False) + '\n')