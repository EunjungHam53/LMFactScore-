import itertools
import logging
import re
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import networkx as nx

from .utils import strip_and_remove_empty_strings

logger = logging.getLogger(__name__)

PROMPT_FILE_PATH = Path('templates/story-prompt.txt')
MAX_RESPONSE_EDGE_COUNT = 15
MAX_PREDICATE_WORD_COUNT = 5
MAX_POSSESSION_WORD_COUNT = 2
MAX_MERGEABLE_NODE_EDGE_COUNT = 2
MIN_NODE_EDGE_COUNT = 1


class NamedEntity:
    """A knowledge graph node representing a named entity."""

    def __init__(self, names):
        self.names = names

    def __repr__(self):
        return ' / '.join(self.names)

def remove_number_prefix(text):
    clean_text = re.sub(r'^\d+\.\s*', '', text)
    return clean_text

def parse_response_text(response_text, identifier, are_edges_numbered=True):
    """
    Parse a response text from the OpenAI model into names (a list of names for
    each entity) and edges (relations between entities). `identifier` is a
    string used to identify the response text in error messages.
    """
    lines = strip_and_remove_empty_strings(response_text.split('\n'))
    if 'Named entities' not in lines[0]:
        logger.error(f'{identifier}: First line of response text does not '
                     f'start with "Named entities:". ("{lines[0]}")')
        return [], []
    mode = 'names'
    names = []
    edges = []
    for line in lines[1:]:
        if 'Knowledge graph edges' in line:
            mode = 'edges'
            continue
        if mode == 'names':
            if line.startswith('-'):
                line = line[1:]
            name_group = strip_and_remove_empty_strings(line.split(' / '))
            name_group = [remove_number_prefix(name) for name in name_group]
            names.append(name_group)
        elif mode == 'edges':
            if are_edges_numbered:
                if not re.match(r'^\d{1,2}\. ', line):
                    break
                if int(line.split('.')[0]) > MAX_RESPONSE_EDGE_COUNT:
                    break;
                line = line[3:]
            edge_components = strip_and_remove_empty_strings(line.split(';'))
            if len(edge_components) not in (2, 3):
                continue
            subjects = strip_and_remove_empty_strings(
                edge_components[0].split(','))
            predicate = edge_components[1]
            if len(edge_components) == 3:
                objects = strip_and_remove_empty_strings(
                    edge_components[2].split(','))
            else:
                objects = [None]
            for subject, object_ in product(subjects, objects):
                edge = (subject, predicate, object_)
                edges.append(edge)
    if not names:
        logger.error(f'{identifier}: No names were parsed from the response '
                     f'text.')
    if not edges:
        logger.error(f'{identifier}: No edges were parsed from the response '
                     f'text.')
    return names, edges


def generate_names_graph(names):
    """
    Generate a graph of names where the nodes are names and the edges indicate
    that two names refer to the same entity.
    """
    names_graph = nx.Graph()
    for name_group in names:
        for name in name_group:
            names_graph.add_node(name)
        for name_pair in combinations(name_group, 2):
            names_graph.add_edge(*name_pair)
    return names_graph


def expand_contracted_possessive(predicate, names):
    """
    Check if a predicate is of the form "<owner>'s <possession>", where the
    owner is a named entity. If so, return a predicate of the form
    "<possession> of" and an object of the form "<owner>".
    """
    match = re.search(
        fr'\'s\s\w+(?:\s\w+)'
        fr'{{0,{MAX_POSSESSION_WORD_COUNT - 1}}}$', predicate)
    if not match:
        return predicate, None
    apostrophe_index = match.start()
    owner = next(
        (name for name in names
         if predicate[:apostrophe_index].endswith(name)), None)
    if owner is None:
        return predicate, None
    possession = predicate[apostrophe_index + 2:].strip()
    predicate = (f'{predicate[:apostrophe_index - len(owner)].strip()} '
                 f'{possession} of')
    object_ = owner
    return predicate, object_


def does_duplicate_edge_exist(knowledge_graph, subject, predicate, object_):
    """
    Check if an edge with a given subject, predicate, and object already exists
    in a knowledge graph. If it exists, return the edge data; otherwise, return None.
    """
    for edge in knowledge_graph.edges(subject, data=True):
        if edge[1] == object_ and edge[2]['predicate'] == predicate:
            return edge
    return None


def add_edge_to_knowledge_graph(knowledge_graph, names, edge, max_predicate_word_count, **edge_attributes):
    """Add an edge to a knowledge graph, updating count if the edge already exists."""
    subject, predicate, object_ = edge
    if subject not in names:
        return
    if object_ is not None and object_ not in names:
        predicate += f' {object_}'
        object_ = None
    if object_ is None:
        object_at_end_of_predicate = next(
            (name for name in names if predicate.endswith(' ' + name)), None)
        if object_at_end_of_predicate is not None:
            object_ = object_at_end_of_predicate
            predicate = predicate[:-len(object_)].strip()
        else:
            predicate, object_ = expand_contracted_possessive(predicate, names)
    while predicate.endswith(('.', ',', '!', '?')):
        predicate = predicate[:-1]
    if (max_predicate_word_count and len(predicate.split()) > max_predicate_word_count):
        return
    if subject == object_:
        return
    if object_ is None:
        object_ = subject
    subject_node = next((node for node in knowledge_graph.nodes if subject in node.names), None)
    object_node = next((node for node in knowledge_graph.nodes if object_ in node.names), None)

    if subject_node is None or object_node is None:
        return

    existing_edge = does_duplicate_edge_exist(knowledge_graph, subject_node, predicate, object_node)
    if existing_edge:
        existing_edge[2]['count'] += 1
    else:
        knowledge_graph.add_edge(subject_node, object_node, predicate=predicate, count=1, **edge_attributes)


def initialize_knowledge_graph(names_graph, edges):
    """
    Initialize a knowledge graph from a graph of names and a dictionary of
    edges grouped by chapter index.
    """
    names = set(names_graph.nodes)
    knowledge_graph = nx.MultiDiGraph()
    for name in names:
        knowledge_graph.add_node(NamedEntity({name}))
    for chapter_index, chapter_edges in edges.items():
        for edge in chapter_edges:
            add_edge_to_knowledge_graph(
                knowledge_graph, names, edge,
                max_predicate_word_count=MAX_PREDICATE_WORD_COUNT,
                chapter_index=chapter_index)
    return knowledge_graph


def get_node_edge_count(knowledge_graph, node):
    """
    Get the number of edges for a node in a knowledge graph, excluding
    self-loops.
    """
    edges = (set(knowledge_graph.in_edges(node))
             | set(knowledge_graph.out_edges(node)))
    edge_count = sum(1 for edge in edges if edge[0] is not edge[1])
    return edge_count


def merge_nodes(knowledge_graph, nodes_to_merge):
    """
    Merge a list of nodes in a knowledge graph into one node, combining their
    sets of names and preserving their edges.
    """
    merged_node = NamedEntity(set())
    for node in nodes_to_merge:
        merged_node.names.update(node.names)
    knowledge_graph.add_node(merged_node)
    for node in nodes_to_merge:
        for edge in itertools.chain(knowledge_graph.out_edges(node, data=True),
                                    knowledge_graph.in_edges(node, data=True)):
            subject, object_, attributes = edge
            if (does_duplicate_edge_exist(knowledge_graph, merged_node,
                                          attributes['predicate'], object_)
                    or does_duplicate_edge_exist(knowledge_graph, subject,
                                                 attributes['predicate'],
                                                 merged_node)):
                continue
            if subject is object_:
                knowledge_graph.add_edge(merged_node, merged_node,
                                         **attributes)
            if subject is node:
                knowledge_graph.add_edge(merged_node, object_, **attributes)
            else:
                knowledge_graph.add_edge(subject, merged_node, **attributes)
        knowledge_graph.remove_node(node)

def merge_same_entity_nodes(knowledge_graph, names_graph):
    """
    Using a graph of names, merge nodes in a knowledge graph corresponding to
    the same entity.
    """
    for name_pair in names_graph.edges:
        first_node = next((node for node in knowledge_graph.nodes
                           if name_pair[0] in node.names), None)
        if first_node is None:
            continue
        if name_pair[1] in first_node.names:
            continue
        second_node = next((node for node in knowledge_graph.nodes
                            if name_pair[1] in node.names), None)
        if second_node is None:
            continue
        if knowledge_graph.has_edge(first_node, second_node):
            continue
        first_node_edge_count = get_node_edge_count(knowledge_graph,
                                                    first_node)
        second_node_edge_count = get_node_edge_count(knowledge_graph,
                                                     second_node)
        if (first_node_edge_count > MAX_MERGEABLE_NODE_EDGE_COUNT
                and second_node_edge_count > MAX_MERGEABLE_NODE_EDGE_COUNT):
            continue
        merge_nodes(knowledge_graph, [first_node, second_node])



def remove_nodes_with_few_edges(knowledge_graph):
    """
    Remove nodes that have fewer than `MIN_NODE_EDGE_COUNT` edges (excluding
    self-loops) from a knowledge graph. Repeat until no more nodes are removed.
    """
    while True:
        nodes_to_remove = []
        for node in knowledge_graph.nodes:
            edge_count = get_node_edge_count(knowledge_graph, node)
            if edge_count < MIN_NODE_EDGE_COUNT:
                nodes_to_remove.append(node)
        if not nodes_to_remove:
            break
        knowledge_graph.remove_nodes_from(nodes_to_remove)


def generate_knowledge_graph(response_texts, project_gutenberg_index):
    """
    Use OpenAI API response texts grouped by chapter index to generate a
    knowledge graph for a book.
    """
    names = []
    edges = defaultdict(list)
    for chapter_index, chapter_response_texts in response_texts.items():
        for response_text in chapter_response_texts:
            identifier = (f'Book {project_gutenberg_index}, chapter '
                          f'{chapter_index}')
            chapter_segment_names, chapter_segment_edges = parse_response_text(
                response_text, identifier)
            names.extend(chapter_segment_names)
            edges[chapter_index].extend(chapter_segment_edges)
    names_graph = generate_names_graph(names)
    knowledge_graph = initialize_knowledge_graph(names_graph, edges)
    merge_same_entity_nodes(knowledge_graph, names_graph)
    remove_nodes_with_few_edges(knowledge_graph)
    return knowledge_graph
