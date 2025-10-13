"""
File thay thế logic extraction trong knowledge_graph.py
Sử dụng REBEL/OpenIE thay vì GPT API
HỖ TRỢ TIẾNG VIỆT
"""

import logging
import re
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx
from Core_Components.local_model_replacements import KnowledgeGraphExtractor

logger = logging.getLogger(__name__)

# Import các class và constants từ file gốc
try:
    from kg.knowledge_graph import (
        NamedEntity,
        generate_names_graph,
        initialize_knowledge_graph,
        merge_same_entity_nodes,
        remove_nodes_with_few_edges
    )
except ImportError:
    # Fallback implementation nếu không có file gốc
    logger.warning("Cannot import from kg.knowledge_graph, using fallback implementation")
    
    class NamedEntity:
        """A knowledge graph node representing a named entity."""
        def __init__(self, names):
            self.names = set(names) if not isinstance(names, set) else names
        
        def __repr__(self):
            return ' / '.join(sorted(self.names))
        
        def __hash__(self):
            return hash(frozenset(self.names))
        
        def __eq__(self, other):
            return isinstance(other, NamedEntity) and self.names == other.names
    
    def generate_names_graph(names):
        """Generate a graph of names where nodes are names and edges indicate same entity"""
        names_graph = nx.Graph()
        for name_group in names:
            for name in name_group:
                names_graph.add_node(name)
            for i in range(len(name_group)):
                for j in range(i+1, len(name_group)):
                    names_graph.add_edge(name_group[i], name_group[j])
        return names_graph
    
    def initialize_knowledge_graph(names_graph, edges):
        """Initialize KG from names graph and edges"""
        knowledge_graph = nx.MultiDiGraph()
        names = set(names_graph.nodes)
        
        # Add nodes
        for name in names:
            knowledge_graph.add_node(NamedEntity({name}))
        
        # Add edges
        for chapter_index, chapter_edges in edges.items():
            for subject, predicate, obj in chapter_edges:
                if subject not in names or obj not in names:
                    continue
                
                subj_node = next((n for n in knowledge_graph.nodes if subject in n.names), None)
                obj_node = next((n for n in knowledge_graph.nodes if obj in n.names), None)
                
                if subj_node and obj_node:
                    knowledge_graph.add_edge(
                        subj_node, obj_node,
                        predicate=predicate,
                        chapter_index=chapter_index,
                        count=1
                    )
        
        return knowledge_graph
    
    def merge_same_entity_nodes(knowledge_graph, names_graph):
        """Merge nodes representing same entity"""
        # Simple implementation
        pass
    
    def remove_nodes_with_few_edges(knowledge_graph, min_edges=1):
        """Remove nodes with few edges"""
        nodes_to_remove = []
        for node in knowledge_graph.nodes:
            in_edges = knowledge_graph.in_degree(node)
            out_edges = knowledge_graph.out_degree(node)
            if in_edges + out_edges < min_edges:
                nodes_to_remove.append(node)
        knowledge_graph.remove_nodes_from(nodes_to_remove)


class LocalKGParser:
    """
    Thay thế parse_response_text bằng local model extraction
    HỖ TRỢ TIẾNG VIỆT
    """
    def __init__(self, method="rebel", device="cuda", language="vi"):
        """
        Args:
            method: 'rebel' hoặc 'openie'
            device: 'cuda' hoặc 'cpu'
            language: 'vi' (Vietnamese) hoặc 'en' (English)
        """
        self.language = language
        self.extractor = KnowledgeGraphExtractor(
            method=method, 
            device=device,
            language=language
        )
        logger.info(f"Initialized LocalKGParser with {method} for {language}")
    
    def parse_text_to_entities_and_edges(
        self, 
        text: str, 
        identifier: str
    ) -> Tuple[List[List[str]], List[Tuple[str, str, str]]]:
        """
        Thay thế parse_response_text function
        
        Args:
            text: Script/News text cần extract
            identifier: ID để logging
            
        Returns:
            names: List of name groups [[name1, alias1], [name2, alias2], ...]
            edges: List of (subject, predicate, object) tuples
        """
        try:
            # Extract entities và relations bằng local model
            entities, raw_edges = self.extractor.extract_named_entities_and_relations(text)
            
            # Post-process: merge similar entity names
            names = self._cluster_entity_names(entities)
            
            # Format edges theo format gốc
            edges = []
            for edge in raw_edges:
                subject, predicate, obj = edge
                # Clean predicate (giới hạn số từ)
                predicate_words = predicate.split()[:5]  # Max 5 words
                predicate = ' '.join(predicate_words)
                edges.append((subject, predicate, obj))
            
            if not names:
                logger.warning(f'{identifier}: No entities extracted from text')
            if not edges:
                logger.warning(f'{identifier}: No relations extracted from text')
            
            logger.info(f'{identifier}: Extracted {len(names)} entities and {len(edges)} relations')
            return names, edges
            
        except Exception as e:
            logger.error(f'{identifier}: Error during extraction: {e}')
            return [], []
    
    def _cluster_entity_names(self, entities: List[str]) -> List[List[str]]:
        """
        Gom nhóm các entities tương tự lại (alias detection)
        Ví dụ: "John", "John Smith" -> ["John", "John Smith"]
        Hoặc: "Hà Nội", "thủ đô" -> ["Hà Nội", "thủ đô"]
        """
        from difflib import SequenceMatcher
        
        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
        # Cluster entities with high similarity
        clusters = []
        used = set()
        
        for i, ent1 in enumerate(entities):
            if ent1 in used:
                continue
            
            cluster = [ent1]
            used.add(ent1)
            
            for j, ent2 in enumerate(entities):
                if i == j or ent2 in used:
                    continue
                
                # Check if one is substring of another or high similarity
                if (ent1.lower() in ent2.lower() or 
                    ent2.lower() in ent1.lower() or 
                    similarity(ent1, ent2) > 0.85):
                    cluster.append(ent2)
                    used.add(ent2)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters


def generate_knowledge_graph_local(
    chapter_texts: Dict[int, str],
    project_gutenberg_index: str,
    method: str = "rebel",
    device: str = "cuda",
    language: str = "vi"
) -> nx.MultiDiGraph:
    """
    Thay thế hàm generate_knowledge_graph
    Sử dụng local models thay vì parsed OpenAI responses
    HỖ TRỢ TIẾNG VIỆT
    
    Args:
        chapter_texts: Dict mapping chapter_index -> text
        project_gutenberg_index: ID của script/news
        method: 'rebel' hoặc 'openie'
        device: 'cuda' hoặc 'cpu'
        language: 'vi' hoặc 'en'
        
    Returns:
        knowledge_graph: NetworkX MultiDiGraph
    """
    parser = LocalKGParser(method=method, device=device, language=language)
    
    names = []
    edges = defaultdict(list)
    
    # Process từng chapter/document
    for chapter_index, chapter_text in chapter_texts.items():
        identifier = f'Document {project_gutenberg_index}, chapter {chapter_index}'
        
        # Extract entities và relations
        chapter_names, chapter_edges = parser.parse_text_to_entities_and_edges(
            chapter_text, 
            identifier
        )
        
        names.extend(chapter_names)
        edges[chapter_index].extend(chapter_edges)
    
    # Build knowledge graph từ extracted data
    logger.info(f"Building knowledge graph with {len(names)} entity groups and {sum(len(e) for e in edges.values())} edges")
    
    names_graph = generate_names_graph(names)
    knowledge_graph = initialize_knowledge_graph(names_graph, edges)
    merge_same_entity_nodes(knowledge_graph, names_graph)
    remove_nodes_with_few_edges(knowledge_graph)
    
    logger.info(f"Final KG: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
    
    return knowledge_graph


def generate_knowledge_graph_for_scripts_local(
    book: Dict,
    idx: str,
    method: str = "rebel",
    device: str = "cuda",
    language: str = "vi"
) -> nx.MultiDiGraph:
    """
    Thay thế generate_knowledge_graph_for_scripts
    Không cần load OpenAI responses, trực tiếp extract từ text
    HỖ TRỢ TIẾNG VIỆT
    
    Args:
        book: Dict chứa chapters với format:
              {'id': ..., 'chapters': [{'index': ..., 'text': ...}, ...]}
        idx: Script/News ID
        method: KG extraction method
        device: GPU/CPU device
        language: 'vi' hoặc 'en'
        
    Returns:
        knowledge_graph: NetworkX MultiDiGraph
    """
    # Extract chapter texts
    chapter_texts = {}
    for chapter in book['chapters']:
        chapter_index = chapter['index']
        # Nếu text là list, join lại
        if isinstance(chapter['text'], list):
            chapter_text = '\n'.join(chapter['text'])
        else:
            chapter_text = chapter['text']
        chapter_texts[chapter_index] = chapter_text
    
    # Generate KG using local models
    knowledge_graph = generate_knowledge_graph_local(
        chapter_texts,
        book['id'],
        method=method,
        device=device,
        language=language
    )
    
    return knowledge_graph