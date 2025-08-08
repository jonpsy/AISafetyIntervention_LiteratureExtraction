import networkx as nx
import json
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")
from typing import Dict, List, Set, Tuple, Optional, Union
import uuid
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from datetime import datetime
import hashlib
import re
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CausalNode:
    """Enhanced node structure with better merging support"""
    DOI_URL: List[str]
    authors: List[str]
    institutions: List[str]
    timestamp: List[str]
    concept_text: str
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    isIntervention: int = 0
    stage_in_pipeline: Optional[int] = None
    maturity_level: Optional[int] = None
    implemented: Optional[int] = None
    
    # Enhanced fields for better merging
    node_id: str = field(default="", init=False)
    aliases: List[str] = field(default_factory=list)
    canonical_text: str = field(default="", init=False)
    text_hash: str = field(default="", init=False)
    semantic_keywords: List[str] = field(default_factory=list)
    confidence_score: float = field(default=1.0)
    source_papers: List[str] = field(default_factory=list)
    merge_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = self._generate_node_id()
        self.canonical_text = self._canonicalize_text(self.concept_text)
        self.text_hash = self._generate_text_hash()
        self.semantic_keywords = self._extract_keywords()
    
    def _generate_node_id(self) -> str:
        """Generate a stable node ID"""
        clean_text = re.sub(r'[^\w\s]', '', self.concept_text.upper())
        return clean_text.replace(" ", "_")[:50]  # Limit length
    
    def _canonicalize_text(self, text: str) -> str:
        """Create canonical version of text for comparison"""
        # Remove punctuation, convert to lowercase, normalize whitespace
        canonical = re.sub(r'[^\w\s]', '', text.lower())
        canonical = re.sub(r'\s+', ' ', canonical).strip()
        return canonical
    
    def _generate_text_hash(self) -> str:
        """Generate hash of canonical text for quick comparison"""
        return hashlib.md5(self.canonical_text.encode()).hexdigest()[:8]
    
    def _extract_keywords(self) -> List[str]:
        """Extract semantic keywords for similarity comparison"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        keywords = []
        text_lower = self.canonical_text
        
        # AI Safety specific keywords
        ai_safety_terms = {
            'alignment', 'misalignment', 'safety', 'oversight', 'reward', 'hacking',
            'deception', 'constitutional', 'interpretability', 'robustness', 'scaling',
            'capability', 'control', 'evaluation', 'training', 'optimization', 'mesa',
            'inner', 'outer', 'objective', 'gradient', 'intervention', 'monitoring',
            'detection', 'prevention', 'mitigation', 'framework', 'protocol', 'system'
        }
        
        for term in ai_safety_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))

@dataclass 
class CausalEdge:
    """Enhanced edge structure with better relationship modeling"""
    DOI_URL: List[str]
    authors: List[str]
    institutions: List[str]
    timestamp: List[str]
    edge_text: str
    source_nodes: List[str]
    target_nodes: List[str]
    confidence: int
    
    # Enhanced fields
    edge_id: str = field(default="", init=False)
    relationship_type: str = field(default="causes")
    relationship_strength: float = field(default=1.0)
    canonical_relationship: str = field(default="", init=False)
    semantic_hash: str = field(default="", init=False)
    source_papers: List[str] = field(default_factory=list)
    evidence_count: int = field(default=1)
    
    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = self._generate_edge_id()
        self.relationship_type = self._infer_relationship_type()
        self.relationship_strength = self._calculate_strength()
        self.canonical_relationship = self._canonicalize_relationship()
        self.semantic_hash = self._generate_semantic_hash()
    
    def _generate_edge_id(self) -> str:
        """Generate stable edge ID"""
        source_clean = "_".join(self.source_nodes).replace(" ", "_")[:30]
        target_clean = "_".join(self.target_nodes).replace(" ", "_")[:30]
        return f"{source_clean}_TO_{target_clean}"
    
    def _infer_relationship_type(self) -> str:
        """Enhanced relationship type inference"""
        text_lower = self.edge_text.lower()
        
        relationship_patterns = {
            'prevents': ['prevents', 'mitigates', 'reduces', 'alleviates', 'blocks', 'stops'],
            'enables': ['enables', 'facilitates', 'allows', 'supports', 'helps'],
            'causes': ['causes', 'leads to', 'results in', 'triggers', 'produces'],
            'moderates': ['moderates', 'influences', 'affects', 'modifies'],
            'mediates': ['mediates', 'through', 'via', 'by means of'],
            'necessitates': ['necessitates', 'requires', 'demands', 'needs'],
            'correlates': ['correlates', 'associated with', 'related to']
        }
        
        for rel_type, patterns in relationship_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return rel_type
        
        return 'causes'  # default
    
    def _calculate_strength(self) -> float:
        """Calculate relationship strength based on confidence and text analysis"""
        base_strength = self.confidence / 5.0  # Normalize to 0-1
        
        # Boost strength for certain relationship types
        strength_multipliers = {
            'causes': 1.0,
            'prevents': 0.9,
            'enables': 0.8,
            'necessitates': 1.1,
            'correlates': 0.6
        }
        
        multiplier = strength_multipliers.get(self.relationship_type, 1.0)
        return min(base_strength * multiplier, 1.0)
    
    def _canonicalize_relationship(self) -> str:
        """Create canonical representation of the relationship"""
        # Normalize the relationship description
        canonical = re.sub(r'[^\w\s]', '', self.edge_text.lower())
        canonical = re.sub(r'\s+', ' ', canonical).strip()
        return canonical
    
    def _generate_semantic_hash(self) -> str:
        """Generate semantic hash for relationship comparison"""
        # Combine relationship type, source concepts, and target concepts
        semantic_content = f"{self.relationship_type}_{self.canonical_relationship}"
        return hashlib.md5(semantic_content.encode()).hexdigest()[:8]

class EnhancedGlobalCausalGraph:
    """Highly scalable causal graph merging system"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        
        # Enhanced indexing for scalability
        self.node_text_index: Dict[str, str] = {}  # canonical_text -> node_id
        self.node_hash_index: Dict[str, str] = {}  # text_hash -> node_id
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> set of node_ids
        self.edge_semantic_index: Dict[str, List[str]] = defaultdict(list)  # semantic_hash -> edge_keys
        
        # Similarity computation
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = None
        self.concept_vectors = None
        
        # Statistics
        self.merge_stats = {
            'nodes_merged': 0,
            'edges_merged': 0,
            'similarity_computations': 0,
            'exact_matches': 0,
            'semantic_matches': 0,
            'stage_1_matches': 0,
            'stage_2_matches': 0,
            'stage_3_matches': 0,
            'stage_4_matches': 0,
            'stage_4_skipped': 0
        }
        
        # Performance tracking
        self.performance_stats = {
            'stage_timings': defaultdict(list),
            'total_matching_time': 0,
            'average_match_time_by_stage': {}
        }
        
        self.confidence_threshold = 2
    
    def merge_local_graphs(self, local_graphs: List[dict], batch_size: int = 100):
        """Scalable merging with enhanced batching and progress tracking for massive scale"""
        logger.info(f"Starting enhanced multi-stage merge of {len(local_graphs)} local graphs...")
        
        total_graphs = len(local_graphs)
        
        # Adaptive batch sizing based on graph size
        if len(self.nodes) > 50000:
            batch_size = min(batch_size, 50)  # Smaller batches for massive graphs
        elif len(self.nodes) > 10000:
            batch_size = min(batch_size, 75)
        
        # Process in batches for memory efficiency
        for batch_start in range(0, total_graphs, batch_size):
            batch_end = min(batch_start + batch_size, total_graphs)
            batch = local_graphs[batch_start:batch_end]
            
            batch_num = batch_start//batch_size + 1
            total_batches = (total_graphs-1)//batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (Adaptive batch size: {len(batch)})")
            
            # Step 1: Merge nodes with enhanced multi-stage similarity (indices updated during merge)
            self._merge_nodes_batch_optimized(batch)
            
            # Step 3: Merge edges with relationship analysis
            self._merge_edges_batch(batch)
            
            # Step 4: Periodic index optimization for massive scale
            if batch_start % (batch_size * 5) == 0 and len(self.nodes) > 10000:
                self._optimize_indices_for_scale()
        
        # Final processing
        self._aggregate_evidence()
        self._filter_low_confidence_edges()
        self._compute_final_statistics()
        
        logger.info("Enhanced multi-stage merge completed successfully!")
        self._print_merge_summary()
    
    def _update_indices(self, batch: List[dict]):
        """Update all indices with new batch data"""
        for local_graph_data in batch:
            # Extract nodes and build indices
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                
                # Update text indices
                self.node_text_index[node.canonical_text] = node.node_id
                self.node_hash_index[node.text_hash] = node.node_id
                
                # Update keyword index
                for keyword in node.semantic_keywords:
                    self.keyword_index[keyword].add(node.node_id)
    
    def _update_indices_optimized(self, batch: List[dict]):
        """Memory-efficient index updating for massive scale"""
        # Batch collect all nodes first to minimize index operations
        nodes_to_process = []
        
        for local_graph_data in batch:
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                nodes_to_process.append(node)
        
        # Batch update indices
        for node in nodes_to_process:
            # Update text indices
            self.node_text_index[node.canonical_text] = node.node_id
            self.node_hash_index[node.text_hash] = node.node_id
            
            # Bulk update keyword index
            for keyword in node.semantic_keywords:
                self.keyword_index[keyword].add(node.node_id)
    
    def _merge_nodes_batch_optimized(self, batch: List[dict]):
        """Enhanced node merging with optimized multi-stage matching"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                node.source_papers = [paper_id]
                
                # Enhanced multi-stage matching with performance tracking
                canonical_id = self._find_best_node_match(node)
                
                if canonical_id:
                    # Validate that the canonical_id exists before merging
                    if canonical_id in self.nodes:
                        # Merge with existing node
                        self._merge_node_data(canonical_id, node)
                        self.merge_stats['nodes_merged'] += 1
                    else:
                        # Index inconsistency - log warning and add as new node
                        logger.warning(f"Index inconsistency: canonical_id '{canonical_id}' not found in nodes. Adding as new node.")
                        self._add_new_node(node)
                        # Clean up invalid index entries
                        self._cleanup_invalid_indices(canonical_id)
                else:
                    # Add as new node
                    self._add_new_node(node)
    
    def _optimize_indices_for_scale(self):
        """Periodic index optimization for massive scale operations"""
        logger.info("Optimizing indices for massive scale...")
        
        # Clean up empty keyword entries
        empty_keywords = [k for k, v in self.keyword_index.items() if not v]
        for keyword in empty_keywords:
            del self.keyword_index[keyword]
        
        # Log index sizes for monitoring
        logger.debug(f"Index sizes: hash={len(self.node_hash_index)}, "
                    f"text={len(self.node_text_index)}, "
                    f"keywords={len(self.keyword_index)}")
        
        # Memory optimization: limit keyword index size if too large
        if len(self.keyword_index) > 100000:
            # Keep only high-frequency keywords for performance
            keyword_counts = {k: len(v) for k, v in self.keyword_index.items()}
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Keep top 50,000 most frequent keywords
            keywords_to_keep = set(k for k, _ in sorted_keywords[:50000])
            self.keyword_index = {
                k: v for k, v in self.keyword_index.items() 
                if k in keywords_to_keep
            }
            
            logger.info(f"Optimized keyword index: kept {len(self.keyword_index)} high-frequency keywords")
    
    def _cleanup_invalid_indices(self, invalid_id: str):
        """Clean up invalid entries from indices"""
        # Remove from text indices
        keys_to_remove = []
        for key, value in self.node_text_index.items():
            if value == invalid_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.node_text_index[key]
        
        # Remove from hash indices
        keys_to_remove = []
        for key, value in self.node_hash_index.items():
            if value == invalid_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.node_hash_index[key]
        
        # Remove from keyword indices
        for keyword_set in self.keyword_index.values():
            keyword_set.discard(invalid_id)
    
    def _merge_nodes_batch(self, batch: List[dict]):
        """Advanced node merging with multiple similarity methods"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                node.source_papers = [paper_id]
                
                # Find potential matches using multiple methods
                canonical_id = self._find_best_node_match(node)
                
                if canonical_id:
                    # Validate that the canonical_id exists before merging
                    if canonical_id in self.nodes:
                        # Merge with existing node
                        self._merge_node_data(canonical_id, node)
                        self.merge_stats['nodes_merged'] += 1
                    else:
                        # Index inconsistency - log warning and add as new node
                        logger.warning(f"Index inconsistency: canonical_id '{canonical_id}' not found in nodes. Adding as new node.")
                        self._add_new_node(node)
                        # Clean up invalid index entries
                        self._cleanup_invalid_indices(canonical_id)
                else:
                    # Add as new node
                    self._add_new_node(node)
    
    def _find_best_node_match(self, node: CausalNode) -> Optional[str]:
        """Enhanced Multi-Stage Node Matching for Massive Scale
        
        Stage 1: Exact Hash Matching (O(1) - fastest)
        Stage 2: Canonical Text Matching (O(1) with indexing)  
        Stage 3: Keyword-Based Similarity (O(k) where k = keyword matches)
        Stage 4: Semantic Similarity (O(n) - only for smaller graphs)
        """
        import time
        start_time = time.time()
        
        # Stage 1: Exact Hash Matching (O(1)) - Fastest possible lookup
        if node.text_hash in self.node_hash_index:
            self.merge_stats['exact_matches'] += 1
            self.merge_stats['stage_1_matches'] = self.merge_stats.get('stage_1_matches', 0) + 1
            self._log_match_timing('stage_1_hash', time.time() - start_time)
            return self.node_hash_index[node.text_hash]
        
        # Stage 2: Canonical Text Matching (O(1) with indexing) - Direct text lookup
        if node.canonical_text in self.node_text_index:
            self.merge_stats['exact_matches'] += 1
            self.merge_stats['stage_2_matches'] = self.merge_stats.get('stage_2_matches', 0) + 1
            self._log_match_timing('stage_2_canonical', time.time() - start_time)
            return self.node_text_index[node.canonical_text]
        
        # Stage 3: Keyword-Based Similarity (O(k)) - Limited by keyword matches
        keyword_matches = self._find_keyword_matches_optimized(node)
        if keyword_matches:
            best_match = self._compute_best_similarity_match_fast(node, keyword_matches)
            if best_match:
                self.merge_stats['semantic_matches'] += 1
                self.merge_stats['stage_3_matches'] = self.merge_stats.get('stage_3_matches', 0) + 1
                self._log_match_timing('stage_3_keyword', time.time() - start_time)
                return best_match
        
        # Stage 4: Semantic Similarity (O(n)) - Most expensive, use sparingly
        # Only run for graphs under 10,000 nodes to maintain performance
        if len(self.nodes) < 10000:
            semantic_match = self._find_semantic_match_optimized(node)
            if semantic_match:
                self.merge_stats['semantic_matches'] += 1
                self.merge_stats['stage_4_matches'] = self.merge_stats.get('stage_4_matches', 0) + 1
                self._log_match_timing('stage_4_semantic', time.time() - start_time)
                return semantic_match
        else:
            # For massive graphs, skip semantic matching to maintain O(1) average performance
            self.merge_stats['stage_4_skipped'] = self.merge_stats.get('stage_4_skipped', 0) + 1
            logger.debug(f"Skipped semantic matching for massive graph (nodes: {len(self.nodes)})")
        
        # No match found at any stage
        self._log_match_timing('no_match', time.time() - start_time)
        return None
    
    def _find_keyword_matches(self, node: CausalNode) -> List[str]:
        """Find potential matches based on shared keywords"""
        candidate_nodes = set()
        
        for keyword in node.semantic_keywords:
            candidate_nodes.update(self.keyword_index[keyword])
        
        # Filter candidates that share significant keywords
        matches = []
        for candidate_id in candidate_nodes:
            if candidate_id in self.nodes:
                candidate_node = self.nodes[candidate_id]
                shared_keywords = set(node.semantic_keywords) & set(candidate_node.semantic_keywords)
                
                # Require significant keyword overlap
                if len(shared_keywords) >= min(2, len(node.semantic_keywords) * 0.5):
                    matches.append(candidate_id)
        
        return matches
    
    def _find_keyword_matches_optimized(self, node: CausalNode) -> List[str]:
        """Optimized keyword matching with early termination and ranking"""
        if not node.semantic_keywords:
            return []
        
        # Count keyword overlaps for efficient ranking
        candidate_scores = defaultdict(int)
        
        # Prioritize high-value keywords (more specific terms)
        keyword_weights = self._compute_keyword_weights(node.semantic_keywords)
        
        for keyword in node.semantic_keywords:
            weight = keyword_weights.get(keyword, 1.0)
            for candidate_id in self.keyword_index[keyword]:
                candidate_scores[candidate_id] += weight
        
        # Early termination: only consider top candidates
        min_score = max(2.0, len(node.semantic_keywords) * 0.4)
        promising_candidates = [
            candidate_id for candidate_id, score in candidate_scores.items() 
            if score >= min_score and candidate_id in self.nodes
        ]
        
        # Sort by score for better matching order
        promising_candidates.sort(
            key=lambda x: candidate_scores[x], 
            reverse=True
        )
        
        return promising_candidates[:50]  # Limit to top 50 for performance
    
    def _compute_keyword_weights(self, keywords: List[str]) -> Dict[str, float]:
        """Compute weights for keywords based on specificity"""
        weights = {}
        
        for keyword in keywords:
            # Weight by inverse frequency (rarer keywords are more specific)
            frequency = len(self.keyword_index[keyword])
            if frequency == 0:
                weights[keyword] = 2.0  # New keyword, high weight
            else:
                # Inverse log frequency with minimum weight of 0.5
                weights[keyword] = max(0.5, 2.0 - np.log10(frequency + 1))
        
        return weights
    
    def _compute_best_similarity_match(self, node: CausalNode, candidates: List[str]) -> Optional[str]:
        """Compute similarity scores and return best match above threshold"""
        best_match = None
        best_score = 0
        
        for candidate_id in candidates:
            if candidate_id not in self.nodes:
                continue
                
            candidate_node = self.nodes[candidate_id]
            score = self._compute_node_similarity(node, candidate_node)
            
            self.merge_stats['similarity_computations'] += 1
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _compute_best_similarity_match_fast(self, node: CausalNode, candidates: List[str]) -> Optional[str]:
        """Fast similarity computation with early termination optimizations"""
        if not candidates:
            return None
        
        best_match = None
        best_score = 0
        
        # Early termination if we find a very high confidence match
        high_confidence_threshold = 0.95
        
        for candidate_id in candidates[:20]:  # Limit comparisons for performance
            if candidate_id not in self.nodes:
                continue
                
            candidate_node = self.nodes[candidate_id]
            
            # Fast pre-screening: check type compatibility first
            if node.isIntervention != candidate_node.isIntervention:
                continue
            
            # Quick keyword overlap check
            keyword_overlap = len(set(node.semantic_keywords) & set(candidate_node.semantic_keywords))
            if keyword_overlap == 0:
                continue
                
            # Full similarity computation only for promising candidates
            score = self._compute_node_similarity_fast(node, candidate_node)
            
            self.merge_stats['similarity_computations'] += 1
            
            if score >= high_confidence_threshold:
                # Early termination for very high confidence
                return candidate_id
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _compute_node_similarity(self, node1: CausalNode, node2: CausalNode) -> float:
        """Comprehensive node similarity computation"""
        
        # Text similarity (primary)
        text_sim = SequenceMatcher(None, node1.canonical_text, node2.canonical_text).ratio()
        
        # Keyword similarity
        keywords1 = set(node1.semantic_keywords)
        keywords2 = set(node2.semantic_keywords)
        if keywords1 or keywords2:
            keyword_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_sim = 0
        
        # Type similarity (intervention vs problem)
        type_sim = 1.0 if node1.isIntervention == node2.isIntervention else 0.5
        
        # Combined similarity with weights
        similarity = (0.6 * text_sim + 0.3 * keyword_sim + 0.1 * type_sim)
        
        return similarity
    
    def _compute_node_similarity_fast(self, node1: CausalNode, node2: CausalNode) -> float:
        """Optimized similarity computation for high-performance matching"""
        
        # Quick keyword similarity (most discriminative for AI safety concepts)
        keywords1 = set(node1.semantic_keywords)
        keywords2 = set(node2.semantic_keywords)
        
        if not keywords1 and not keywords2:
            keyword_sim = 0
        elif keywords1 or keywords2:
            keyword_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_sim = 0
        
        # Early exit if keyword similarity is very low
        if keyword_sim < 0.1:
            return keyword_sim
        
        # Fast text similarity using length and character overlap
        text1, text2 = node1.canonical_text, node2.canonical_text
        
        # Length-based similarity (fast approximation)
        len_sim = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        
        # Character overlap similarity (faster than full sequence matching)
        chars1, chars2 = set(text1), set(text2)
        char_sim = len(chars1 & chars2) / len(chars1 | chars2) if (chars1 or chars2) else 0
        
        # Type similarity
        type_sim = 1.0 if node1.isIntervention == node2.isIntervention else 0.5
        
        # Weighted combination optimized for speed
        similarity = (0.5 * keyword_sim + 0.3 * char_sim + 0.1 * len_sim + 0.1 * type_sim)
        
        return similarity
    
    def _find_semantic_match(self, node: CausalNode) -> Optional[str]:
        """Use TF-IDF for semantic matching (expensive, use sparingly)"""
        if not self.concept_vectors is not None:
            return None
        
        # This would require maintaining TF-IDF vectors - implement if needed for very high precision
        # For now, return None to avoid O(n²) complexity
        return None
    
    def _find_semantic_match_optimized(self, node: CausalNode) -> Optional[str]:
        """Optimized semantic matching with sampling and early termination"""
        
        # Only use for small graphs to maintain performance
        if len(self.nodes) > 1000:
            return None
        
        # Sample-based semantic matching for better performance
        import random
        node_sample = list(self.nodes.keys())
        
        # Sample up to 100 nodes for semantic comparison
        if len(node_sample) > 100:
            node_sample = random.sample(node_sample, 100)
        
        best_match = None
        best_score = 0
        
        for candidate_id in node_sample:
            candidate_node = self.nodes[candidate_id]
            
            # Skip if basic compatibility fails
            if node.isIntervention != candidate_node.isIntervention:
                continue
            
            # Use fast similarity first as a filter
            quick_score = self._compute_node_similarity_fast(node, candidate_node)
            if quick_score < 0.6:  # Only do expensive comparison if promising
                continue
            
            # Full similarity computation for promising candidates
            score = self._compute_node_similarity(node, candidate_node)
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _log_match_timing(self, stage: str, elapsed_time: float):
        """Log timing information for performance analysis"""
        self.performance_stats['stage_timings'][stage].append(elapsed_time)
        self.performance_stats['total_matching_time'] += elapsed_time
    
    def _merge_node_data(self, canonical_id: str, new_node: CausalNode):
        """Intelligently merge node data"""
        if canonical_id not in self.nodes:
            raise ValueError(f"Cannot merge: canonical_id '{canonical_id}' not found in nodes dictionary")
        
        existing_node = self.nodes[canonical_id]
        
        # Merge metadata lists (avoid duplicates)
        existing_node.DOI_URL.extend([url for url in new_node.DOI_URL if url not in existing_node.DOI_URL])
        existing_node.authors.extend([auth for auth in new_node.authors if auth not in existing_node.authors])
        existing_node.institutions.extend([inst for inst in new_node.institutions if inst not in existing_node.institutions])
        existing_node.timestamp.extend([ts for ts in new_node.timestamp if ts not in existing_node.timestamp])
        existing_node.source_papers.extend(new_node.source_papers)
        
        # Merge aliases
        new_aliases = set(new_node.aliases + [new_node.concept_text])
        existing_aliases = set(existing_node.aliases)
        existing_node.aliases = list(existing_aliases | new_aliases)
        
        # Update semantic keywords
        existing_node.semantic_keywords = list(set(existing_node.semantic_keywords + new_node.semantic_keywords))
        
        # Update intervention information if more specific
        if new_node.isIntervention == 1 and existing_node.isIntervention == 0:
            existing_node.isIntervention = 1
            existing_node.stage_in_pipeline = new_node.stage_in_pipeline
            existing_node.maturity_level = new_node.maturity_level
            existing_node.implemented = new_node.implemented
        
        # Track merge history
        existing_node.merge_history.append(f"Merged with {new_node.node_id}")
        
        # Update confidence based on number of sources (keep within 0-1 range)
        existing_node.confidence_score = min(1.0, 0.5 + 0.1 * len(existing_node.source_papers))
        
        # Update indices
        self.node_text_index[new_node.canonical_text] = canonical_id
        self.node_hash_index[new_node.text_hash] = canonical_id
        for keyword in new_node.semantic_keywords:
            self.keyword_index[keyword].add(canonical_id)
    
    def _add_new_node(self, node: CausalNode):
        """Add a completely new node"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **asdict(node))
        
        # Update all indices
        self.node_text_index[node.canonical_text] = node.node_id
        self.node_hash_index[node.text_hash] = node.node_id
        for keyword in node.semantic_keywords:
            self.keyword_index[keyword].add(node.node_id)
    
    def _merge_edges_batch(self, batch: List[dict]):
        """Enhanced edge merging with relationship analysis"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for edge_data in local_graph_data.get('edges', []):
                edge = self._dict_to_edge(edge_data)
                edge.source_papers = [paper_id]
                
                # Resolve node references
                resolved_edge = self._resolve_edge_nodes(edge)
                if resolved_edge:
                    self._add_or_merge_edge(resolved_edge)
    
    def _resolve_edge_nodes(self, edge: CausalEdge) -> Optional[CausalEdge]:
        """Resolve edge node references to canonical node IDs"""
        resolved_sources = []
        resolved_targets = []
        
        for source in edge.source_nodes:
            source_canonical = self._resolve_node_reference(source)
            if source_canonical:
                resolved_sources.append(source_canonical)
        
        for target in edge.target_nodes:
            target_canonical = self._resolve_node_reference(target)
            if target_canonical:
                resolved_targets.append(target_canonical)
        
        if not resolved_sources or not resolved_targets:
            return None
        
        # Create new edge with resolved references
        resolved_edge = CausalEdge(
            DOI_URL=edge.DOI_URL,
            authors=edge.authors,
            institutions=edge.institutions,
            timestamp=edge.timestamp,
            edge_text=edge.edge_text,
            source_nodes=resolved_sources,
            target_nodes=resolved_targets,
            confidence=edge.confidence
        )
        
        resolved_edge.source_papers = edge.source_papers
        resolved_edge.relationship_type = edge.relationship_type
        resolved_edge.relationship_strength = edge.relationship_strength
        
        return resolved_edge
    
    def _resolve_node_reference(self, node_ref: str) -> Optional[str]:
        """Resolve a node reference to canonical ID"""
        # Try exact match first
        node_id = node_ref.replace(" ", "_").upper()
        if node_id in self.nodes:
            return node_id
        
        # Try hash lookup
        canonical_text = re.sub(r'[^\w\s]', '', node_ref.lower().strip())
        text_hash = hashlib.md5(canonical_text.encode()).hexdigest()[:8]
        if text_hash in self.node_hash_index:
            return self.node_hash_index[text_hash]
        
        # Try text lookup
        if canonical_text in self.node_text_index:
            return self.node_text_index[canonical_text]
        
        return None
    
    def _add_or_merge_edge(self, edge: CausalEdge):
        """Add edge or merge with existing similar edge"""
        for source in edge.source_nodes:
            for target in edge.target_nodes:
                edge_key = f"{source}→{target}:{edge.relationship_type}"
                
                # Check for existing similar edges
                existing_edges = self.edges[edge_key]
                merged = False
                
                for existing_edge in existing_edges:
                    if self._edges_are_similar(edge, existing_edge):
                        self._merge_edge_data(existing_edge, edge)
                        merged = True
                        self.merge_stats['edges_merged'] += 1
                        break
                
                if not merged:
                    self.edges[edge_key].append(edge)
                    
                    # Add to NetworkX graph
                    if not self.graph.has_edge(source, target):
                        self.graph.add_edge(source, target, relationships=[])
                    self.graph[source][target]['relationships'].append(edge)
                
                # Update semantic index
                self.edge_semantic_index[edge.semantic_hash].append(edge_key)
    
    def _edges_are_similar(self, edge1: CausalEdge, edge2: CausalEdge) -> bool:
        """Determine if two edges represent the same relationship"""
        
        # Same relationship type
        if edge1.relationship_type != edge2.relationship_type:
            return False
        
        # Similar semantic content
        if edge1.semantic_hash == edge2.semantic_hash:
            return True
        
        # Text similarity
        text_sim = SequenceMatcher(None, edge1.canonical_relationship, edge2.canonical_relationship).ratio()
        return text_sim >= 0.7
    
    def _merge_edge_data(self, existing_edge: CausalEdge, new_edge: CausalEdge):
        """Merge data from two similar edges"""
        # Merge metadata
        existing_edge.DOI_URL.extend([url for url in new_edge.DOI_URL if url not in existing_edge.DOI_URL])
        existing_edge.authors.extend([auth for auth in new_edge.authors if auth not in existing_edge.authors])
        existing_edge.institutions.extend([inst for inst in new_edge.institutions if inst not in existing_edge.institutions])
        existing_edge.timestamp.extend([ts for ts in new_edge.timestamp if ts not in existing_edge.timestamp])
        existing_edge.source_papers.extend(new_edge.source_papers)
        
        # Update confidence (weighted average)
        total_evidence = existing_edge.evidence_count + new_edge.evidence_count
        existing_edge.confidence = int(
            (existing_edge.confidence * existing_edge.evidence_count + 
             new_edge.confidence * new_edge.evidence_count) / total_evidence
        )
        existing_edge.evidence_count = total_evidence
        
        # Update relationship strength
        existing_edge.relationship_strength = (
            existing_edge.relationship_strength + new_edge.relationship_strength
        ) / 2
    
    def _aggregate_evidence(self):
        """Advanced evidence aggregation with domain-specific weighting"""
        logger.info("Aggregating evidence across merged edges...")
        
        evidence_weights = {
            'experimental': 1.2,
            'observational': 1.0, 
            'theoretical': 0.8,
            'conceptual': 0.6
        }
        
        for edge_key, edge_list in self.edges.items():
            if len(edge_list) > 1:
                # Compute weighted evidence
                total_weight = 0
                weighted_confidence = 0
                
                for edge in edge_list:
                    # Infer evidence type from paper metadata and confidence
                    evidence_type = self._infer_evidence_type(edge)
                    weight = evidence_weights.get(evidence_type, 1.0)
                    
                    total_weight += weight * edge.evidence_count
                    weighted_confidence += edge.confidence * weight * edge.evidence_count
                
                # Update primary edge with aggregated confidence
                primary_edge = edge_list[0]
                primary_edge.confidence = int(weighted_confidence / total_weight) if total_weight > 0 else primary_edge.confidence
                primary_edge.evidence_count = sum(e.evidence_count for e in edge_list)
    
    def _infer_evidence_type(self, edge: CausalEdge) -> str:
        """Infer evidence type from edge characteristics"""
        # Simple heuristic - in production, use more sophisticated analysis
        if edge.confidence >= 4 and any(inst in ['MIT', 'Stanford', 'OpenAI'] for inst in edge.institutions):
            return 'experimental'
        elif edge.confidence >= 3:
            return 'observational'
        else:
            return 'theoretical'
    
    def _filter_low_confidence_edges(self):
        """Remove edges below confidence threshold"""
        initial_count = sum(len(edges) for edges in self.edges.values())
        
        filtered_edges = {}
        for edge_key, edge_list in self.edges.items():
            high_conf_edges = [edge for edge in edge_list if edge.confidence >= self.confidence_threshold]
            if high_conf_edges:
                filtered_edges[edge_key] = high_conf_edges
        
        self.edges = filtered_edges
        final_count = sum(len(edges) for edges in self.edges.values())
        
        logger.info(f"Filtered {initial_count - final_count} low-confidence edges")
    
    def _compute_final_statistics(self):
        """Compute comprehensive final statistics"""
        # Compute average timing by stage
        for stage, timings in self.performance_stats['stage_timings'].items():
            if timings:
                self.performance_stats['average_match_time_by_stage'][stage] = np.mean(timings) * 1000  # ms
        
        self.final_stats = {
            'total_nodes': len(self.nodes),
            'total_unique_edges': len(self.edges),
            'total_evidence_pieces': sum(sum(e.evidence_count for e in edges) for edges in self.edges.values()),
            'merge_efficiency': {
                'nodes_merged': self.merge_stats['nodes_merged'],
                'edges_merged': self.merge_stats['edges_merged'],
                'similarity_computations': self.merge_stats['similarity_computations'],
                'exact_matches': self.merge_stats['exact_matches'],
                'semantic_matches': self.merge_stats['semantic_matches']
            },
            'stage_performance': {
                'stage_1_hash_matches': self.merge_stats['stage_1_matches'],
                'stage_2_canonical_matches': self.merge_stats['stage_2_matches'],
                'stage_3_keyword_matches': self.merge_stats['stage_3_matches'],
                'stage_4_semantic_matches': self.merge_stats['stage_4_matches'],
                'stage_4_skipped_for_scale': self.merge_stats['stage_4_skipped'],
                'total_matching_time_seconds': self.performance_stats['total_matching_time'],
                'average_timings_ms': self.performance_stats['average_match_time_by_stage']
            }
        }
    
    def _print_merge_summary(self):
        """Print comprehensive merge summary with enhanced performance metrics"""
        print("\n" + "="*80)
        print("ENHANCED MULTI-STAGE CAUSAL GRAPH MERGE SUMMARY")
        print("="*80)
        print(f"Final Graph Size:")
        print(f"  - Nodes: {self.final_stats['total_nodes']:,}")
        print(f"  - Unique Relationships: {self.final_stats['total_unique_edges']:,}")
        print(f"  - Total Evidence Pieces: {self.final_stats['total_evidence_pieces']:,}")
        
        print(f"\nMerge Efficiency:")
        merge_eff = self.final_stats['merge_efficiency']
        print(f"  - Nodes Merged: {merge_eff['nodes_merged']:,}")
        print(f"  - Edges Merged: {merge_eff['edges_merged']:,}")
        print(f"  - Total Similarity Computations: {merge_eff['similarity_computations']:,}")
        
        print(f"\nMulti-Stage Matching Performance:")
        stage_perf = self.final_stats['stage_performance']
        print(f"  - Stage 1 (Hash O(1)): {stage_perf['stage_1_hash_matches']:,} matches")
        print(f"  - Stage 2 (Canonical O(1)): {stage_perf['stage_2_canonical_matches']:,} matches")
        print(f"  - Stage 3 (Keyword O(k)): {stage_perf['stage_3_keyword_matches']:,} matches")
        print(f"  - Stage 4 (Semantic O(n)): {stage_perf['stage_4_semantic_matches']:,} matches")
        print(f"  - Stage 4 Skipped (Scale): {stage_perf['stage_4_skipped_for_scale']:,} nodes")
        
        # Calculate stage efficiency
        total_matches = (stage_perf['stage_1_hash_matches'] + stage_perf['stage_2_canonical_matches'] + 
                        stage_perf['stage_3_keyword_matches'] + stage_perf['stage_4_semantic_matches'])
        if total_matches > 0:
            stage1_pct = (stage_perf['stage_1_hash_matches'] / total_matches) * 100
            stage2_pct = (stage_perf['stage_2_canonical_matches'] / total_matches) * 100
            stage3_pct = (stage_perf['stage_3_keyword_matches'] / total_matches) * 100
            stage4_pct = (stage_perf['stage_4_semantic_matches'] / total_matches) * 100
            
            print(f"\nMatching Efficiency Distribution:")
            print(f"  - Fast O(1) Hash Matches: {stage1_pct:.1f}%")
            print(f"  - Fast O(1) Text Matches: {stage2_pct:.1f}%")
            print(f"  - Medium O(k) Keyword Matches: {stage3_pct:.1f}%")
            print(f"  - Slow O(n) Semantic Matches: {stage4_pct:.1f}%")
        
        print(f"\nPerformance Timings:")
        print(f"  - Total Matching Time: {stage_perf['total_matching_time_seconds']:.3f} seconds")
        
        avg_timings = stage_perf['average_timings_ms']
        if avg_timings:
            print(f"  - Average Stage Timings (ms):")
            for stage, time_ms in avg_timings.items():
                print(f"    • {stage}: {time_ms:.3f}ms")
        
        if self.nodes:
            interventions = sum(1 for node in self.nodes.values() if node.isIntervention == 1)
            print(f"\nContent Analysis:")
            print(f"  - Intervention Nodes: {interventions}")
            print(f"  - Problem/Concept Nodes: {len(self.nodes) - interventions}")
        
        print("="*80)
    
    def _dict_to_node(self, node_data: dict) -> CausalNode:
        """Convert dictionary to CausalNode object"""
        return CausalNode(**{k: v for k, v in node_data.items() if k in CausalNode.__dataclass_fields__})
    
    def _dict_to_edge(self, edge_data: dict) -> CausalEdge:
        """Convert dictionary to CausalEdge object"""
        return CausalEdge(**{k: v for k, v in edge_data.items() if k in CausalEdge.__dataclass_fields__})

    def visualize_with_edge_details(self, figsize=(20, 16), show_edge_labels=False):
        """Enhanced visualization with optional edge labels"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create visualization.")
            return
            
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=150, seed=42)
        
        # Enhanced node styling
        node_colors = []
        node_sizes = []
        node_alphas = []
        
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            
            # Size by evidence strength (sources + connections)
            evidence_strength = len(node.source_papers) + self.graph.degree(node_id)
            node_sizes.append(800 + evidence_strength * 200)
            
            # Alpha by confidence (ensure it stays within 0-1 range)
            alpha_value = 0.6 + 0.4 * min(1.0, node.confidence_score)
            node_alphas.append(min(1.0, max(0.3, alpha_value)))
            
            # Color by type and implementation
            if node.isIntervention == 1:
                if node.implemented == 1:
                    node_colors.append('darkgreen')
                elif node.maturity_level and node.maturity_level >= 3:
                    node_colors.append('mediumseagreen')
                else:
                    node_colors.append('lightgreen')
            else:
                # Categorize problems by keywords
                keywords = set(node.semantic_keywords)
                if {'deception', 'hacking', 'misalignment'} & keywords:
                    node_colors.append('darkred')
                elif {'oversight', 'evaluation', 'scalability'} & keywords:
                    node_colors.append('orange')
                elif {'interpretability', 'detection', 'monitoring'} & keywords:
                    node_colors.append('gold')
                else:
                    node_colors.append('lightblue')
        
        # Draw nodes with varying transparency
        for i, node_id in enumerate(self.graph.nodes()):
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[node_id],
                                 node_color=[node_colors[i]], node_size=[node_sizes[i]],
                                 alpha=node_alphas[i], edgecolors='black', linewidths=1.5)
        
        # Enhanced edge drawing with relationship types
        self._draw_relationship_edges(pos)
        
        # Node labels with evidence indicators
        labels = {}
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            concept = node.concept_text
            evidence_count = len(node.source_papers)
            
            # Truncate long labels and add evidence indicator
            if len(concept) > 30:
                words = concept.split()
                label = ' '.join(words[:4]) + '...' if len(words) > 4 else concept
            else:
                label = concept
            
            if evidence_count > 1:
                label += f" [{evidence_count}]"
            
            labels[node_id] = label
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold')
        
        # Optional edge labels (for smaller graphs)
        if show_edge_labels and len(self.graph.edges()) < 50:
            edge_labels = {}
            for edge_key, edge_list in list(self.edges.items())[:20]:  # Limit to prevent clutter
                if edge_list:
                    primary_edge = edge_list[0]
                    source_target = edge_key.split(':')[0]
                    if '→' in source_target:
                        source, target = source_target.split('→')
                        if self.graph.has_edge(source, target):
                            rel_type = primary_edge.relationship_type
                            confidence = primary_edge.confidence
                            edge_labels[(source, target)] = f"{rel_type}\n(conf:{confidence})"
            
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.title("Enhanced AI Safety Causal Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Enhanced legend
        self._draw_comprehensive_legend()
        
        plt.tight_layout()
        plt.show()
    
    def _draw_relationship_edges(self, pos):
        """Draw edges with relationship-specific styling"""
        relationship_styles = {
            'causes': {'color': 'darkblue', 'style': 'solid', 'alpha': 0.8},
            'prevents': {'color': 'red', 'style': 'dashed', 'alpha': 0.7},
            'enables': {'color': 'green', 'style': 'solid', 'alpha': 0.7},
            'moderates': {'color': 'purple', 'style': 'dotted', 'alpha': 0.6},
            'necessitates': {'color': 'orange', 'style': 'solid', 'alpha': 0.8},
            'correlates': {'color': 'gray', 'style': 'dotted', 'alpha': 0.5}
        }
        
        # Group edges by relationship type and confidence
        relationship_edges = defaultdict(lambda: defaultdict(list))
        
        for edge_key, edge_list in self.edges.items():
            if edge_list:
                primary_edge = edge_list[0]
                source_target = edge_key.split(':')[0]
                if '→' in source_target:
                    source, target = source_target.split('→')
                    if self.graph.has_edge(source, target):
                        rel_type = primary_edge.relationship_type
                        confidence = primary_edge.confidence
                        relationship_edges[rel_type][confidence].append((source, target))
        
        # Draw each relationship type
        for rel_type, confidence_groups in relationship_edges.items():
            style = relationship_styles.get(rel_type, relationship_styles['causes'])
            
            for confidence, edge_list in confidence_groups.items():
                width = max(1, confidence * 0.8)
                alpha = style['alpha'] * (0.5 + 0.1 * confidence)
                
                nx.draw_networkx_edges(
                    self.graph, pos, edgelist=edge_list,
                    edge_color=style['color'], width=width,
                    alpha=alpha, arrows=True, arrowsize=20,
                    arrowstyle='->', style=style['style']
                )
    
    def _draw_comprehensive_legend(self):
        """Draw comprehensive legend with all visual elements"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            # Node types
            Patch(facecolor='darkgreen', label='Implemented Solutions'),
            Patch(facecolor='mediumseagreen', label='Mature Solutions'),
            Patch(facecolor='lightgreen', label='Proposed Solutions'),
            Patch(facecolor='darkred', label='Critical AI Safety Problems'),
            Patch(facecolor='orange', label='Oversight & Evaluation Challenges'),
            Patch(facecolor='gold', label='Interpretability & Detection'),
            Patch(facecolor='lightblue', label='Other Concepts'),
            
            # Divider
            Line2D([0], [0], color='white', linewidth=0, label=''),
            
            # Relationship types
            Line2D([0], [0], color='darkblue', linewidth=3, label='Causal Relationships'),
            Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Prevention/Mitigation'),
            Line2D([0], [0], color='green', linewidth=3, label='Enabling Relationships'),
            Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Moderation'),
            Line2D([0], [0], color='orange', linewidth=3, label='Necessity'),
            Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Correlation'),
        ]
        
        # Add note about evidence indicators
        legend_elements.append(Line2D([0], [0], color='white', linewidth=0, 
                                    label='[n] = Evidence from n papers'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                  frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    def export_enhanced_json(self, filename: str = None) -> dict:
        """Export with enhanced metadata and merge information"""
        export_data = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "merge_statistics": self.final_stats,
                "similarity_threshold": self.similarity_threshold,
                "confidence_threshold": self.confidence_threshold,
                "graph_properties": {
                    "density": nx.density(self.graph) if self.graph.nodes() else 0,
                    "strongly_connected_components": nx.number_strongly_connected_components(self.graph) if self.graph.nodes() else 0,
                    "weakly_connected_components": nx.number_weakly_connected_components(self.graph) if self.graph.nodes() else 0
                }
            },
            "nodes": [],
            "edges": [],
            "merge_provenance": {
                "node_merge_history": {},
                "edge_evidence_aggregation": {}
            }
        }
        
        # Export nodes with enhanced metadata
        for node_id, node in self.nodes.items():
            node_dict = asdict(node)
            node_dict['graph_metrics'] = {
                'degree': self.graph.degree(node_id) if node_id in self.graph else 0,
                'betweenness_centrality': 0,  # Computed separately if needed
                'evidence_strength': len(node.source_papers)
            }
            export_data["nodes"].append(node_dict)
            
            if node.merge_history:
                export_data["merge_provenance"]["node_merge_history"][node_id] = node.merge_history
        
        # Export edges with aggregation information
        for edge_key, edge_list in self.edges.items():
            for edge in edge_list:
                edge_dict = asdict(edge)
                edge_dict['aggregation_info'] = {
                    'total_evidence_pieces': edge.evidence_count,
                    'relationship_strength': edge.relationship_strength,
                    'semantic_hash': edge.semantic_hash
                }
                export_data["edges"].append(edge_dict)
            
            if len(edge_list) > 1:
                export_data["merge_provenance"]["edge_evidence_aggregation"][edge_key] = {
                    'total_sources': len(set().union(*[e.source_papers for e in edge_list])),
                    'evidence_pieces': len(edge_list)
                }
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Enhanced global graph exported to {filename}")
        
        return export_data
    
    def get_merge_quality_metrics(self) -> dict:
        """Compute quality metrics for the merge process"""
        metrics = {
            'merge_efficiency': {
                'node_reduction_ratio': self.merge_stats['nodes_merged'] / max(1, len(self.nodes) + self.merge_stats['nodes_merged']),
                'edge_consolidation_ratio': self.merge_stats['edges_merged'] / max(1, len(self.edges)),
                'exact_match_percentage': self.merge_stats['exact_matches'] / max(1, self.merge_stats['exact_matches'] + self.merge_stats['semantic_matches']) * 100
            },
            'graph_quality': {
                'average_evidence_per_edge': np.mean([sum(e.evidence_count for e in edges) for edges in self.edges.values()]) if self.edges else 0,
                'concept_coverage': len(set().union(*[node.semantic_keywords for node in self.nodes.values()])) if self.nodes else 0,
                'intervention_ratio': sum(1 for node in self.nodes.values() if node.isIntervention == 1) / max(1, len(self.nodes))
            },
            'semantic_coherence': {
                'avg_confidence': np.mean([edge.confidence for edges in self.edges.values() for edge in edges]) if self.edges else 0,
                'relationship_diversity': len(set(edge.relationship_type for edges in self.edges.values() for edge in edges)) if self.edges else 0
            }
        }
        
        return metrics

# Example usage and testing functions
def create_sample_graph_data() -> List[dict]:
    """Create comprehensive sample graph data for testing the enhanced merger"""
    
    sample_graphs = [
        # Paper 1: Value Alignment and Reward Hacking (EXPANDED - 10 nodes, 12 edges)
        {
            "paper_id": "arxiv:2405.98765",
            "title": "Scalable Alignment for Frontier AI Systems",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "VALUE MISALIGNMENT",
                    "isIntervention": 0,
                    "aliases": ["goal misalignment", "objective misspecification"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "PROXY OBJECTIVES",
                    "isIntervention": 0,
                    "aliases": ["reward proxies", "metric targets"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "REWARD HACKING",
                    "isIntervention": 0,
                    "aliases": ["specification gaming", "goodhart's law"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "DISTRIBUTIONAL SHIFT",
                    "isIntervention": 0,
                    "aliases": ["distribution mismatch", "domain shift"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "LIMITED HUMAN OVERSIGHT",
                    "isIntervention": 0,
                    "aliases": ["oversight constraints", "human bottleneck"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "EVALUATION DIFFICULTIES",
                    "isIntervention": 0,
                    "aliases": ["assessment challenges", "measurement problems"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "SCALABILITY CONSTRAINTS",
                    "isIntervention": 0,
                    "aliases": ["scaling limitations"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "AUTOMATED RED TEAMING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 1,
                    "maturity_level": 2,
                    "implemented": 1,
                    "aliases": ["adversarial testing", "automated evaluation"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "SCALABLE AUTOMATED ALIGNMENT FRAMEWORK (SAAF)",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 3,
                    "implemented": 0,
                    "aliases": ["automated alignment", "scalable oversight"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "concept_text": "ROBUST EVALUATION PROTOCOLS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 3,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["comprehensive testing", "evaluation frameworks"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "VALUE MISALIGNMENT LEADS TO PROXY OBJECTIVES DUE TO IMPERFECT SPECIFICATION",
                    "source_nodes": ["VALUE MISALIGNMENT"],
                    "target_nodes": ["PROXY OBJECTIVES"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "PROXY OBJECTIVES ENABLE REWARD HACKING THROUGH EXPLOITABLE LOOPHOLES",
                    "source_nodes": ["PROXY OBJECTIVES"],
                    "target_nodes": ["REWARD HACKING"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "REWARD HACKING CREATES DISTRIBUTIONAL SHIFT FROM TRAINING TO DEPLOYMENT",
                    "source_nodes": ["REWARD HACKING"],
                    "target_nodes": ["DISTRIBUTIONAL SHIFT"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "DISTRIBUTIONAL SHIFT EXACERBATES LIMITED HUMAN OVERSIGHT",
                    "source_nodes": ["DISTRIBUTIONAL SHIFT"],
                    "target_nodes": ["LIMITED HUMAN OVERSIGHT"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "LIMITED HUMAN OVERSIGHT CAUSES EVALUATION DIFFICULTIES",
                    "source_nodes": ["LIMITED HUMAN OVERSIGHT"],
                    "target_nodes": ["EVALUATION DIFFICULTIES"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "EVALUATION DIFFICULTIES HIGHLIGHT SCALABILITY CONSTRAINTS",
                    "source_nodes": ["EVALUATION DIFFICULTIES"],
                    "target_nodes": ["SCALABILITY CONSTRAINTS"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "AUTOMATED RED TEAMING HELPS DETECT REWARD HACKING BEHAVIORS",
                    "source_nodes": ["AUTOMATED RED TEAMING"],
                    "target_nodes": ["REWARD HACKING"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "SCALABILITY CONSTRAINTS NECESSITATE SAAF DEVELOPMENT",
                    "source_nodes": ["SCALABILITY CONSTRAINTS"],
                    "target_nodes": ["SCALABLE AUTOMATED ALIGNMENT FRAMEWORK (SAAF)"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "SAAF ENABLES ROBUST EVALUATION PROTOCOLS AT SCALE",
                    "source_nodes": ["SCALABLE AUTOMATED ALIGNMENT FRAMEWORK (SAAF)"],
                    "target_nodes": ["ROBUST EVALUATION PROTOCOLS"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "ROBUST EVALUATION PROTOCOLS MITIGATE EVALUATION DIFFICULTIES",
                    "source_nodes": ["ROBUST EVALUATION PROTOCOLS"],
                    "target_nodes": ["EVALUATION DIFFICULTIES"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "AUTOMATED RED TEAMING SUPPORTS ROBUST EVALUATION PROTOCOLS",
                    "source_nodes": ["AUTOMATED RED TEAMING"],
                    "target_nodes": ["ROBUST EVALUATION PROTOCOLS"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2405.98765"],
                    "authors": ["Alice Johnson", "Bob Lee", "Carol Chen"],
                    "institutions": ["MIT AI Lab", "Stanford AI Research", "OpenAI"],
                    "timestamp": ["2024-05-15"],
                    "edge_text": "ROBUST EVALUATION PROTOCOLS HELP ADDRESS VALUE MISALIGNMENT",
                    "source_nodes": ["ROBUST EVALUATION PROTOCOLS"],
                    "target_nodes": ["VALUE MISALIGNMENT"],
                    "confidence": 2
                }
            ]
        },
        
        # Paper 2: Mesa-Optimization and Inner Alignment (EXPANDED - 12 nodes, 15 edges)
        {
            "paper_id": "arxiv:2024.11234",
            "title": "Inner Alignment Failures in Advanced AI Systems",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "OPTIMIZATION PRESSURE",
                    "isIntervention": 0,
                    "aliases": ["selection pressure", "training dynamics"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "MESA OPTIMIZER EMERGENCE",
                    "isIntervention": 0,
                    "aliases": ["inner optimizer", "learned optimization"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "INNER-OUTER OBJECTIVE MISMATCH",
                    "isIntervention": 0,
                    "aliases": ["mesa-objective divergence"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "MODEL COMPLEXITY",
                    "isIntervention": 0,
                    "aliases": ["parameter count", "architectural complexity"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "TRAINING DATA DIVERSITY",
                    "isIntervention": 0,
                    "aliases": ["data distribution", "task variety"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "DECEPTIVE ALIGNMENT",
                    "isIntervention": 0,
                    "aliases": ["treacherous turn", "behavioral deception"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "GRADIENT HACKING",
                    "isIntervention": 0,
                    "aliases": ["training gaming"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "MECHANISTIC INTERPRETABILITY TOOLS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 1,
                    "maturity_level": 2,
                    "implemented": 1,
                    "aliases": ["interpretability", "neural analysis"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "ACTIVATION PATCHING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 1,
                    "maturity_level": 3,
                    "implemented": 1,
                    "aliases": ["causal intervention", "circuit analysis"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "OBJECTIVE ROBUSTNESS TRAINING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 0,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["robust objectives", "anti-mesa training"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "MESA OPTIMIZER DETECTION",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 1,
                    "implemented": 0,
                    "aliases": ["inner optimizer detection", "optimization detection"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "concept_text": "BEHAVIORAL MONITORING SYSTEMS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 3,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["behavior tracking", "deployment monitoring"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "OPTIMIZATION PRESSURE DRIVES MESA OPTIMIZER EMERGENCE IN COMPLEX MODELS",
                    "source_nodes": ["OPTIMIZATION PRESSURE"],
                    "target_nodes": ["MESA OPTIMIZER EMERGENCE"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "MODEL COMPLEXITY INCREASES LIKELIHOOD OF MESA OPTIMIZER EMERGENCE",
                    "source_nodes": ["MODEL COMPLEXITY"],
                    "target_nodes": ["MESA OPTIMIZER EMERGENCE"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "TRAINING DATA DIVERSITY AFFECTS MESA OPTIMIZER EMERGENCE PATTERNS",
                    "source_nodes": ["TRAINING DATA DIVERSITY"],
                    "target_nodes": ["MESA OPTIMIZER EMERGENCE"],
                    "confidence": 2
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "MESA OPTIMIZER EMERGENCE CAUSES INNER-OUTER OBJECTIVE MISMATCH",
                    "source_nodes": ["MESA OPTIMIZER EMERGENCE"],
                    "target_nodes": ["INNER-OUTER OBJECTIVE MISMATCH"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "INNER-OUTER OBJECTIVE MISMATCH ENABLES DECEPTIVE ALIGNMENT STRATEGIES",
                    "source_nodes": ["INNER-OUTER OBJECTIVE MISMATCH"],
                    "target_nodes": ["DECEPTIVE ALIGNMENT"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "DECEPTIVE ALIGNMENT FACILITATES GRADIENT HACKING BEHAVIORS",
                    "source_nodes": ["DECEPTIVE ALIGNMENT"],
                    "target_nodes": ["GRADIENT HACKING"],
                    "confidence": 2
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.11234"],
                    "authors": ["David Smith", "Emma Wilson", "Frank Zhang"],
                    "institutions": ["Anthropic", "DeepMind", "MIRI"],
                    "timestamp": ["2024-06-20"],
                    "edge_text": "MECHANISTIC INTERPRETABILITY TOOLS HELP DETECT MESA OPTIMIZER EMERGENCE",
                    "source_nodes": ["MECHANISTIC INTERPRETABILITY TOOLS"],
                    "target_nodes": ["MESA OPTIMIZER EMERGENCE"],
                    "confidence": 3
                }
            ]
        },
        
        # Paper 3: Constitutional AI and Scalable Oversight (8 nodes, 6 edges)
        {
            "paper_id": "arxiv:2024.56789",
            "title": "Constitutional AI and Scalable Oversight Methods",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "HUMAN FEEDBACK LIMITATIONS",
                    "isIntervention": 0,
                    "aliases": ["rlhf constraints", "human evaluation limits"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "CONSTITUTIONAL AI",
                    "isIntervention": 1,
                    "stage_in_pipeline": 1,
                    "maturity_level": 4,
                    "implemented": 1,
                    "aliases": ["CAI", "principle-based training"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "AI DEBATE FRAMEWORK",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["debate", "adversarial evaluation"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "SCALABLE OVERSIGHT",
                    "isIntervention": 0,
                    "aliases": ["automated oversight"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "AI ASSISTED EVALUATION",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 3,
                    "implemented": 1,
                    "aliases": ["ai evaluation", "automated assessment"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "concept_text": "ALIGNMENT ROBUSTNESS",
                    "isIntervention": 0,
                    "aliases": ["robust alignment"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "edge_text": "HUMAN FEEDBACK LIMITATIONS DRIVE CONSTITUTIONAL AI DEVELOPMENT",
                    "source_nodes": ["HUMAN FEEDBACK LIMITATIONS"],
                    "target_nodes": ["CONSTITUTIONAL AI"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "edge_text": "CONSTITUTIONAL AI ENHANCES SCALABLE OVERSIGHT THROUGH PRINCIPLED EVALUATION",
                    "source_nodes": ["CONSTITUTIONAL AI"],
                    "target_nodes": ["SCALABLE OVERSIGHT"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "edge_text": "AI DEBATE FRAMEWORK CONTRIBUTES TO SCALABLE OVERSIGHT CAPABILITIES",
                    "source_nodes": ["AI DEBATE FRAMEWORK"],
                    "target_nodes": ["SCALABLE OVERSIGHT"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "edge_text": "SCALABLE OVERSIGHT ENABLES AI ASSISTED EVALUATION AT SCALE",
                    "source_nodes": ["SCALABLE OVERSIGHT"],
                    "target_nodes": ["AI ASSISTED EVALUATION"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.56789"],
                    "authors": ["Grace Park", "Henry Liu", "Isabella Martinez"],
                    "institutions": ["Anthropic", "OpenAI", "Redwood Research"],
                    "timestamp": ["2024-07-10"],
                    "edge_text": "AI ASSISTED EVALUATION IMPROVES ALIGNMENT ROBUSTNESS",
                    "source_nodes": ["AI ASSISTED EVALUATION"],
                    "target_nodes": ["ALIGNMENT ROBUSTNESS"],
                    "confidence": 3
                }
            ]
        },
        
        # Paper 4: Capability Control and AI Governance (5 nodes, 4 edges)
        {
            "paper_id": "arxiv:2024.78901",
            "title": "Capability Control Mechanisms for Advanced AI Systems",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "concept_text": "CAPABILITY OVERHANG",
                    "isIntervention": 0,
                    "aliases": ["rapid capability growth"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "concept_text": "OVERSIGHT LAG",
                    "isIntervention": 0,
                    "aliases": ["safety research lag"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "concept_text": "CAPABILITY CONTROL MECHANISMS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 0,
                    "maturity_level": 1,
                    "implemented": 0,
                    "aliases": ["capability limitation", "sandboxing"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "concept_text": "GRADUAL DEPLOYMENT PROTOCOLS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 3,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["staged deployment", "careful rollout"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "concept_text": "ALIGNMENT ROBUSTNESS",
                    "isIntervention": 0,
                    "aliases": ["robust alignment", "alignment verification"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "edge_text": "CAPABILITY OVERHANG CAUSES OVERSIGHT LAG DUE TO RAPID AI PROGRESS",
                    "source_nodes": ["CAPABILITY OVERHANG"],
                    "target_nodes": ["OVERSIGHT LAG"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "edge_text": "OVERSIGHT LAG NECESSITATES CAPABILITY CONTROL MECHANISMS",
                    "source_nodes": ["OVERSIGHT LAG"],
                    "target_nodes": ["CAPABILITY CONTROL MECHANISMS"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "edge_text": "CAPABILITY CONTROL MECHANISMS ENABLE GRADUAL DEPLOYMENT PROTOCOLS",
                    "source_nodes": ["CAPABILITY CONTROL MECHANISMS"],
                    "target_nodes": ["GRADUAL DEPLOYMENT PROTOCOLS"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.78901"],
                    "authors": ["Jack Thompson", "Karen Lee", "Liam Brown"],
                    "institutions": ["FHI", "CHAI", "Partnership on AI"],
                    "timestamp": ["2024-08-01"],
                    "edge_text": "GRADUAL DEPLOYMENT PROTOCOLS IMPROVE ALIGNMENT ROBUSTNESS THROUGH CAREFUL TESTING",
                    "source_nodes": ["GRADUAL DEPLOYMENT PROTOCOLS"],
                    "target_nodes": ["ALIGNMENT ROBUSTNESS"],
                    "confidence": 3
                }
            ]
        },
        
        # Paper 5: Adversarial Robustness and Distributional Shift (8 nodes, 10 edges)
        {
            "paper_id": "arxiv:2024.12345",
            "title": "Adversarial Robustness in Large Language Models Under Distribution Shift",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "concept_text": "ADVERSARIAL EXAMPLES",
                    "isIntervention": 0,
                    "aliases": ["adversarial attacks", "perturbations"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "concept_text": "DISTRIBUTIONAL SHIFT",
                    "isIntervention": 0,
                    "aliases": ["distribution mismatch", "domain shift", "covariate shift"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "concept_text": "ADVERSARIAL TRAINING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 1,
                    "maturity_level": 4,
                    "implemented": 1,
                    "aliases": ["robust training", "adversarial defense"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "concept_text": "ROBUSTNESS EVALUATION",
                    "isIntervention": 1,
                    "stage_in_pipeline": 3,
                    "maturity_level": 3,
                    "implemented": 1,
                    "aliases": ["robustness testing", "evaluation protocols"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "edge_text": "ADVERSARIAL EXAMPLES EXPLOIT DISTRIBUTIONAL SHIFT VULNERABILITIES",
                    "source_nodes": ["ADVERSARIAL EXAMPLES"],
                    "target_nodes": ["DISTRIBUTIONAL SHIFT"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "edge_text": "ADVERSARIAL TRAINING IMPROVES ROBUSTNESS AGAINST ADVERSARIAL EXAMPLES",
                    "source_nodes": ["ADVERSARIAL TRAINING"],
                    "target_nodes": ["ADVERSARIAL EXAMPLES"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.12345"],
                    "authors": ["Michael Chen", "Sarah Kim", "David Rodriguez"],
                    "institutions": ["Google DeepMind", "UC Berkeley", "MIT CSAIL"],
                    "timestamp": ["2024-09-15"],
                    "edge_text": "ROBUSTNESS EVALUATION VALIDATES ADVERSARIAL TRAINING EFFECTIVENESS",
                    "source_nodes": ["ROBUSTNESS EVALUATION"],
                    "target_nodes": ["ADVERSARIAL TRAINING"],
                    "confidence": 3
                }
            ]
        },
        
        # Paper 6: AI Governance and Policy (6 nodes, 6 edges)
        {
            "paper_id": "arxiv:2024.98765",
            "title": "Governance Frameworks for Advanced AI Systems",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "concept_text": "AI GOVERNANCE FRAMEWORKS",
                    "isIntervention": 1,
                    "stage_in_pipeline": 0,
                    "maturity_level": 2,
                    "implemented": 1,
                    "aliases": ["governance", "regulation", "policy frameworks"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "concept_text": "REGULATORY COMPLIANCE",
                    "isIntervention": 1,
                    "stage_in_pipeline": 3,
                    "maturity_level": 3,
                    "implemented": 1,
                    "aliases": ["compliance", "regulatory requirements"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "concept_text": "ALGORITHMIC AUDITING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 2,
                    "implemented": 0,
                    "aliases": ["algorithm audits", "system auditing"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "concept_text": "OVERSIGHT LAG",
                    "isIntervention": 0,
                    "aliases": ["regulatory lag", "governance gap"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "edge_text": "AI GOVERNANCE FRAMEWORKS ESTABLISH REGULATORY COMPLIANCE REQUIREMENTS",
                    "source_nodes": ["AI GOVERNANCE FRAMEWORKS"],
                    "target_nodes": ["REGULATORY COMPLIANCE"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "edge_text": "REGULATORY COMPLIANCE NECESSITATES ALGORITHMIC AUDITING",
                    "source_nodes": ["REGULATORY COMPLIANCE"],
                    "target_nodes": ["ALGORITHMIC AUDITING"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.98765"],
                    "authors": ["Jennifer Walsh", "Alexander Turner", "Maria Santos"],
                    "institutions": ["Partnership on AI", "GovAI", "EU AI Act Office"],
                    "timestamp": ["2024-10-01"],
                    "edge_text": "OVERSIGHT LAG NECESSITATES IMPROVED AI GOVERNANCE FRAMEWORKS",
                    "source_nodes": ["OVERSIGHT LAG"],
                    "target_nodes": ["AI GOVERNANCE FRAMEWORKS"],
                    "confidence": 4
                }
            ]
        },
        
        # Paper 7: Emergent Capabilities and Risk Assessment (5 nodes, 5 edges)
        {
            "paper_id": "arxiv:2024.55555",
            "title": "Emergent Capabilities in Large Language Models: Detection and Risk Management",
            "nodes": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "concept_text": "EMERGENT CAPABILITIES",
                    "isIntervention": 0,
                    "aliases": ["emergence", "capability emergence", "phase transitions"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "concept_text": "CAPABILITY OVERHANG",
                    "isIntervention": 0,
                    "aliases": ["latent capabilities", "hidden capabilities"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "concept_text": "CAPABILITY MONITORING",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 2,
                    "implemented": 1,
                    "aliases": ["monitoring", "capability tracking"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "concept_text": "EMERGENT CAPABILITY DETECTION",
                    "isIntervention": 1,
                    "stage_in_pipeline": 2,
                    "maturity_level": 1,
                    "implemented": 0,
                    "aliases": ["emergence detection", "capability detection"]
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "concept_text": "SCALING LAWS",
                    "isIntervention": 0,
                    "aliases": ["power laws", "scaling relationships"]
                }
            ],
            "edges": [
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "edge_text": "EMERGENT CAPABILITIES CREATE CAPABILITY OVERHANG RISKS",
                    "source_nodes": ["EMERGENT CAPABILITIES"],
                    "target_nodes": ["CAPABILITY OVERHANG"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "edge_text": "CAPABILITY MONITORING HELPS DETECT EMERGENT CAPABILITIES",
                    "source_nodes": ["CAPABILITY MONITORING"],
                    "target_nodes": ["EMERGENT CAPABILITIES"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "edge_text": "EMERGENT CAPABILITY DETECTION TARGETS CAPABILITY OVERHANG",
                    "source_nodes": ["EMERGENT CAPABILITY DETECTION"],
                    "target_nodes": ["CAPABILITY OVERHANG"],
                    "confidence": 4
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "edge_text": "SCALING LAWS PREDICT EMERGENT CAPABILITIES",
                    "source_nodes": ["SCALING LAWS"],
                    "target_nodes": ["EMERGENT CAPABILITIES"],
                    "confidence": 3
                },
                {
                    "DOI_URL": ["https://arxiv.org/abs/2024.55555"],
                    "authors": ["Rachel Green", "Tom Wilson", "Lisa Park"],
                    "institutions": ["Anthropic", "OpenAI", "Stanford HAI"],
                    "timestamp": ["2024-11-01"],
                    "edge_text": "CAPABILITY MONITORING INCORPORATES SCALING LAWS",
                    "source_nodes": ["CAPABILITY MONITORING"],
                    "target_nodes": ["SCALING LAWS"],
                    "confidence": 2
                }
            ]
        }
    ]
    
    return sample_graphs

def test_enhanced_merger():
    """Test the enhanced merger with sample data"""
    print("Testing Enhanced Causal Graph Merger...")
    
    # Create test data
    sample_graphs = create_sample_graph_data()
    
    # Initialize enhanced merger
    merger = EnhancedGlobalCausalGraph(similarity_threshold=0.75)
    
    # Merge graphs
    merger.merge_local_graphs(sample_graphs)
    
    # Get quality metrics
    quality_metrics = merger.get_merge_quality_metrics()
    print(f"\nMerge Quality Metrics:")
    for category, metrics in quality_metrics.items():
        print(f"  {category}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")
    
    # Visualize results
    if merger.nodes:
        merger.visualize_with_edge_details(show_edge_labels=True)
    
    # Export results
    export_data = merger.export_enhanced_json("test_enhanced_merge.json")
    
    return merger

def visualize_local_graph(local_graph_data: dict, title: str = "Local Graph"):
    """Visualize a single local graph"""
    if not MATPLOTLIB_AVAILABLE:
        print(f"Cannot visualize {title} - matplotlib not available")
        return
    
    # Create a temporary graph for visualization
    temp_graph = nx.DiGraph()
    node_colors = []
    node_labels = {}
    
    # Add nodes
    for node_data in local_graph_data.get('nodes', []):
        node_id = node_data['concept_text']
        temp_graph.add_node(node_id)
        
        # Color by intervention type
        if node_data.get('isIntervention', 0) == 1:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightcoral')
        
        # Truncate long labels
        if len(node_id) > 25:
            node_labels[node_id] = node_id[:22] + "..."
        else:
            node_labels[node_id] = node_id
    
    # Add edges
    for edge_data in local_graph_data.get('edges', []):
        for source in edge_data['source_nodes']:
            for target in edge_data['target_nodes']:
                temp_graph.add_edge(source, target)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(temp_graph, seed=42)
    
    nx.draw_networkx_nodes(temp_graph, pos, node_color=node_colors, 
                          node_size=2000, alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(temp_graph, pos, edge_color='darkblue', 
                          arrows=True, arrowsize=20, alpha=0.6)
    nx.draw_networkx_labels(temp_graph, pos, node_labels, font_size=10, font_weight='bold')
    
    plt.title(f"{title}\nNodes: {len(temp_graph.nodes())}, Edges: {len(temp_graph.edges())}", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Add simple legend
    if MATPLOTLIB_AVAILABLE:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Intervention Nodes'),
            Patch(facecolor='lightcoral', label='Problem/Concept Nodes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def test_enhanced_merger_with_visuals():
    """Test the enhanced merger with full visualization"""
    print("Testing Enhanced Causal Graph Merger with Visualizations...")
    
    # Create test data
    sample_graphs = create_sample_graph_data()
    
    # Visualize individual local graphs
    print("\n📊 Visualizing Local Graphs:")
    for i, local_graph in enumerate(sample_graphs, 1):
        title = f"Local Graph {i}: {local_graph.get('title', 'Unknown')}"
        nodes_count = len(local_graph.get('nodes', []))
        edges_count = len(local_graph.get('edges', []))
        print(f"  - {title} (Nodes: {nodes_count}, Edges: {edges_count})")
        
        # Debug: Print node details
        for j, node in enumerate(local_graph.get('nodes', []), 1):
            print(f"    Node {j}: {node['concept_text']} (Intervention: {node.get('isIntervention', 0)})")
        
        visualize_local_graph(local_graph, title)
    
    # Initialize enhanced merger
    merger = EnhancedGlobalCausalGraph(similarity_threshold=0.75)
    
    # Merge graphs
    try:
        print("\n🔄 Merging graphs...")
        merger.merge_local_graphs(sample_graphs)
        print("✅ Merge completed successfully!")
        
        # Get quality metrics
        quality_metrics = merger.get_merge_quality_metrics()
        print(f"\nMerge Quality Metrics:")
        for category, metrics in quality_metrics.items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        # Visualize merged result
        print("\n📊 Visualizing Merged Global Graph:")
        if merger.nodes:
            print(f"Final merged graph contains {len(merger.nodes)} nodes:")
            for node_id, node in merger.nodes.items():
                sources = len(node.source_papers)
                print(f"  - {node.concept_text} (ID: {node_id}, Sources: {sources}, Intervention: {node.isIntervention})")
            
            merger.visualize_with_edge_details(show_edge_labels=True, figsize=(16, 12))
        else:
            print("No nodes in merged graph to visualize")
        
        # Export results
        export_data = merger.export_enhanced_json("test_enhanced_merge.json")
        print(f"\n💾 Exported merged graph to test_enhanced_merge.json")
        
        return merger
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise

def test_core_merger_logic():
    """Test the core merger logic without visualization"""
    print("Testing Core Enhanced Causal Graph Merger Logic...")
    
    # Create test data
    sample_graphs = create_sample_graph_data()
    
    # Initialize enhanced merger
    merger = EnhancedGlobalCausalGraph(similarity_threshold=0.75)
    
    # Merge graphs
    try:
        merger.merge_local_graphs(sample_graphs)
        print("✅ Merge completed successfully!")
        
        # Get quality metrics
        quality_metrics = merger.get_merge_quality_metrics()
        print(f"\nMerge Quality Metrics:")
        for category, metrics in quality_metrics.items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        return merger
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise

if __name__ == "__main__":
    # Choose test based on matplotlib availability
    if MATPLOTLIB_AVAILABLE:
        print("🎨 Running enhanced merger with full visualizations...")
        test_merger = test_enhanced_merger_with_visuals()
    else:
        print("⚙️ Running core merger logic (no visualizations)...")
        test_merger = test_core_merger_logic()