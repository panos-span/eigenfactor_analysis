#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Citation Network Analysis for Detecting Citation Cliques

This script analyzes academic citation networks to identify potentially
problematic citation patterns among authors, with a focus on detecting
cliques and citation rings among low-eigenfactor ("bottom") authors.

The script works with two SQLite databases:
- ROLAP_DB: Contains author and publication data
- IMPACT_DB: Contains citation data

Author: Panagiotis Spanakis
Date: March 2025
"""

import sqlite3
import networkx as nx
import csv
import traceback
import statistics
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque, Counter
from pathlib import Path
from tqdm import tqdm

# Database paths (adjust to your environment)
ROLAP_DB = "rolap.db"
IMPACT_DB = "impact.db"

# Analysis parameters (can be adjusted)
MAX_BFS_NODES = 10000  # Cap the BFS expansion to prevent explosion
DEFAULT_DEPTH = 6      # Depth of BFS traversal
SUSPICIOUS_CLIQUE_MIN_SIZE = 3  # Minimum size of suspicious cliques
SUSPICIOUS_CLIQUE_MIN_DENSITY = 0.7  # Minimum density for suspicious cliques
SUSPICIOUS_CLIQUE_MIN_BAD_FRACTION = 0.7  # Minimum fraction of bad authors
CITATION_RING_MIN_SIZE = 3  # Minimum size of citation rings
CITATION_RING_MAX_SIZE = 12  # Maximum size of citation rings
CITATION_RING_MIN_BAD_FRACTION = 0.6  # Minimum fraction of bad authors in ring

# Output directory
OUTPUT_DIR = Path("output_new_bad_0.2_temporal")


##############################################################################
# 1) Database Schema Detection
##############################################################################
def detect_schema(rolap_conn, impact_conn):
    """
    Detect available tables and columns in the databases.
    This helps make the code more resilient to different database schemas.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        
    Returns:
        dict: Available tables and columns
    """
    print("[INFO] Detecting database schema...")
    
    schema = {
        'rolap': {'tables': {}, 'has_works': False, 'has_author_metrics': False},
        'impact': {'tables': {}, 'has_work_references': False}
    }
    
    # Check ROLAP tables
    r_cursor = rolap_conn.cursor()
    try:
        r_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in r_cursor.fetchall()]
        schema['rolap']['tables_list'] = tables
        
        for table in tables:
            try:
                r_cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in r_cursor.fetchall()]
                schema['rolap']['tables'][table] = columns
                
                # Check for specific tables
                if table == 'works_orcid':
                    schema['rolap']['has_works_orcid'] = True
                    schema['rolap']['works_orcid_columns'] = columns
                
                if table == 'orcid_h5_bottom':
                    schema['rolap']['has_orcid_h5_bottom'] = True
                    
                if table == 'works':
                    schema['rolap']['has_works'] = True
                    schema['rolap']['works_columns'] = columns
                
                if table == 'author_metrics':
                    schema['rolap']['has_author_metrics'] = True
                
                if table == 'matched_authors_with_counts':
                    schema['rolap']['has_matched_authors'] = True
                    
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        print("[WARNING] Could not detect ROLAP schema.")
    
    # Check IMPACT tables
    i_cursor = impact_conn.cursor()
    try:
        i_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in i_cursor.fetchall()]
        schema['impact']['tables_list'] = tables
        
        for table in tables:
            try:
                i_cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in i_cursor.fetchall()]
                schema['impact']['tables'][table] = columns
                
                # Check for specific tables
                if table == 'work_references':
                    schema['impact']['has_work_references'] = True
                    schema['impact']['work_references_columns'] = columns
                
                if table == 'works':
                    schema['impact']['has_works'] = True
                    schema['impact']['works_columns'] = columns
                    
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        print("[WARNING] Could not detect IMPACT schema.")
    
    r_cursor.close()
    i_cursor.close()
    
    # Print schema summary
    print(f"[INFO] ROLAP tables: {', '.join(schema['rolap'].get('tables_list', []))}")
    print(f"[INFO] IMPACT tables: {', '.join(schema['impact'].get('tables_list', []))}")
    
    # Check for required tables and print warnings
    if not schema['rolap'].get('has_works_orcid', False):
        print("[WARNING] works_orcid table not found in ROLAP. Author-paper connections will be limited.")
    
    if not schema['rolap'].get('has_orcid_h5_bottom', False):
        print("[WARNING] orcid_h5_bottom table not found in ROLAP. Bad author detection will be limited.")
        
    if not schema['impact'].get('has_work_references', False):
        print("[WARNING] work_references table not found in IMPACT. Citation analysis will be limited.")
    
    return schema


##############################################################################
# 2) Data Loading Functions
##############################################################################
def load_bad_authors(rolap_conn, schema):
    """
    Load the set of "bad" authors (those with low eigenfactor scores).
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        
    Returns:
        set: Set of ORCID strings for bad authors
    """
    bad_authors = set()
    
    if schema['rolap'].get('has_orcid_h5_bottom', False):
        try:
            cursor = rolap_conn.cursor()
            cursor.execute("SELECT orcid FROM orcid_h5_bottom WHERE orcid IS NOT NULL")
            for (orcid,) in cursor.fetchall():
                if orcid:  # Ensure we don't add None values
                    bad_authors.add(orcid)
            cursor.close()
            print(f"[INFO] Loaded {len(bad_authors)} bad authors from orcid_h5_bottom.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load bad authors: {e}")
    else:
        print("[WARNING] orcid_h5_bottom table not found. No bad authors loaded.")
    
    return bad_authors


def load_publication_years(impact_conn, schema):
    """
    Load publication years for works from the impact database.
    
    Args:
        impact_conn: SQLite connection to IMPACT database
        schema: Database schema information
        
    Returns:
        dict: Dictionary mapping work IDs to publication years
    """
    publication_years = {}
    
    if schema['impact'].get('has_works', False):
        works_columns = schema['impact'].get('works_columns', [])
        
        # Check if year column exists
        if 'year' in works_columns:
            try:
                cursor = impact_conn.cursor()
                cursor.execute("SELECT id, year FROM works WHERE year IS NOT NULL")
                for work_id, year in cursor.fetchall():
                    try:
                        # Store the year as an integer
                        publication_years[work_id] = int(year)
                    except (ValueError, TypeError):
                        # Skip if year cannot be parsed
                        continue
                cursor.close()
                print(f"[INFO] Loaded {len(publication_years)} publication years from impact database.")
            except sqlite3.OperationalError as e:
                print(f"[WARNING] Could not load publication years from impact database: {e}")
        else:
            print("[WARNING] 'year' column not found in works table in impact database.")
    else:
        print("[WARNING] works table not found in impact database. No publication years loaded.")
    
    return publication_years


def load_author_metrics(rolap_conn, schema):
    """
    Load additional metrics for authors (h-index, citation counts, etc.).
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        
    Returns:
        dict: Dictionary mapping ORCIDs to author metrics
    """
    author_metrics = {}
    
    if schema['rolap'].get('has_author_metrics', False):
        try:
            cursor = rolap_conn.cursor()
            cursor.execute("""
                SELECT orcid, h_index, total_citations, num_works
                FROM author_metrics
                WHERE orcid IS NOT NULL
            """)
            for orcid, h_idx, citations, works in cursor.fetchall():
                author_metrics[orcid] = {
                    'h_index': h_idx if h_idx is not None else 0,
                    'total_citations': citations if citations is not None else 0,
                    'num_works': works if works is not None else 0
                }
            cursor.close()
            print(f"[INFO] Loaded metrics for {len(author_metrics)} authors.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load author metrics: {e}")
    else:
        print("[WARNING] author_metrics table not found. No author metrics loaded.")
    
    return author_metrics


def load_work_author_mappings(rolap_conn, schema):
    """
    Load mappings between works and authors.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        
    Returns:
        tuple: (orcid_to_works, work_to_authors)
            - orcid_to_works: Dictionary mapping ORCIDs to lists of work IDs
            - work_to_authors: Dictionary mapping work IDs to lists of ORCIDs
    """
    orcid_to_works = {}
    work_to_authors = {}
    
    if schema['rolap'].get('has_works_orcid', False):
        try:
            cursor = rolap_conn.cursor()
            cursor.execute("""
                SELECT orcid, id
                FROM works_orcid
                WHERE orcid IS NOT NULL AND id IS NOT NULL
            """)
            
            for orcid, work_id in tqdm(cursor.fetchall(), desc="Loading work-author mappings"):
                orcid_to_works.setdefault(orcid, []).append(work_id)
                work_to_authors.setdefault(work_id, []).append(orcid)
                
            cursor.close()
            print(f"[INFO] Loaded {len(orcid_to_works)} author->works mappings and {len(work_to_authors)} work->authors mappings.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load work-author mappings: {e}")
    else:
        print("[WARNING] works_orcid table not found. No work-author mappings loaded.")
    
    return orcid_to_works, work_to_authors


def load_citation_data(impact_conn, schema):
    """
    Load citation relationships between works.
    
    Args:
        impact_conn: SQLite connection to IMPACT database
        schema: Database schema information
        
    Returns:
        tuple: (work_to_cited, citation_years)
            - work_to_cited: Dictionary mapping work IDs to lists of cited work IDs
            - citation_years: Dictionary mapping (source_id, target_id) to citation years
    """
    work_to_cited = {}
    citation_years = {}
    
    if schema['impact'].get('has_work_references', False) and schema['impact'].get('has_works', False):
        work_ref_columns = schema['impact'].get('work_references_columns', [])
        
        try:
            cursor = impact_conn.cursor()
            
            # Check if year column exists for temporal analysis
            if 'year' in work_ref_columns:
                cursor.execute("""
                    SELECT wr.work_id, w.id, wr.year
                    FROM work_references wr
                    JOIN works w ON wr.doi = w.doi
                    WHERE wr.doi IS NOT NULL AND wr.work_id IS NOT NULL
                """)
                
                for src_id, tgt_id, year in tqdm(cursor.fetchall(), desc="Loading citation data"):
                    work_to_cited.setdefault(src_id, []).append(tgt_id)
                    
                    # Store citation year if available
                    if year is not None:
                        try:
                            citation_years[(src_id, tgt_id)] = int(year)
                        except (ValueError, TypeError):
                            pass
            else:
                # Fallback if year isn't available
                cursor.execute("""
                    SELECT wr.work_id, w.id
                    FROM work_references wr
                    JOIN works w ON wr.doi = w.doi
                    WHERE wr.doi IS NOT NULL AND wr.work_id IS NOT NULL
                """)
                
                for src_id, tgt_id in tqdm(cursor.fetchall(), desc="Loading citation data"):
                    work_to_cited.setdefault(src_id, []).append(tgt_id)
            
            cursor.close()
            print(f"[INFO] Loaded {len(work_to_cited)} citation relationships and {len(citation_years)} citation years.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load citation data: {e}")
    else:
        print("[WARNING] work_references or works table not found. No citation data loaded.")
    
    return work_to_cited, citation_years


def load_author_pairs(rolap_conn, schema, num_pairs=5):
    pairs = []
    
    if schema['rolap'].get('has_matched_authors', False):
        try:
            cursor = rolap_conn.cursor()
            
            # First, check how many unique bottom authors exist
            cursor.execute("""
                SELECT COUNT(DISTINCT bottom_orcid) 
                FROM matched_authors_with_counts
                WHERE bottom_orcid IS NOT NULL
            """)
            unique_bottom_count = cursor.fetchone()[0]
            print(f"[INFO] Found {unique_bottom_count} unique bottom authors in database.")
            
            # Get a large batch of potential pairs
            cursor.execute("""
                SELECT bottom_orcid, random_orcid,
                       (COALESCE(bottom_n_works,0) + COALESCE(random_n_works,0)) AS total_works
                FROM matched_authors_with_counts
                WHERE bottom_orcid IS NOT NULL AND random_orcid IS NOT NULL
                ORDER BY total_works DESC
                LIMIT 500
            """)
            all_pairs = cursor.fetchall()
            cursor.close()
            
            print(f"[INFO] Retrieved {len(all_pairs)} total potential pairs.")
            
            # Strategy 1: Try to get unique bottom authors
            seen_bottom_authors = set()
            unique_pairs = []
            
            for bottom, random, total in all_pairs:
                if bottom not in seen_bottom_authors:
                    unique_pairs.append((bottom, random, total))
                    seen_bottom_authors.add(bottom)
                if len(unique_pairs) >= num_pairs:
                    break
            
            # If we found enough unique pairs, use them
            if len(unique_pairs) >= num_pairs:
                pairs = unique_pairs[:num_pairs]
                print(f"[INFO] Using {len(pairs)} pairs with unique bottom authors.")
            else:
                # Strategy 2: If not enough unique bottom authors, take the best pairs available
                print(f"[WARNING] Only found {len(unique_pairs)} unique bottom authors. Using best available pairs.")
                pairs = all_pairs[:num_pairs]
                
                # Print what we're using to help debugging
                used_bottoms = [p[0] for p in pairs]
                print(f"[INFO] Bottom authors being used: {used_bottoms}")
                
            print(f"[INFO] Loaded {len(pairs)} author pairs from matched_authors_with_counts.")
            
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load author pairs: {e}")
    
    # If no pairs were loaded, use fallback pairs
    if not pairs:
        print("[INFO] Using fallback pairs.")
        fallback_pairs = [
            ("0000-0003-0094-1778", "0000-0001-5204-3465", 100),
            ("0000-0001-6645-8645", "0000-0001-5236-4592", 90),
            ("0000-0002-9840-3726", "0000-0002-8656-1444", 85),
            ("0000-0001-5125-7648", "0000-0001-9215-9737", 75),
            ("0000-0001-9872-8742", "0000-0002-1871-1850", 60)
        ]
        
        # Trim to requested number of pairs
        pairs = fallback_pairs[:num_pairs]
    
    return pairs


##############################################################################
# 3) Citation Network Building
##############################################################################
def build_citation_network(
    start_author, 
    depth, 
    max_nodes,
    orcid_to_works, 
    work_to_authors, 
    work_to_cited,
    citation_years=None
):
    """
    Build a citation network starting from a given author.
    
    This function constructs a directed graph where:
    - Nodes are authors (identified by ORCID)
    - Edges represent citations (author A cites author B)
    - Edge weights indicate citation frequency
    - Edge attributes include citation years when available
    
    Args:
        start_author: The ORCID of the starting author
        depth: Maximum BFS traversal depth
        max_nodes: Maximum number of nodes to include
        orcid_to_works: Dictionary mapping ORCIDs to work IDs
        work_to_authors: Dictionary mapping work IDs to ORCIDs
        work_to_cited: Dictionary mapping work IDs to cited work IDs
        citation_years: Dictionary mapping (source_id, target_id) to years
        
    Returns:
        nx.DiGraph: Directed graph representing the citation network
    """
    # Initialize directed graph
    G = nx.DiGraph()
    G.add_node(start_author, seed=True)  # Mark the seed author
    
    # Track visited nodes and initialize queue
    visited = set([start_author])
    queue = deque([(start_author, 0)])  # (author, depth)
    
    # Create progress bar
    pbar = tqdm(total=min(max_nodes, 1000), desc=f"Building network from {start_author}")
    pbar.update(1)  # Count the seed author
    
    # Breadth-first traversal
    while queue:
        current_author, current_depth = queue.popleft()
        
        # Stop if we've reached max depth or max nodes
        if current_depth >= depth or len(visited) >= max_nodes:
            continue
            
        # Get works by the current author
        author_works = orcid_to_works.get(current_author, [])
        
        # Process each work by the current author
        for work_id in author_works:
            # Find works cited by this work
            cited_work_ids = work_to_cited.get(work_id, [])
            
            # Process each cited work
            for cited_work_id in cited_work_ids:
                # Find authors of the cited work
                cited_authors = work_to_authors.get(cited_work_id, [])
                
                # Process each cited author
                for cited_author in cited_authors:
                    # Skip self-citations to avoid self-loops
                    if cited_author == current_author:
                        continue
                        
                    # Add or update edge (current_author cites cited_author)
                    if G.has_edge(current_author, cited_author):
                        # Increase weight for existing edge
                        G[current_author][cited_author]["weight"] += 1
                        
                        # Add this work pair to citation list
                        G[current_author][cited_author]["citations"].append((work_id, cited_work_id))
                    else:
                        # Create new edge
                        G.add_edge(
                            current_author, 
                            cited_author, 
                            weight=1, 
                            citations=[(work_id, cited_work_id)]
                        )
                    
                    # Add citation year if available
                    if citation_years and (work_id, cited_work_id) in citation_years:
                        # Initialize or append to citation_years list
                        if "citation_years" not in G[current_author][cited_author]:
                            G[current_author][cited_author]["citation_years"] = []
                        
                        G[current_author][cited_author]["citation_years"].append(
                            citation_years[(work_id, cited_work_id)]
                        )
                    
                    # Add to BFS queue if not visited
                    if cited_author not in visited:
                        visited.add(cited_author)
                        queue.append((cited_author, current_depth + 1))
                        pbar.update(1)
                        
                        # Stop if we've reached max nodes
                        if len(visited) >= max_nodes:
                            pbar.close()
                            print(f"[INFO] Network building stopped at {len(visited)} nodes (max limit reached).")
                            return G
    
    pbar.close()
    print(f"[INFO] Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


##############################################################################
# 4) Clique Analysis
##############################################################################
def find_suspicious_cliques(
    graph, 
    bad_authors, 
    min_size=SUSPICIOUS_CLIQUE_MIN_SIZE, 
    min_density=SUSPICIOUS_CLIQUE_MIN_DENSITY,
    min_bad_fraction=SUSPICIOUS_CLIQUE_MIN_BAD_FRACTION
):
    """
    Find suspicious cliques in the citation network.
    
    A clique is a subgraph where every node is connected to every other node.
    This function identifies cliques that have a high proportion of bad authors
    and other suspicious characteristics.
    
    Args:
        graph: NetworkX directed graph of the citation network
        bad_authors: Set of author IDs considered 'bad'
        min_size: Minimum clique size to consider
        min_density: Minimum clique density threshold
        min_bad_fraction: Minimum fraction of bad authors in clique
        
    Returns:
        list: List of dictionaries with clique information and metrics
    """
    print("[INFO] Finding suspicious cliques...")
    
    # Convert to undirected graph for clique finding
    UG = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data["weight"]
        existing = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing + w)
    
    # Early exit if graph is too small
    if UG.number_of_nodes() < min_size:
        print(f"[WARNING] Graph too small ({UG.number_of_nodes()} nodes) for clique detection.")
        return []
    
    # Find all maximal cliques
    try:
        all_cliques = list(nx.find_cliques(UG))
        print(f"[INFO] Found {len(all_cliques)} maximal cliques in total")
    except nx.NetworkXError as e:
        print(f"[WARNING] Error finding cliques: {e}")
        return []
    
    # Filter and analyze suspicious cliques
    suspicious_cliques = []
    
    for clique in tqdm(all_cliques, desc="Analyzing cliques"):
        # Skip cliques that are too small
        if len(clique) < min_size:
            continue
            
        # Extract subgraph for this clique
        subg = UG.subgraph(clique).copy()
        
        # Calculate basic metrics
        n_nodes = subg.number_of_nodes()
        n_edges = subg.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        
        # Skip if density is too low
        if density < min_density:
            continue
            
        # Calculate bad author metrics
        bad_nodes = [node for node in clique if node in bad_authors]
        bad_count = len(bad_nodes)
        bad_fraction = bad_count / n_nodes if n_nodes > 0 else 0
        
        # Skip if bad fraction is too low
        if bad_fraction < min_bad_fraction:
            continue
            
        # Calculate edge weights
        weights = [data["weight"] for _, _, data in subg.edges(data=True)]
        avg_weight = statistics.mean(weights) if weights else 0
        max_weight = max(weights) if weights else 0
        min_weight = min(weights) if weights else 0
        
        # Create directed subgraph for directional analysis
        directed_subg = graph.subgraph(clique).copy()
        
        # Calculate reciprocity (bidirectional citations)
        recip_edges = sum(1 for u, v in directed_subg.edges() 
                         if directed_subg.has_edge(v, u))
        recip_ratio = recip_edges / (2 * directed_subg.number_of_edges()) if directed_subg.number_of_edges() > 0 else 0
        
        # Calculate citation imbalance within clique
        out_citations = {node: 0 for node in clique}
        in_citations = {node: 0 for node in clique}
        
        for u, v in directed_subg.edges():
            out_citations[u] += 1
            in_citations[v] += 1
        
        # Calculate standard deviations of in/out citations
        try:
            out_std = statistics.stdev(list(out_citations.values())) if len(clique) > 1 else 0
            in_std = statistics.stdev(list(in_citations.values())) if len(clique) > 1 else 0
            citation_balance = (out_std + in_std) / 2  # Higher means more imbalance
        except statistics.StatisticsError:
            citation_balance = 0
        
        # Calculate temporal metrics if available
        temporal_metrics = {}
        years_available = False
        
        for u, v in directed_subg.edges():
            if "citation_years" in directed_subg[u][v]:
                years_available = True
                break
                
        if years_available:
            # Calculate year differences
            year_diffs = []
            for u, v in directed_subg.edges():
                if "citation_years" in directed_subg[u][v]:
                    years = directed_subg[u][v]["citation_years"]
                    for year in years:
                        year_diffs.append(year)
            
            if year_diffs:
                temporal_metrics["median_year"] = statistics.median(year_diffs)
                temporal_metrics["year_range"] = max(year_diffs) - min(year_diffs)
        
        # Calculate potential anomaly score based on various metrics
        anomaly_factors = [
            bad_fraction * 0.4,  # Higher proportion of bad authors
            density * 0.2,  # Higher density
            (1 - recip_ratio) * 0.1,  # Lower reciprocity is more suspicious
            (avg_weight / 10) * 0.1,  # Higher average weight
            min(1.0, citation_balance * 0.5) * 0.1,  # Higher citation imbalance
            min(1.0, n_nodes / 10) * 0.1  # Larger cliques are more suspicious
        ]
        
        anomaly_score = sum(anomaly_factors)
        
        # Store clique with all metrics
        clique_data = {
            'nodes': clique,
            'size': n_nodes,
            'density': density,
            'bad_count': bad_count,
            'bad_fraction': bad_fraction,
            'avg_edge_weight': avg_weight,
            'max_edge_weight': max_weight,
            'min_edge_weight': min_weight,
            'reciprocity': recip_ratio,
            'citation_balance': citation_balance,
            'out_citation_std': out_std,
            'in_citation_std': in_std,
            'anomaly_score': anomaly_score
        }
        
        # Add temporal metrics if available
        clique_data.update(temporal_metrics)
        
        suspicious_cliques.append(clique_data)
    
    # Sort by anomaly score
    suspicious_cliques.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    print(f"[INFO] Found {len(suspicious_cliques)} suspicious cliques")
    return suspicious_cliques


##############################################################################
# 5) Citation Ring Detection
##############################################################################
def find_suspicious_cliques(
    graph, 
    bad_authors, 
    min_size=SUSPICIOUS_CLIQUE_MIN_SIZE, 
    min_density=SUSPICIOUS_CLIQUE_MIN_DENSITY,
    min_bad_fraction=SUSPICIOUS_CLIQUE_MIN_BAD_FRACTION
):
    """
    Find suspicious cliques in the citation network.
    
    A clique is a subgraph where every node is connected to every other node.
    This function identifies cliques that have a high proportion of bad authors
    and other suspicious characteristics.
    
    Args:
        graph: NetworkX directed graph of the citation network
        bad_authors: Set of author IDs considered 'bad'
        min_size: Minimum clique size to consider
        min_density: Minimum clique density threshold
        min_bad_fraction: Minimum fraction of bad authors in clique
        
    Returns:
        list: List of dictionaries with clique information and metrics
    """
    print("[INFO] Finding suspicious cliques...")
    
    # Convert to undirected graph for clique finding
    UG = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data["weight"]
        existing = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing + w)
    
    # Early exit if graph is too small
    if UG.number_of_nodes() < min_size:
        print(f"[WARNING] Graph too small ({UG.number_of_nodes()} nodes) for clique detection.")
        return []
    
    # Find all maximal cliques
    try:
        all_cliques = list(nx.find_cliques(UG))
        print(f"[INFO] Found {len(all_cliques)} maximal cliques in total")
    except nx.NetworkXError as e:
        print(f"[WARNING] Error finding cliques: {e}")
        return []
    
    # Filter and analyze suspicious cliques
    suspicious_cliques = []
    
    for clique in tqdm(all_cliques, desc="Analyzing cliques"):
        # Skip cliques that are too small
        if len(clique) < min_size:
            continue
            
        # Extract subgraph for this clique
        subg = UG.subgraph(clique).copy()
        
        # Calculate basic metrics
        n_nodes = subg.number_of_nodes()
        n_edges = subg.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        
        # Skip if density is too low
        if density < min_density:
            continue
            
        # Calculate bad author metrics
        bad_nodes = [node for node in clique if node in bad_authors]
        bad_count = len(bad_nodes)
        bad_fraction = bad_count / n_nodes if n_nodes > 0 else 0
        
        # Skip if bad fraction is too low
        if bad_fraction < min_bad_fraction:
            continue
            
        # Calculate edge weights
        weights = [data["weight"] for _, _, data in subg.edges(data=True)]
        avg_weight = statistics.mean(weights) if weights else 0
        max_weight = max(weights) if weights else 0
        min_weight = min(weights) if weights else 0
        
        # Create directed subgraph for directional analysis
        directed_subg = graph.subgraph(clique).copy()
        
        # Calculate reciprocity (bidirectional citations)
        recip_edges = sum(1 for u, v in directed_subg.edges() 
                         if directed_subg.has_edge(v, u))
        recip_ratio = recip_edges / (2 * directed_subg.number_of_edges()) if directed_subg.number_of_edges() > 0 else 0
        
        # Calculate citation imbalance within clique
        out_citations = {node: 0 for node in clique}
        in_citations = {node: 0 for node in clique}
        
        for u, v in directed_subg.edges():
            out_citations[u] += 1
            in_citations[v] += 1
        
        # Calculate standard deviations of in/out citations
        try:
            out_std = statistics.stdev(list(out_citations.values())) if len(clique) > 1 else 0
            in_std = statistics.stdev(list(in_citations.values())) if len(clique) > 1 else 0
            citation_balance = (out_std + in_std) / 2  # Higher means more imbalance
        except statistics.StatisticsError:
            citation_balance = 0
        
        # Calculate temporal metrics if available
        temporal_metrics = {}
        years_available = False
        
        for u, v in directed_subg.edges():
            if "citation_years" in directed_subg[u][v]:
                years_available = True
                break
                
        if years_available:
            # Calculate year differences
            year_diffs = []
            for u, v in directed_subg.edges():
                if "citation_years" in directed_subg[u][v]:
                    years = directed_subg[u][v]["citation_years"]
                    for year in years:
                        year_diffs.append(year)
            
            if year_diffs:
                temporal_metrics["median_year"] = statistics.median(year_diffs)
                temporal_metrics["year_range"] = max(year_diffs) - min(year_diffs)
                # Add standard deviation of years if there are multiple years
                if len(year_diffs) > 1:
                    temporal_metrics["year_std"] = statistics.stdev(year_diffs)
        
        # Calculate potential anomaly score based on various metrics
        anomaly_factors = [
            bad_fraction * 0.4,  # Higher proportion of bad authors
            density * 0.2,  # Higher density
            (1 - recip_ratio) * 0.1,  # Lower reciprocity is more suspicious
            (avg_weight / 10) * 0.1,  # Higher average weight
            min(1.0, citation_balance * 0.5) * 0.1,  # Higher citation imbalance
            min(1.0, n_nodes / 10) * 0.1  # Larger cliques are more suspicious
        ]
        
        # Add temporal factor if available
        if "year_range" in temporal_metrics:
            # Small year range is more suspicious (coordinated)
            year_range_factor = max(0, 1 - (temporal_metrics["year_range"] / 5))
            anomaly_factors.append(year_range_factor * 0.1)
        
        anomaly_score = sum(anomaly_factors)
        
        # Store clique with all metrics
        clique_data = {
            'nodes': clique,
            'size': n_nodes,
            'density': density,
            'bad_count': bad_count,
            'bad_fraction': bad_fraction,
            'avg_edge_weight': avg_weight,
            'max_edge_weight': max_weight,
            'min_edge_weight': min_weight,
            'reciprocity': recip_ratio,
            'citation_balance': citation_balance,
            'out_citation_std': out_std,
            'in_citation_std': in_std,
            'anomaly_score': anomaly_score
        }
        
        # Add temporal metrics if available
        clique_data.update(temporal_metrics)
        
        suspicious_cliques.append(clique_data)
    
    # Sort by anomaly score
    suspicious_cliques.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    print(f"[INFO] Found {len(suspicious_cliques)} suspicious cliques")
    return suspicious_cliques


def find_citation_rings(
    graph, 
    bad_authors, 
    min_size=CITATION_RING_MIN_SIZE, 
    max_size=12,  # Reduced from CITATION_RING_MAX_SIZE (likely 12) to 7
    min_bad_fraction=CITATION_RING_MIN_BAD_FRACTION,
    max_cycles=200_000,  # Reduced from 5,000,000 to 100,000
    timeout_seconds=7200  # Reduced from 2 hours
):
    """
    Find circular citation patterns (rings) in the directed citation graph.
    Uses custom cycle detection with early termination for better performance.
    
    Args:
        graph: NetworkX directed graph of the citation network
        bad_authors: Set of author IDs considered 'bad'
        min_size: Minimum ring size to consider
        max_size: Maximum ring size to consider
        min_bad_fraction: Minimum fraction of bad authors in ring
        max_cycles: Maximum number of cycles to process
        timeout_seconds: Maximum seconds to spend on cycle detection
        
    Returns:
        list: List of dictionaries with ring information and metrics
    """
    print("[INFO] Finding citation rings...")
    
    # Early exit for small or empty graphs
    if graph.number_of_nodes() < min_size:
        print(f"[WARNING] Graph too small ({graph.number_of_nodes()} nodes) for ring detection.")
        return []
    
    rings = []
    
    try:
        import time
        start_time = time.time()
        
        # Create a smaller, focused subgraph for analysis
        if graph.number_of_nodes() > 1000:
            # Prioritize bad authors and their immediate connections
            seed_nodes = set()
            for bad_author in bad_authors.intersection(set(graph.nodes())):
                seed_nodes.add(bad_author)
                # Reduced neighbor exploration
                seed_nodes.update(list(graph.successors(bad_author))[:15])
                seed_nodes.update(list(graph.predecessors(bad_author))[:15])
            
            # Keep subgraph small enough for efficient processing
            if len(seed_nodes) > 400:
                seed_nodes = set(list(seed_nodes)[:400])
                
            subgraph = graph.subgraph(seed_nodes).copy()
        else:
            subgraph = graph
            
        print(f"[INFO] Using subgraph with {subgraph.number_of_nodes()} nodes for cycle detection")
        
        # Custom cycle detection algorithm that's more efficient than nx.simple_cycles
        def find_limited_cycles(graph, min_size, max_size, max_cycles, timeout_seconds, start_time):
            """More efficient cycle detection with size filtering built in"""
            cycles = []
            cycle_count = 0
            
            # Process nodes with priority given to bad authors first
            node_priority = [(node, node in bad_authors) for node in graph.nodes()]
            node_priority.sort(key=lambda x: x[1], reverse=True)  # Bad authors first
            
            for start_node, _ in node_priority:
                # Check timeout and cycle count limits
                if time.time() - start_time > timeout_seconds:
                    print(f"[WARNING] Cycle detection timeout after {int(time.time() - start_time)} seconds.")
                    break
                    
                if cycle_count >= max_cycles:
                    print(f"[WARNING] Reached maximum cycle count limit of {max_cycles}.")
                    break
                
                # Filter nodes that can't be part of a cycle of min_size
                if graph.out_degree(start_node) == 0 or graph.in_degree(start_node) == 0:
                    continue
                
                # Track visited nodes and path
                visited = {start_node: 0}  # Node -> position in path
                path = [start_node]
                
                # Start DFS for this node
                def dfs_find_cycles(current):
                    nonlocal cycle_count
                    
                    # Too early to close a cycle
                    if len(path) < min_size - 1:
                        search_nodes = list(graph.successors(current))
                    else:
                        # For potential cycle completion, only look at edges to start_node
                        if start_node in graph.successors(current):
                            search_nodes = [start_node]
                        else:
                            search_nodes = list(graph.successors(current))
                    
                    for successor in search_nodes:
                        # Check timeout and cycle count frequently
                        cycle_count += 1
                        if cycle_count % 50000 == 0:
                            if time.time() - start_time > timeout_seconds:
                                return True  # Signal to stop
                            print(f"[INFO] Processed {cycle_count} potential cycles...")
                        
                        # Found a cycle back to start node
                        if successor == start_node and len(path) >= min_size - 1:
                            cycle = path.copy()
                            cycles.append(cycle)
                            if len(cycles) % 1000 == 0:
                                print(f"[INFO] Found {len(cycles)} valid cycles so far...")
                            if len(cycles) >= max_cycles // 10:
                                return True  # Signal to stop
                                
                        # Continue DFS if not creating a path that's too long
                        elif successor not in visited and len(path) < max_size - 1:
                            visited[successor] = len(path)
                            path.append(successor)
                            if dfs_find_cycles(successor):
                                return True  # Propagate stop signal
                            path.pop()  # Backtrack
                            del visited[successor]
                    
                    return False
                
                # Run DFS from this start node
                if dfs_find_cycles(start_node):
                    break  # Stop processing if timeout or limits hit
            
            print(f"[INFO] Found {len(cycles)} valid cycles (processed {cycle_count} potential cycles)")
            return cycles
        
        # Find cycles using our custom algorithm
        cycles_list = find_limited_cycles(
            subgraph, min_size, max_size, max_cycles, timeout_seconds, start_time
        )
        
        # If we still have too many cycles, sample a reasonable number
        if len(cycles_list) > 10000:
            import random
            print(f"[INFO] Sampling 10000 cycles from {len(cycles_list)} total cycles")
            cycles_list = random.sample(cycles_list, 10000)
        
        # Process each cycle with the existing analysis code
        for cycle in tqdm(cycles_list, desc="Analyzing citation rings"):
            # Calculate bad author fraction
            bad_count = sum(1 for node in cycle if node in bad_authors)
            bad_fraction = bad_count / len(cycle)
            
            # Apply size-dependent thresholds for bad author fraction
            if len(cycle) > 5:
                # For larger rings, reduce the required bad author percentage
                adjusted_min_bad_fraction = min_bad_fraction * (0.9 - (len(cycle) - 5) * 0.05)
                # Ensures it doesn't go below 0.4
                adjusted_min_bad_fraction = max(0.4, adjusted_min_bad_fraction)
                
                if bad_fraction < adjusted_min_bad_fraction:
                    continue
            else:
                if bad_fraction < min_bad_fraction:
                    continue
                
            # Create a subgraph for this cycle
            cycle_graph = graph.subgraph(cycle).copy()
            
            # Check if it's a complete cycle (each node connects to the next)
            # This is necessary because our cycle detection might return paths that aren't in order
            ordered_cycle = False
            for i in range(len(cycle)):
                is_complete = all(cycle_graph.has_edge(cycle[j], cycle[(j+1) % len(cycle)]) 
                                for j in range(len(cycle)))
                if is_complete:
                    ordered_cycle = True
                    break
                # Try a rotation of the cycle
                cycle = cycle[1:] + [cycle[0]]
            
            if not ordered_cycle:
                continue
                
            # Calculate edge weights
            edge_weights = [cycle_graph[cycle[i]][cycle[(i+1) % len(cycle)]].get('weight', 1) 
                           for i in range(len(cycle))]
            
            avg_weight = statistics.mean(edge_weights)
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            
            # Calculate weight consistency (low standard deviation is more suspicious)
            try:
                weight_std = statistics.stdev(edge_weights) if len(edge_weights) > 1 else 0
                weight_consistency = 1.0 - min(1.0, weight_std / avg_weight)  # Higher means more consistent
            except statistics.StatisticsError:
                weight_consistency = 0
            
            # Calculate temporal metrics if available
            temporal_metrics = {}
            years_available = False
            
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                if "citation_years" in cycle_graph[u][v]:
                    years_available = True
                    break
                    
            if years_available:
                # Collect all years
                years = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    if "citation_years" in cycle_graph[u][v]:
                        years.extend(cycle_graph[u][v]["citation_years"])
                
                if years:
                    temporal_metrics["median_year"] = statistics.median(years)
                    temporal_metrics["year_range"] = max(years) - min(years) if len(years) > 1 else 0
                    temporal_metrics["year_std"] = statistics.stdev(years) if len(years) > 1 else 0
            
            # Calculate enhanced anomaly score for the ring
            anomaly_factors = [
                bad_fraction * 0.35,
                (min_weight / 5) * 0.10,
                (avg_weight / 10) * 0.10,
                weight_consistency * 0.10,
                (len(cycle) / max_size) * 0.25,
                0.10  # Base score
            ]
            
            # Add penalty for large temporal spread if available
            if years_available and "year_range" in temporal_metrics:
                # Small year range is more suspicious (coordinated)
                year_range_factor = max(0, 1 - (temporal_metrics["year_range"] / 10))
                anomaly_factors.append(year_range_factor * 0.15)
            
            anomaly_score = sum(anomaly_factors)
            
            # Store the ring with metrics
            ring_data = {
                'nodes': cycle,
                'size': len(cycle),
                'bad_count': bad_count,
                'bad_fraction': bad_fraction,
                'avg_edge_weight': avg_weight,
                'min_edge_weight': min_weight,
                'max_edge_weight': max_weight,
                'weight_consistency': weight_consistency,
                'anomaly_score': anomaly_score
            }
            
            # Add temporal metrics if available
            ring_data.update(temporal_metrics)
            
            rings.append(ring_data)
    
    except (nx.NetworkXError, MemoryError) as e:
        print(f"[WARNING] Error detecting cycles: {e}")
    
    # Sort by a composite score that prioritizes larger rings
    rings.sort(key=lambda x: (x['anomaly_score'] * (1 + 0.1 * min(x['size'], 10))), reverse=True)
    
    # Additional reporting on ring sizes
    if rings:
        size_distribution = Counter([r['size'] for r in rings])
        print("[INFO] Ring size distribution:")
        for size in sorted(size_distribution.keys()):
            print(f"  Size {size}: {size_distribution[size]} rings")
            
        # Print details of largest rings
        largest_size = max(r['size'] for r in rings)
        if largest_size > 5:
            large_rings = [r for r in rings if r['size'] > 5]
            print(f"[INFO] Found {len(large_rings)} larger rings (>5 authors)")
            if large_rings:
                for i, ring in enumerate(sorted(large_rings, key=lambda x: x['size'], reverse=True)[:3]):
                    print(f"  Large ring #{i+1}: Size={ring['size']}, Bad={ring['bad_count']}/{ring['size']}, "
                          f"Avg. Weight={ring['avg_edge_weight']:.1f}, Score={ring['anomaly_score']:.4f}")
    
    print(f"[INFO] Found {len(rings)} suspicious citation rings")
    return rings


##############################################################################
# 6) Author Citation Behavior Analysis
##############################################################################
def analyze_author_citation_behavior(graph, bad_authors, author_metrics=None):
    """
    Analyze individual author citation behavior to detect anomalies.
    
    This function calculates various metrics for each author in the network,
    such as citation patterns, connections to bad authors, and reciprocity.
    
    Args:
        graph: NetworkX directed graph of the citation network
        bad_authors: Set of author IDs considered 'bad'
        author_metrics: Dictionary of external author metrics (optional)
        
    Returns:
        dict: Dictionary of author IDs to behavior metrics
    """
    print("[INFO] Analyzing author citation behavior...")
    
    # Early exit for empty graphs
    if graph.number_of_nodes() == 0:
        print("[WARNING] Empty graph, no author behavior to analyze.")
        return {}
    
    author_analysis = {}
    
    # Process each author in the graph
    for author in tqdm(graph.nodes(), desc="Analyzing authors"):
        # Get basic degree information
        out_degree = graph.out_degree(author)
        in_degree = graph.in_degree(author)
        
        # Skip authors with no connections
        if out_degree == 0 and in_degree == 0:
            continue
            
        # Calculate bad citation ratios
        bad_out_edges = sum(1 for _, v in graph.out_edges(author) if v in bad_authors)
        bad_in_edges = sum(1 for u, _ in graph.in_edges(author) if u in bad_authors)
        
        bad_out_ratio = bad_out_edges / out_degree if out_degree > 0 else 0
        bad_in_ratio = bad_in_edges / in_degree if in_degree > 0 else 0
        
        # Calculate reciprocity for this author
        recip_count = sum(1 for _, v in graph.out_edges(author) if graph.has_edge(v, author))
        recip_ratio = recip_count / out_degree if out_degree > 0 else 0
        
        # Calculate local clustering coefficient (tendency to form cliques)
        try:
            local_clustering = nx.clustering(nx.Graph(graph), author)
        except:
            local_clustering = 0
        
        # Identify citation clusters (authors frequently cited together)
        cited_authors = [v for _, v in graph.out_edges(author)]
        citing_authors = [u for u, _ in graph.in_edges(author)]
        
        # Count co-citations (citing the same authors together)
        co_citation_count = Counter()
        for i in range(len(cited_authors)):
            for j in range(i+1, len(cited_authors)):
                co_citation_count[(cited_authors[i], cited_authors[j])] += 1
        
        # Find strong co-citation pairs involving bad authors
        strong_co_citations = [(pair, count) for pair, count in co_citation_count.items() 
                              if count >= 2 and (pair[0] in bad_authors or pair[1] in bad_authors)]
        
        # Calculate citation weights
        out_weights = [data["weight"] for _, _, data in graph.out_edges(author, data=True)]
        in_weights = [data["weight"] for _, _, data in graph.in_edges(author, data=True)]
        
        avg_out_weight = statistics.mean(out_weights) if out_weights else 0
        avg_in_weight = statistics.mean(in_weights) if in_weights else 0
        max_out_weight = max(out_weights) if out_weights else 0
        max_in_weight = max(in_weights) if in_weights else 0
        
        # Calculate synthetic anomaly metrics
        citation_anomaly = 0
        
        # More citations to bad authors is suspicious
        citation_anomaly += bad_out_ratio * 0.3
        
        # More citations from bad authors is suspicious
        citation_anomaly += bad_in_ratio * 0.2
        
        # High reciprocity with bad authors is suspicious
        if author in bad_authors:
            citation_anomaly += recip_ratio * 0.2
        
        # High clustering coefficient is suspicious for bad authors
        if author in bad_authors:
            citation_anomaly += local_clustering * 0.15
        
        # Strong citation weights are suspicious
        citation_anomaly += min(1.0, avg_out_weight / 10) * 0.1
        
        # Strong co-citations of bad authors are suspicious
        citation_anomaly += min(len(strong_co_citations) * 0.05, 0.15)
        
        # Store all metrics
        author_analysis[author] = {
            'is_bad': author in bad_authors,
            'out_degree': out_degree,
            'in_degree': in_degree,
            'bad_out_ratio': bad_out_ratio,
            'bad_in_ratio': bad_in_ratio,
            'reciprocity': recip_ratio,
            'clustering': local_clustering,
            'avg_out_weight': avg_out_weight,
            'avg_in_weight': avg_in_weight,
            'max_out_weight': max_out_weight,
            'max_in_weight': max_in_weight,
            'strong_co_citations': len(strong_co_citations),
            'citation_anomaly_score': citation_anomaly
        }
        
        # Add external metrics if available
        if author_metrics and author in author_metrics:
            for key, value in author_metrics[author].items():
                author_analysis[author][key] = value
    
    print(f"[INFO] Analyzed {len(author_analysis)} authors in the network.")
    return author_analysis


##############################################################################
# 7) Visualization Functions
##############################################################################
def visualize_suspicious_clique(graph, clique_data, bad_authors, output_path):
    """
    Create a detailed visualization of a suspicious clique.
    
    Args:
        graph: NetworkX directed graph of the citation network
        clique_data: Dictionary with clique metrics and node list
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
    """
    # Extract nodes from clique data
    nodes = clique_data['nodes']
    
    # Create a subgraph with just these nodes
    subgraph = graph.subgraph(nodes).copy()
    
    # Create an undirected version for layout
    undirected = nx.Graph(subgraph)
    
    # Use a spring layout for node positioning
    try:
        pos = nx.spring_layout(undirected, seed=42)
    except:
        # Fallback to circular layout if spring layout fails
        pos = nx.circular_layout(undirected)
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Node colors based on 'bad' status
    node_colors = ['#ff6666' if node in bad_authors else '#66b3ff' for node in subgraph.nodes()]
    
    # Node sizes based on degree
    node_sizes = [300 + 100 * (subgraph.in_degree(node) + subgraph.out_degree(node)) 
                 for node in subgraph.nodes()]
    
    # Edge colors and widths based on weight
    edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    
    edge_colors = [cm.plasma(data['weight'] / max_weight) 
                  for _, _, data in subgraph.edges(data=True)]
    
    edge_widths = [0.5 + 2.0 * (data['weight'] / max_weight) 
                  for _, _, data in subgraph.edges(data=True)]
    
    # Draw the graph
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    nx.draw_networkx_edges(subgraph, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7,
                          arrowsize=15,
                          connectionstyle='arc3,rad=0.1')  # Curved edges
    
    # Add labels to the nodes (shortened for readability)
    labels = {node: str(node)[:8] + '...' for node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    # Add a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(1, max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Citation Frequency')
    
    # Add title and metrics
    title = (f"Suspicious Clique (Size: {clique_data['size']}, "
            f"Bad Authors: {clique_data['bad_count']}/{clique_data['size']}, "
            f"Density: {clique_data['density']:.2f})")
    
    plt.title(title, fontsize=14)
    
    # Add legend
    plt.plot([0], [0], 'o', c='#ff6666', markersize=10, label='Low Eigenfactor Authors')
    plt.plot([0], [0], 'o', c='#66b3ff', markersize=10, label='Other Authors')
    plt.legend()
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved clique visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


def visualize_citation_ring(graph, ring_data, bad_authors, output_path):
    """
    Create a visualization of a suspicious citation ring.
    Enhanced to better display larger rings.
    
    Args:
        graph: NetworkX directed graph of the citation network
        ring_data: Dictionary with ring metrics and node list
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
    """
    # Extract nodes from ring data
    nodes = ring_data['nodes']
    
    # Create a subgraph with just these nodes
    subgraph = graph.subgraph(nodes).copy()
    
    # Set up the figure - adjust figure size for larger rings
    ring_size = len(nodes)
    fig_size = 10 + ring_size * 0.5  # Scale figure size with ring size
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use a circular layout for rings
    pos = nx.circular_layout(subgraph)
    
    # Node colors based on 'bad' status
    node_colors = ['#ff6666' if node in bad_authors else '#66b3ff' for node in subgraph.nodes()]
    
    # Node sizes - smaller for larger rings to prevent overlap
    node_size = 500 if ring_size <= 6 else max(200, 500 - (ring_size - 6) * 30)
    
    # Draw the nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_size,
                          alpha=0.8)
    
    # Get edge weights for color and width
    edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    
    edge_colors = [cm.inferno(data['weight'] / max_weight) 
                  for _, _, data in subgraph.edges(data=True)]
    
    edge_widths = [1.0 + 3.0 * (data['weight'] / max_weight) 
                  for _, _, data in subgraph.edges(data=True)]
    
    # Draw the edges
    nx.draw_networkx_edges(subgraph, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.8,
                          arrowsize=20)
    
    # Add labels to the nodes (adjusted font size for larger rings)
    font_size = 10 if ring_size <= 6 else max(6, 10 - (ring_size - 6) * 0.5)
    labels = {node: str(node)[:8] + '...' for node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=font_size)
    
    # Add a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(1, max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Citation Frequency')
    
    # Add title and metrics
    title = (f"Citation Ring (Size: {ring_data['size']}, "
            f"Bad Authors: {ring_data['bad_count']}/{ring_data['size']}, "
            f"Avg. Citation Frequency: {ring_data['avg_edge_weight']:.1f})")
    
    plt.title(title, fontsize=14)
    
    # Add legend
    plt.plot([0], [0], 'o', c='#ff6666', markersize=10, label='Low Eigenfactor Authors')
    plt.plot([0], [0], 'o', c='#66b3ff', markersize=10, label='Other Authors')
    plt.legend()
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved ring visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


def visualize_author_network(graph, author_analysis, bad_authors, output_path,
                           max_nodes=150, highlight_top=20):
    """
    Visualize the author network highlighting suspicious authors.
    
    Args:
        graph: NetworkX directed graph of the citation network
        author_analysis: Dictionary of author analysis results
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to include
        highlight_top: Number of top suspicious authors to highlight
    """
    # Early exit if no analysis results
    if not author_analysis:
        print("[WARNING] No author analysis results to visualize.")
        return
    
    # If graph is too large, select a subset based on anomaly scores
    if graph.number_of_nodes() > max_nodes:
        # Sort authors by anomaly score
        sorted_authors = sorted(author_analysis.items(), 
                               key=lambda x: x[1]['citation_anomaly_score'], 
                               reverse=True)
        
        # Take top authors and their neighbors
        top_authors = [author for author, _ in sorted_authors[:max_nodes//4]]
        
        # Add immediate neighbors
        neighbors = set()
        for author in top_authors:
            if author in graph:
                neighbors.update(list(graph.successors(author))[:5])
                neighbors.update(list(graph.predecessors(author))[:5])
        
        # Combine and limit
        selected_nodes = set(top_authors) | neighbors
        if len(selected_nodes) > max_nodes:
            # Prioritize authors with higher anomaly scores
            selected_nodes = set(sorted(selected_nodes, 
                                      key=lambda x: author_analysis.get(x, {}).get('citation_anomaly_score', 0),
                                      reverse=True)[:max_nodes])
        
        # Create subgraph
        subgraph = graph.subgraph(selected_nodes).copy()
    else:
        subgraph = graph.copy()
    
    # Set up the figure
    plt.figure(figsize=(18, 15))
    
    # Use spring layout for the network
    try:
        pos = nx.spring_layout(subgraph, k=0.2, iterations=50, seed=42)
    except:
        # Fallback to circular layout if spring layout fails
        pos = nx.circular_layout(subgraph)
    
    # Node sizes based on anomaly score
    node_sizes = []
    for node in subgraph.nodes():
        if node in author_analysis:
            score = author_analysis[node]['citation_anomaly_score']
            node_sizes.append(100 + 1000 * score)
        else:
            node_sizes.append(100)
    
    # Node colors based on status
    node_colors = []
    for node in subgraph.nodes():
        if node in bad_authors:
            if node in author_analysis and author_analysis[node]['citation_anomaly_score'] > 0.5:
                # High-risk bad author
                node_colors.append('#ff0000')
            else:
                # Normal bad author
                node_colors.append('#ff9999')
        else:
            if node in author_analysis and author_analysis[node]['citation_anomaly_score'] > 0.4:
                # Suspicious non-bad author
                node_colors.append('#ff9900')
            else:
                # Normal author
                node_colors.append('#66b3ff')
    
    # Draw the edges
    nx.draw_networkx_edges(subgraph, pos, 
                          alpha=0.2,
                          arrowsize=5,
                          width=0.5)
    
    # Draw the nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Label only the top suspicious authors
    top_authors = sorted([(node, author_analysis.get(node, {}).get('citation_anomaly_score', 0)) 
                         for node in subgraph.nodes()],
                        key=lambda x: x[1], reverse=True)[:highlight_top]
    
    labels = {author: f"{str(author)[:8]}..." for author, _ in top_authors}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    # Add title
    plt.title(f"Citation Network with Suspicious Authors Highlighted", fontsize=16)
    
    # Add legend
    plt.plot([0], [0], 'o', c='#ff0000', markersize=15, label='High-Risk Low Eigenfactor')
    plt.plot([0], [0], 'o', c='#ff9999', markersize=10, label='Low Eigenfactor')
    plt.plot([0], [0], 'o', c='#ff9900', markersize=10, label='Suspicious Normal')
    plt.plot([0], [0], 'o', c='#66b3ff', markersize=7, label='Normal')
    plt.legend(fontsize=12)
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved network visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


##############################################################################
# 8) Reporting Functions
##############################################################################
def save_clique_metrics(cliques, output_path):
    """
    Save clique metrics to a CSV file.
    
    Args:
        cliques: List of clique dictionaries
        output_path: Path to save the CSV file
    """
    if not cliques:
        print("[WARNING] No cliques to save.")
        return
    
    try:
        with open(output_path, 'w', newline='') as f:
            # Determine all possible fields from the first clique
            fields = set()
            for clique in cliques:
                fields.update(clique.keys())
            
            # Remove 'nodes' since it's a list and handled separately
            if 'nodes' in fields:
                fields.remove('nodes')
            
            # Sort fields for consistency
            sorted_fields = sorted(fields)
            
            # Create header with clique_id first, followed by other fields
            header = ['clique_id'] + sorted_fields + ['nodes_sample']
            
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write each clique
            for i, clique in enumerate(cliques):
                # Sample of nodes (first 5)
                nodes_sample = ','.join(str(node) for node in clique.get('nodes', [])[:5])
                if len(clique.get('nodes', [])) > 5:
                    nodes_sample += '...'
                
                # Build row with clique_id first
                row = [i+1]
                
                # Add all other fields
                for field in sorted_fields:
                    value = clique.get(field, '')
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(value)
                
                # Add nodes sample
                row.append(nodes_sample)
                
                writer.writerow(row)
        
        print(f"[INFO] Saved clique metrics to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save clique metrics: {e}")


def save_ring_metrics(rings, output_path):
    """
    Save ring metrics to a CSV file with additional summary for larger rings.
    
    Args:
        rings: List of ring dictionaries
        output_path: Path to save the CSV file
    """
    if not rings:
        print("[WARNING] No rings to save.")
        return
    
    try:
        with open(output_path, 'w', newline='') as f:
            # Determine all possible fields
            fields = set()
            for ring in rings:
                fields.update(ring.keys())
            
            # Remove 'nodes' since it's a list and handled separately
            if 'nodes' in fields:
                fields.remove('nodes')
            
            # Sort fields for consistency
            sorted_fields = sorted(fields)
            
            # Create header with ring_id first, followed by other fields
            header = ['ring_id'] + sorted_fields + ['nodes_sample']
            
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write each ring
            for i, ring in enumerate(rings):
                # Sample of nodes (first 5)
                nodes_sample = ','.join(str(node) for node in ring.get('nodes', [])[:5])
                if len(ring.get('nodes', [])) > 5:
                    nodes_sample += '...'
                
                # Build row with ring_id first
                row = [i+1]
                
                # Add all other fields
                for field in sorted_fields:
                    value = ring.get(field, '')
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(value)
                
                # Add nodes sample
                row.append(nodes_sample)
                
                writer.writerow(row)
        
        # Create a summary file for ring size distribution
        with open(f"{Path(output_path).parent}/{Path(output_path).stem}_summary.txt", 'w') as f:
            size_distribution = Counter([r['size'] for r in rings])
            f.write("Ring size distribution:\n")
            for size in sorted(size_distribution.keys()):
                f.write(f"  Size {size}: {size_distribution[size]} rings\n")
            
            # Report on larger rings
            large_rings = [r for r in rings if r['size'] > 5]
            f.write(f"\nLarger rings (>5 authors): {len(large_rings)} found\n")
            if large_rings:
                for i, ring in enumerate(sorted(large_rings, key=lambda x: x['size'], reverse=True)[:10]):
                    f.write(f"  Large ring #{i+1}: Size={ring['size']}, Bad={ring['bad_count']}/{ring['size']}, "
                          f"Avg. Weight={ring['avg_edge_weight']:.1f}, Score={ring['anomaly_score']:.4f}\n")
                    
                    # List member ORCIDs
                    f.write(f"    Members: {', '.join([str(node) for node in ring['nodes']])}\n\n")
        
        print(f"[INFO] Saved ring metrics to {output_path}")
        print(f"[INFO] Saved ring size summary to {Path(output_path).parent}/{Path(output_path).stem}_summary.txt")
    except Exception as e:
        print(f"[WARNING] Could not save ring metrics: {e}")

def save_author_metrics(author_analysis, output_path):
    """
    Save author metrics to a CSV file.
    
    Args:
        author_analysis: Dictionary of author analysis results
        output_path: Path to save the CSV file
    """
    if not author_analysis:
        print("[WARNING] No author analysis to save.")
        return
    
    try:
        with open(output_path, 'w', newline='') as f:
            # Determine all possible fields
            fields = set()
            for author_data in author_analysis.values():
                fields.update(author_data.keys())
            
            # Sort fields for consistency
            sorted_fields = sorted(fields)
            
            # Create header with author_id first, followed by other fields
            header = ['author_id'] + sorted_fields
            
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Sort authors by anomaly score
            sorted_authors = sorted(author_analysis.items(), 
                                  key=lambda x: x[1].get('citation_anomaly_score', 0), 
                                  reverse=True)
            
            # Write each author
            for author_id, metrics in sorted_authors:
                # Build row with author_id first
                row = [author_id]
                
                # Add all other fields
                for field in sorted_fields:
                    value = metrics.get(field, '')
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(value)
                
                writer.writerow(row)
        
        print(f"[INFO] Saved author metrics to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save author metrics: {e}")


def save_comparative_analysis(results, output_path):
    """
    Save comparative analysis of author pairs to a CSV file.
    
    Args:
        results: List of comparison results
        output_path: Path to save the CSV file
    """
    if not results:
        print("[WARNING] No comparative results to save.")
        return
    
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'pair_idx', 
                'bottom_author', 'random_author', 'total_works',
                # Network size metrics
                'bottom_nodes', 'random_nodes',
                'bottom_edges', 'random_edges',
                'bottom_density', 'random_density',
                # Bad author metrics
                'bottom_bad_node_fraction', 'random_bad_node_fraction',
                'bottom_bad_edge_fraction', 'random_bad_edge_fraction',
                # Suspicious pattern metrics
                'bottom_suspicious_cliques', 'random_suspicious_cliques',
                'bottom_citation_rings', 'random_citation_rings',
                'bottom_top_clique_size', 'random_top_clique_size',
                'bottom_top_clique_score', 'random_top_clique_score',
                'bottom_top_ring_size', 'random_top_ring_size',
                'bottom_top_ring_score', 'random_top_ring_score'
            ]
            writer.writerow(header)
            
            for result in results:
                row = [
                    result["pair_idx"],
                    result["bottom_author"], result["random_author"], result["total_works"],
                    # Network size metrics
                    result["bottom_metrics"]["nodes"], result["random_metrics"]["nodes"],
                    result["bottom_metrics"]["edges"], result["random_metrics"]["edges"],
                    f"{result['bottom_metrics']['density']:.6f}", f"{result['random_metrics']['density']:.6f}",
                    # Bad author metrics
                    f"{result['bottom_metrics']['bad_node_fraction']:.4f}", f"{result['random_metrics']['bad_node_fraction']:.4f}",
                    f"{result['bottom_metrics']['bad_edge_fraction']:.4f}", f"{result['random_metrics']['bad_edge_fraction']:.4f}",
                    # Suspicious pattern metrics
                    result["bottom_metrics"]["suspicious_cliques"], result["random_metrics"]["suspicious_cliques"],
                    result["bottom_metrics"]["citation_rings"], result["random_metrics"]["citation_rings"],
                    result["bottom_metrics"]["top_clique_size"], result["random_metrics"]["top_clique_size"],
                    f"{result['bottom_metrics']['top_clique_score']:.4f}", f"{result['random_metrics']['top_clique_score']:.4f}",
                    result["bottom_metrics"]["top_ring_size"], result["random_metrics"]["top_ring_size"],
                    f"{result['bottom_metrics']['top_ring_score']:.4f}", f"{result['random_metrics']['top_ring_score']:.4f}"
                ]
                writer.writerow(row)
        
        print(f"[INFO] Saved comparative analysis to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save comparative analysis: {e}")
        
##############################################################################
# 9) Statistical Analysis Functions
##############################################################################
def perform_network_permutation_test(G_bottom, G_random, statistic_func, n_permutations=1000):
    """
    Perform a permutation test to compare a network statistic between two graphs.
    
    Parameters:
    -----------
    G_bottom : networkx.Graph
        The citation network of bottom authors
    G_random : networkx.Graph
        The citation network of random authors
    statistic_func : function
        Function that calculates the desired statistic on a graph (e.g., clustering)
    n_permutations : int
        Number of permutations to perform
    
    Returns:
    --------
    p_value : float
        The p-value from the permutation test
    observed_diff : float
        The observed difference in the statistic between groups
    """
    import random
    
    # Calculate observed statistic difference
    bottom_stat = statistic_func(G_bottom)
    random_stat = statistic_func(G_random)
    observed_diff = bottom_stat - random_stat
    
    # Combine all nodes 
    all_nodes = list(set(G_bottom.nodes()).union(set(G_random.nodes())))
    
    # Calculate permutation distribution
    permutation_diffs = []
    for _ in range(n_permutations):
        # Randomly assign nodes to two groups
        random.shuffle(all_nodes)
        split_point = len(G_bottom.nodes())
        perm_bottom_nodes = all_nodes[:split_point]
        perm_random_nodes = all_nodes[split_point:]
        
        # Create subgraphs based on random assignment
        perm_bottom = nx.Graph()
        perm_random = nx.Graph()
        
        # Add edges if both endpoints are in the respective group
        for u, v in G_bottom.edges():
            if u in perm_bottom_nodes and v in perm_bottom_nodes:
                perm_bottom.add_edge(u, v)
        
        for u, v in G_random.edges():
            if u in perm_random_nodes and v in perm_random_nodes:
                perm_random.add_edge(u, v)
        
        # Calculate statistic difference
        perm_bottom_stat = statistic_func(perm_bottom)
        perm_random_stat = statistic_func(perm_random)
        perm_diff = perm_bottom_stat - perm_random_stat
        permutation_diffs.append(perm_diff)
    
    # Calculate two-sided p-value
    p_value = sum(1 for d in permutation_diffs if abs(d) >= abs(observed_diff)) / n_permutations
    
    return p_value, observed_diff


##############################################################################
# Statistical Helper Functions
##############################################################################

def bootstrap_citation_pattern(counts, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for citation pattern counts.
    
    Args:
        counts: Dictionary with 'bottom' and 'random' lists of counts
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (bottom_ci, random_ci, is_significant)
    """
    import numpy as np
    
    bottom_counts = np.array(counts['bottom'])
    random_counts = np.array(counts['random'])
    
    # Bootstrap bottom authors
    bottom_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(bottom_counts, size=len(bottom_counts), replace=True)
        bottom_means.append(np.mean(sample))
    
    # Bootstrap random authors
    random_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(random_counts, size=len(random_counts), replace=True)
        random_means.append(np.mean(sample))
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    bottom_ci = (np.percentile(bottom_means, 100 * alpha), 
                np.percentile(bottom_means, 100 * (1 - alpha)))
    random_ci = (np.percentile(random_means, 100 * alpha), 
                np.percentile(random_means, 100 * (1 - alpha)))
    
    # Check if CIs overlap
    is_significant = not (bottom_ci[0] <= random_ci[1] and random_ci[0] <= bottom_ci[1])
    
    return bottom_ci, random_ci, is_significant

def compare_pattern_properties(bottom_props, random_props):
    """
    Run Mann-Whitney U tests on pattern properties.
    
    Args:
        bottom_props: Dictionary of property values for bottom authors
        random_props: Dictionary of property values for random authors
    
    Returns:
        dict: Test results for each property
    """
    from scipy.stats import mannwhitneyu
    import numpy as np
    
    results = {}
    for prop in bottom_props:
        if bottom_props[prop] and random_props[prop]:  # Non-empty lists
            try:
                u_stat, p_value = mannwhitneyu(bottom_props[prop], random_props[prop])
                
                # Calculate effect size (r)
                n1 = len(bottom_props[prop])
                n2 = len(random_props[prop])
                effect_size = u_stat / (n1 * n2)  # Normalized U statistic
                
                results[prop] = {
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'bottom_median': np.median(bottom_props[prop]),
                    'random_median': np.median(random_props[prop]),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                results[prop] = {'error': str(e)}
    
    return results

def self_citation_analysis(bottom_authors, random_authors, combined_graph):
    """
    Analyze self-citation patterns between bottom and random authors.
    
    Args:
        bottom_authors: List of bottom author IDs
        random_authors: List of random author IDs
        combined_graph: Combined citation network graph
    
    Returns:
        dict: Self-citation metrics and test results
    """
    import numpy as np
    from scipy.stats import mannwhitneyu
    
    # Calculate self-citation rates for each author
    bottom_self_citation_rates = []
    random_self_citation_rates = []
    
    # Bottom authors
    for author in bottom_authors:
        if author in combined_graph:
            all_citations = combined_graph.out_degree(author) or 1  # Avoid division by zero
            # Count self-loops
            self_citations = 1 if combined_graph.has_edge(author, author) else 0
            rate = self_citations / all_citations
            bottom_self_citation_rates.append(rate)
    
    # Random authors
    for author in random_authors:
        if author in combined_graph:
            all_citations = combined_graph.out_degree(author) or 1  # Avoid division by zero
            # Count self-loops
            self_citations = 1 if combined_graph.has_edge(author, author) else 0
            rate = self_citations / all_citations
            random_self_citation_rates.append(rate)
    
    # Run Mann-Whitney U test
    try:
        u_stat, p_value = mannwhitneyu(bottom_self_citation_rates, random_self_citation_rates)
        
        return {
            'bottom_median_rate': np.median(bottom_self_citation_rates) if bottom_self_citation_rates else None,
            'random_median_rate': np.median(random_self_citation_rates) if random_self_citation_rates else None,
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    except Exception as e:
        return {'error': str(e)}

def citation_exchange_analysis(graph, bad_authors):
    """
    Analyze citation exchange patterns between bad authors.
    
    Args:
        graph: Combined citation network
        bad_authors: Set of bad author IDs
    
    Returns:
        dict: Citation exchange metrics
    """
    # Create subgraph of only bad authors
    bad_subgraph = graph.subgraph(list(bad_authors & set(graph.nodes())))
    
    # Calculate reciprocity in bad author subgraph
    try:
        bad_reciprocity = nx.reciprocity(bad_subgraph)
    except:
        bad_reciprocity = 0
    
    # Create random subgraph of the same size for comparison
    import random
    non_bad_authors = list(set(graph.nodes()) - bad_authors)
    
    if len(non_bad_authors) >= len(bad_authors):
        random_sample = random.sample(non_bad_authors, len(bad_authors))
        random_subgraph = graph.subgraph(random_sample)
        
        try:
            random_reciprocity = nx.reciprocity(random_subgraph)
        except:
            random_reciprocity = 0
            
        exchange_ratio = bad_reciprocity / random_reciprocity if random_reciprocity > 0 else float('inf')
        
        return {
            'bad_reciprocity': bad_reciprocity,
            'random_reciprocity': random_reciprocity,
            'exchange_ratio': exchange_ratio,
            'significant': exchange_ratio > 2.0  # Arbitrary threshold
        }
    else:
        return {
            'bad_reciprocity': bad_reciprocity,
            'error': 'Not enough non-bad authors for comparison'
        }

def perform_network_permutation_test(graph1, graph2, metric_func, n_permutations=1000):
    """
    Perform a permutation test on network metrics.
    
    Args:
        graph1, graph2: NetworkX graphs to compare
        metric_func: Function that calculates metric on a graph
        n_permutations: Number of permutations
    
    Returns:
        tuple: (p_value, observed_difference)
    """
    import numpy as np
    import random
    
    # Calculate observed metric difference
    metric1 = metric_func(graph1)
    metric2 = metric_func(graph2)
    observed_diff = metric1 - metric2
    
    # Combine nodes and randomly reassign
    combined_nodes = list(set(graph1.nodes()) | set(graph2.nodes()))
    
    # Count how many permutations exceed observed difference
    exceed_count = 0
    
    for _ in range(n_permutations):
        # Shuffle nodes
        random.shuffle(combined_nodes)
        split_point = len(graph1.nodes())
        
        # Create permuted graphs (of same size as originals)
        perm_nodes1 = combined_nodes[:split_point]
        perm_nodes2 = combined_nodes[split_point:split_point+len(graph2.nodes())]
        
        perm_graph1 = graph1.subgraph(perm_nodes1)
        perm_graph2 = graph2.subgraph(perm_nodes2)
        
        # Calculate metrics on permuted graphs
        perm_metric1 = metric_func(perm_graph1)
        perm_metric2 = metric_func(perm_graph2)
        perm_diff = perm_metric1 - perm_metric2
        
        if abs(perm_diff) >= abs(observed_diff):
            exceed_count += 1
    
    # Calculate p-value
    p_value = exceed_count / n_permutations
    
    return p_value, observed_diff

def analyze_ring_temporal_patterns(results, citation_years):
    """
    Analyze temporal patterns in citation rings by reading from saved CSV files.
    
    Args:
        results: List of analysis results
        citation_years: Dictionary of citation years
    
    Returns:
        dict: Temporal analysis metrics
    """
    import numpy as np
    import csv
    from pathlib import Path
    
    # Initialize lists to store temporal metrics
    year_ranges = []
    year_stds = []
    
    # Iterate through all results to find the files
    for result in results:
        bottom_author = result["bottom_author"]
        pair_idx = result["pair_idx"]
        
        # Try to read the ring data from the saved CSV
        ring_file = Path(f"{OUTPUT_DIR}/pair{pair_idx}_BOTTOM_{bottom_author}_citation_rings.csv")
        
        if ring_file.exists():
            try:
                with open(ring_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract temporal data if available
                        if 'year_range' in row and row['year_range']:
                            try:
                                year_range = float(row['year_range'])
                                year_ranges.append(year_range)
                            except (ValueError, TypeError):
                                pass
                        
                        if 'year_std' in row and row['year_std']:
                            try:
                                year_std = float(row['year_std'])
                                year_stds.append(year_std)
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                print(f"[WARNING] Could not read ring file {ring_file}: {e}")
    
    # Calculate metrics if we have temporal data
    if year_ranges:
        return {
            'median_year_range': np.median(year_ranges),
            'mean_year_range': np.mean(year_ranges),
            'median_year_std': np.median(year_stds) if year_stds else None,
            'compact_rings_pct': sum(1 for yr in year_ranges if yr <= 2) / len(year_ranges) * 100
        }
    else:
        return {'error': 'No temporal data available'}

def save_statistical_results(results, output_path):
    """
    Save statistical analysis results to CSV.
    
    Args:
        results: Dictionary of statistical test results
        output_path: Path to save CSV file
    """
    with open(output_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Test', 'Metric', 'Value', 'Significant'])
        
        # Write flattened results
        for test_name, test_results in results.items():
            if isinstance(test_results, dict):
                for metric, value in test_results.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for sub_metric, sub_value in value.items():
                            significant = value.get('significant', 'N/A')
                            writer.writerow([test_name, f"{metric}_{sub_metric}", sub_value, significant])
                    else:
                        # Direct metrics
                        if metric == 'significant':
                            significant = value
                        else:
                            significant = 'N/A'
                        writer.writerow([test_name, metric, value, significant])
        

##############################################################################
# 9) Single Author Analysis
##############################################################################
def analyze_author_network(rolap_conn, impact_conn, author_id, output_prefix, schema):
    """
    Perform comprehensive analysis on a single author's citation network.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        author_id: The author's ORCID to analyze
        output_prefix: Prefix for output files
        schema: Database schema information
        
    Returns:
        tuple: (graph, cliques, rings, author_analysis)
    """
    print(f"[INFO] Starting comprehensive analysis for author {author_id}")
    
    # Step 1: Load data
    bad_authors = load_bad_authors(rolap_conn, schema)
    publication_years = load_publication_years(impact_conn, schema)  # Changed to impact_conn
    author_metrics = load_author_metrics(rolap_conn, schema)
    orcid_to_works, work_to_authors = load_work_author_mappings(rolap_conn, schema)
    work_to_cited, citation_years = load_citation_data(impact_conn, schema)
    
    # Step 2: Build citation network
    graph = build_citation_network(
        author_id, 
        DEFAULT_DEPTH,
        MAX_BFS_NODES,
        orcid_to_works, 
        work_to_authors, 
        work_to_cited,
        citation_years
    )
    
    # Save basic graph metrics
    try:
        with open(f"{OUTPUT_DIR}/{output_prefix}_graph_metrics.txt", 'w') as f:
            f.write(f"Author ID: {author_id}\n")
            f.write(f"Number of nodes: {graph.number_of_nodes()}\n")
            f.write(f"Number of edges: {graph.number_of_edges()}\n")
            if graph.number_of_nodes() > 0:
                f.write(f"Density: {nx.density(graph):.6f}\n")
                f.write(f"Bad authors in network: {sum(1 for n in graph.nodes() if n in bad_authors)}\n")
                f.write(f"Fraction of bad authors: {sum(1 for n in graph.nodes() if n in bad_authors) / graph.number_of_nodes():.4f}\n")
                
                # Add publication year statistics if available
                if publication_years:
                    # Get years for works authored by this author
                    author_works = orcid_to_works.get(author_id, [])
                    author_years = [publication_years.get(work_id) for work_id in author_works 
                                   if work_id in publication_years]
                    
                    if author_years:
                        f.write(f"\nPublication Years:\n")
                        f.write(f"  Earliest: {min(author_years)}\n")
                        f.write(f"  Latest: {max(author_years)}\n")
                        f.write(f"  Count: {len(author_years)}\n")
                        
                        # Calculate publication frequency by year
                        from collections import Counter
                        year_counts = Counter(author_years)
                        f.write(f"\nPublications by Year:\n")
                        for year in sorted(year_counts.keys()):
                            f.write(f"  {year}: {year_counts[year]}\n")
    except Exception as e:
        print(f"[WARNING] Could not save graph metrics: {e}")
    
    # Pass publication years to functions that can utilize them
    # Step 3: Find suspicious cliques
    suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
    
    # Step 4: Find citation rings
    citation_rings = find_citation_rings(graph, bad_authors)
    
    # Step 5: Analyze author citation behavior with publication years
    author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_metrics)
    
    # Step 6: Generate visualizations
    if suspicious_cliques:
        for i, clique in enumerate(suspicious_cliques[:5]):
            output_path = f"{OUTPUT_DIR}/{output_prefix}_suspicious_clique_{i+1}.png"
            visualize_suspicious_clique(graph, clique, bad_authors, output_path)
    
    if citation_rings:
        for i, ring in enumerate(citation_rings[:5]):
            output_path = f"{OUTPUT_DIR}/{output_prefix}_citation_ring_{i+1}.png"
            visualize_citation_ring(graph, ring, bad_authors, output_path)
    
    if author_analysis:
        visualize_author_network(
            graph, 
            author_analysis, 
            bad_authors, 
            f"{OUTPUT_DIR}/{output_prefix}_author_network.png"
        )
    
    # Step 7: Save metrics to CSV
    save_clique_metrics(suspicious_cliques, f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv")
    save_ring_metrics(citation_rings, f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv")
    save_author_metrics(author_analysis, f"{OUTPUT_DIR}/{output_prefix}_author_analysis.csv")
    
    print(f"[INFO] Analysis complete for author {author_id}")
    return graph, suspicious_cliques, citation_rings, author_analysis


##############################################################################
# 10) Batch Analysis for Multiple Author Pairs
##############################################################################
def batch_analyze_authors(rolap_conn, impact_conn, schema, num_pairs=5):
    """
    Analyze multiple author pairs and generate comparative reports.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        schema: Database schema information
        num_pairs: Number of author pairs to analyze
        
    Returns:
        list: Analysis results for each pair
    """
    print(f"[INFO] Starting batch analysis for {num_pairs} author pairs")
    
    # Step 1: Load data
    bad_authors = load_bad_authors(rolap_conn, schema)
    publication_years = load_publication_years(impact_conn, schema)  # Changed to impact_conn
    orcid_to_works, work_to_authors = load_work_author_mappings(rolap_conn, schema)
    work_to_cited, citation_years = load_citation_data(impact_conn, schema)
    author_pairs = load_author_pairs(rolap_conn, schema, num_pairs)
    
    if not author_pairs:
        print("[WARNING] No author pairs to analyze.")
        return []
    
    # Step 2: Prepare results collection
    results = []
    
    # Prepare data structures for statistical analysis
    all_bottom_clique_counts = []
    all_random_clique_counts = []
    all_bottom_ring_counts = []
    all_random_ring_counts = []
    
    bottom_clique_properties = {
        'size': [], 'density': [], 'bad_fraction': [], 
        'avg_edge_weight': [], 'reciprocity': []
    }
    
    random_clique_properties = {
        'size': [], 'density': [], 'bad_fraction': [], 
        'avg_edge_weight': [], 'reciprocity': []
    }
    
    bottom_ring_properties = {
        'size': [], 'bad_fraction': [], 'avg_edge_weight': [], 
        'weight_consistency': []
    }
    
    random_ring_properties = {
        'size': [], 'bad_fraction': [], 'avg_edge_weight': [], 
        'weight_consistency': []
    }
    
    all_bottom_authors = []
    all_random_authors = []
    combined_graph = nx.DiGraph()
    
    # Add temporal analysis structures
    bottom_publication_years = {}
    random_publication_years = {}
    
    # Step 3: Process each author pair
    for pair_idx, (bottom_orcid, random_orcid, total_works) in enumerate(author_pairs, 1):
        print(f"\n[INFO] Processing pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM)")
        
        pair_results = {}
        all_bottom_authors.append(bottom_orcid)
        all_random_authors.append(random_orcid)
        
        # Extract publication years for both authors
        bottom_works = orcid_to_works.get(bottom_orcid, [])
        random_works = orcid_to_works.get(random_orcid, [])
        
        bottom_years = [publication_years.get(work_id) for work_id in bottom_works 
                       if work_id in publication_years]
        random_years = [publication_years.get(work_id) for work_id in random_works 
                       if work_id in publication_years]
        
        bottom_publication_years[bottom_orcid] = bottom_years
        random_publication_years[random_orcid] = random_years
        
        # Analyze each author in the pair
        for author_id, category in [(bottom_orcid, "BOTTOM"), (random_orcid, "RANDOM")]:
            print(f"[INFO] Analyzing {category} author {author_id}")
            
            # Build citation network
            graph = build_citation_network(
                author_id, 
                DEFAULT_DEPTH,
                MAX_BFS_NODES,
                orcid_to_works, 
                work_to_authors, 
                work_to_cited,
                citation_years
            )
            
            # Add to combined graph for global analysis
            combined_graph.add_nodes_from(graph.nodes())
            combined_graph.add_edges_from(graph.edges(data=True))
            
            # Find suspicious patterns
            suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
            citation_rings = find_citation_rings(graph, bad_authors)
            author_analysis = analyze_author_citation_behavior(graph, bad_authors)
            
            # Update statistical data structures
            if category == "BOTTOM":
                all_bottom_clique_counts.append(len(suspicious_cliques))
                all_bottom_ring_counts.append(len(citation_rings))
                
                # Collect properties for Mann-Whitney U test
                for clique in suspicious_cliques:
                    for prop in bottom_clique_properties:
                        if prop in clique:
                            bottom_clique_properties[prop].append(clique[prop])
                
                for ring in citation_rings:
                    for prop in bottom_ring_properties:
                        if prop in ring:
                            bottom_ring_properties[prop].append(ring[prop])
            else:
                all_random_clique_counts.append(len(suspicious_cliques))
                all_random_ring_counts.append(len(citation_rings))
                
                # Collect properties for Mann-Whitney U test
                for clique in suspicious_cliques:
                    for prop in random_clique_properties:
                        if prop in clique:
                            random_clique_properties[prop].append(clique[prop])
                
                for ring in citation_rings:
                    for prop in random_ring_properties:
                        if prop in ring:
                            random_ring_properties[prop].append(ring[prop])
            
            # Compute summary metrics
            bad_node_count = sum(1 for n in graph.nodes() if n in bad_authors)
            bad_edge_count = sum(1 for u, v in graph.edges() if u in bad_authors and v in bad_authors)
            
            # Network metrics
            network_metrics = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
                "bad_node_fraction": bad_node_count / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
                "bad_edge_fraction": bad_edge_count / graph.number_of_edges() if graph.number_of_edges() > 0 else 0,
                "suspicious_cliques": len(suspicious_cliques),
                "citation_rings": len(citation_rings),
                "top_clique_size": suspicious_cliques[0]['size'] if suspicious_cliques else 0,
                "top_clique_score": suspicious_cliques[0]['anomaly_score'] if suspicious_cliques else 0,
                "top_ring_size": citation_rings[0]['size'] if citation_rings else 0,
                "top_ring_score": citation_rings[0]['anomaly_score'] if citation_rings else 0
            }
            
            # Add temporal metrics
            author_years = bottom_years if category == "BOTTOM" else random_years
            if author_years:
                network_metrics["publication_years"] = len(author_years)
                network_metrics["earliest_year"] = min(author_years)
                network_metrics["latest_year"] = max(author_years)
                network_metrics["publication_span"] = max(author_years) - min(author_years) if len(author_years) > 1 else 0
            
            # Save visualization for this author
            output_prefix = f"pair{pair_idx}_{category}_{author_id}"
            
            # Save publication year information
            try:
                if author_years:
                    with open(f"{OUTPUT_DIR}/{output_prefix}_publication_years.txt", 'w') as f:
                        f.write(f"Author: {author_id} ({category})\n")
                        f.write(f"Number of works with year data: {len(author_years)}\n")
                        f.write(f"Publication span: {min(author_years)} - {max(author_years)}\n")
                        
                        # Calculate year distribution
                        from collections import Counter
                        year_counts = Counter(author_years)
                        f.write("\nPublications by year:\n")
                        for year in sorted(year_counts.keys()):
                            f.write(f"{year}: {year_counts[year]}\n")
            except Exception as e:
                print(f"[WARNING] Could not save publication year data: {e}")
            
            # Visualize network overview
            if author_analysis:
                visualize_author_network(
                    graph, 
                    author_analysis, 
                    bad_authors, 
                    f"{OUTPUT_DIR}/{output_prefix}_network.png"
                )
            
            # Visualize top clique and ring if they exist
            if suspicious_cliques:
                visualize_suspicious_clique(
                    graph, 
                    suspicious_cliques[0], 
                    bad_authors, 
                    f"{OUTPUT_DIR}/{output_prefix}_top_clique.png"
                )
                
                # Save clique metrics
                save_clique_metrics(
                    suspicious_cliques[:10],  # Save top 10
                    f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv"
                )
            
            if citation_rings:
                visualize_citation_ring(
                    graph, 
                    citation_rings[0], 
                    bad_authors, 
                    f"{OUTPUT_DIR}/{output_prefix}_top_ring.png"
                )
                
                # Save ring metrics
                save_ring_metrics(
                    citation_rings[:10],  # Save top 10
                    f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv"
                )
            
            # Store results
            pair_results[category] = {
                "author_id": author_id,
                "metrics": network_metrics,
                "suspicious_cliques": suspicious_cliques[:10],  # Store top 10
                "citation_rings": citation_rings[:10],  # Store top 10
                "author_analysis": author_analysis
            }
        
        # Add to overall results
        results.append({
            "pair_idx": pair_idx,
            "bottom_author": pair_results["BOTTOM"]["author_id"],
            "random_author": pair_results["RANDOM"]["author_id"],
            "total_works": total_works,
            "bottom_metrics": pair_results["BOTTOM"]["metrics"],
            "random_metrics": pair_results["RANDOM"]["metrics"]
        })
    
    # Save comparative report
    save_comparative_analysis(results, f"{OUTPUT_DIR}/comparative_analysis.csv")
    
    # Step 4: Perform statistical analysis after all pairs are processed
    print("[INFO] Performing statistical analysis on collected data...")
    
    statistical_results = {}
    
    # 1. Bootstrap confidence intervals for clique and ring counts
    clique_counts = {
        'bottom': all_bottom_clique_counts,
        'random': all_random_clique_counts
    }
    
    ring_counts = {
        'bottom': all_bottom_ring_counts,
        'random': all_random_ring_counts
    }
    
    ci_cliques = bootstrap_citation_pattern(clique_counts)
    ci_rings = bootstrap_citation_pattern(ring_counts)
    
    statistical_results['bootstrap_cliques'] = {
        'bottom_ci': ci_cliques[0],
        'random_ci': ci_cliques[1],
        'significant': ci_cliques[2]
    }
    
    statistical_results['bootstrap_rings'] = {
        'bottom_ci': ci_rings[0],
        'random_ci': ci_rings[1],
        'significant': ci_rings[2]
    }
    
    # 2. Mann-Whitney U tests for clique and ring properties
    mw_clique_results = compare_pattern_properties(bottom_clique_properties, random_clique_properties)
    mw_ring_results = compare_pattern_properties(bottom_ring_properties, random_ring_properties)
    
    statistical_results['mannwhitney_clique_properties'] = mw_clique_results
    statistical_results['mannwhitney_ring_properties'] = mw_ring_results
    
    # 3. Self-citation analysis
    self_citation_results = self_citation_analysis(
        all_bottom_authors, all_random_authors, combined_graph
    )
    statistical_results['self_citation_analysis'] = self_citation_results
    
    # 4. Citation exchange analysis
    exchange_results = citation_exchange_analysis(combined_graph, bad_authors)
    statistical_results['citation_exchange_analysis'] = exchange_results
    
    # 5. Ring temporal pattern analysis
    temporal_results = analyze_ring_temporal_patterns(results, citation_years)
    statistical_results['ring_temporal_analysis'] = temporal_results
    
    # 6. Publication year comparison analysis
    try:
        import numpy as np
        from scipy.stats import mannwhitneyu
        
        # Flatten years for each group
        bottom_all_years = [year for years in bottom_publication_years.values() for year in years if year]
        random_all_years = [year for years in random_publication_years.values() for year in years if year]
        
        if bottom_all_years and random_all_years:
            # Calculate basic statistics
            temporal_results = {
                'bottom_earliest': min(bottom_all_years),
                'bottom_latest': max(bottom_all_years),
                'bottom_median_year': np.median(bottom_all_years),
                'bottom_year_span': max(bottom_all_years) - min(bottom_all_years),
                'random_earliest': min(random_all_years),
                'random_latest': max(random_all_years),
                'random_median_year': np.median(random_all_years),
                'random_year_span': max(random_all_years) - min(random_all_years),
            }
            
            # Statistical test for year distribution
            try:
                u_stat, p_value = mannwhitneyu(bottom_all_years, random_all_years)
                temporal_results['year_distribution_p_value'] = p_value
                temporal_results['year_distribution_significant'] = p_value < 0.05
            except Exception as e:
                temporal_results['year_distribution_error'] = str(e)
                
            statistical_results['publication_year_analysis'] = temporal_results
    except Exception as e:
        print(f"[WARNING] Error in publication year analysis: {e}")
        import traceback
        traceback.print_exc()

    # 7. Network permutation test for clustering coefficient
    # Create undirected graphs for bottom and random authors
    bottom_graph = nx.Graph()
    random_graph = nx.Graph()
    
    # Add edges for each author category
    for result in results:
        bottom_author = result["bottom_author"]
        random_author = result["random_author"]
        
        # Get all neighbors from combined graph
        for u, v, data in combined_graph.edges(data=True):
            if u == bottom_author or v == bottom_author:
                bottom_graph.add_edge(u, v)
            if u == random_author or v == random_author:
                random_graph.add_edge(u, v)
    
    # Perform permutation test on clustering coefficient
    clustering_perm_p, clustering_diff = perform_network_permutation_test(
        bottom_graph, random_graph, 
        lambda g: nx.average_clustering(g) if g.number_of_nodes() > 1 else 0
    )
    
    statistical_results['permutation_clustering'] = {
        'p_value': clustering_perm_p,
        'observed_diff': clustering_diff,
        'significant': clustering_perm_p < 0.05
    }
    
    # Save all statistical results
    save_statistical_results(
        statistical_results, 
        f"{OUTPUT_DIR}/statistical_analysis.csv"
    )
    
    print(f"[INFO] Batch analysis complete for {len(results)} author pairs")
    return results

def analyze_temporal_citation_patterns(graph, work_to_authors, publication_years, citation_years):
    """
    Analyze temporal patterns in citation network.
    
    Args:
        graph: NetworkX directed graph of the citation network
        work_to_authors: Dictionary mapping work IDs to author ORCIDs
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        
    Returns:
        dict: Temporal analysis metrics
    """
    import numpy as np
    from collections import defaultdict
    
    # Initialize metrics
    temporal_metrics = {
        'citation_age': [],  # Years between publication and citation
        'citation_speed': defaultdict(list),  # Speed of citations by author
        'citation_span': [],  # Publication year range for citations
        'self_citation_age': [],  # Age of self-citations
        'other_citation_age': [],  # Age of non-self citations
    }
    
    # Analyze each edge in the graph
    for source, target, data in graph.edges(data=True):
        # Extract citation relationships
        for work_id, cited_work_id in data.get('citations', []):
            # Skip if we don't have publication years for both works
            if work_id not in publication_years or cited_work_id not in publication_years:
                continue
                
            source_year = publication_years[work_id]
            target_year = publication_years[cited_work_id]
            
            # Calculate citation age (how old was the paper when cited)
            citation_age = source_year - target_year
            temporal_metrics['citation_age'].append(citation_age)
            
            # Record by author for citation speed analysis
            temporal_metrics['citation_speed'][source].append(citation_age)
            
            # Check if this is a self-citation
            source_authors = set(work_to_authors.get(work_id, []))
            target_authors = set(work_to_authors.get(cited_work_id, []))
            
            # Check for author overlap (self-citation)
            if source_authors.intersection(target_authors):
                temporal_metrics['self_citation_age'].append(citation_age)
            else:
                temporal_metrics['other_citation_age'].append(citation_age)
    
    # Calculate aggregate metrics
    results = {}
    
    if temporal_metrics['citation_age']:
        results['median_citation_age'] = np.median(temporal_metrics['citation_age'])
        results['mean_citation_age'] = np.mean(temporal_metrics['citation_age'])
        results['recent_citation_pct'] = sum(1 for age in temporal_metrics['citation_age'] if age <= 2) / len(temporal_metrics['citation_age']) * 100
        
    if temporal_metrics['self_citation_age'] and temporal_metrics['other_citation_age']:
        results['median_self_citation_age'] = np.median(temporal_metrics['self_citation_age'])
        results['median_other_citation_age'] = np.median(temporal_metrics['other_citation_age'])
        results['self_vs_other_diff'] = results['median_self_citation_age'] - results['median_other_citation_age']
    
    # Calculate citation speed by author (lower values mean faster citations)
    author_citation_speeds = {}
    for author, ages in temporal_metrics['citation_speed'].items():
        if len(ages) >= 3:  # Only consider authors with enough citations
            author_citation_speeds[author] = np.median(ages)
    
    # Get fastest citers (potentially suspicious)
    if author_citation_speeds:
        fastest_citers = sorted(author_citation_speeds.items(), key=lambda x: x[1])[:5]
        results['fastest_citers'] = fastest_citers
    
    return results


def visualize_temporal_patterns(publication_years, citation_years, author_id, output_path):
    """
    Create visualizations for temporal citation patterns.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Publication years histogram
    pub_years = [year for work_id, year in publication_years.items()]
    if pub_years:
        axs[0, 0].hist(pub_years, bins=min(20, len(set(pub_years))), alpha=0.7, color='blue')
        axs[0, 0].set_title('Publication Year Distribution')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Number of Publications')
        
        # Add a vertical line for median
        median_year = np.median(pub_years)
        axs[0, 0].axvline(x=median_year, color='red', linestyle='--', 
                         label=f'Median: {median_year:.1f}')
        axs[0, 0].legend()
    
    # 2. Citation age histogram
    citation_ages = []
    for (source_id, target_id), cite_year in citation_years.items():
        if source_id in publication_years and target_id in publication_years:
            citation_age = publication_years[source_id] - publication_years[target_id]
            citation_ages.append(citation_age)
    
    if citation_ages:
        axs[0, 1].hist(citation_ages, bins=min(20, len(set(citation_ages))), alpha=0.7, color='green')
        axs[0, 1].set_title('Citation Age Distribution')
        axs[0, 1].set_xlabel('Age of Cited Work (Years)')
        axs[0, 1].set_ylabel('Number of Citations')
        
        # Add a vertical line for median
        median_age = np.median(citation_ages)
        axs[0, 1].axvline(x=median_age, color='red', linestyle='--', 
                         label=f'Median Age: {median_age:.1f}')
        axs[0, 1].legend()
    
    # 3. Publication frequency over time
    if pub_years:
        # Count publications per year
        year_counts = Counter(pub_years)
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        # Create the line plot
        axs[1, 0].plot(years, counts, marker='o', linestyle='-', color='purple')
        axs[1, 0].set_title('Publication Frequency Over Time')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Number of Publications')
        
        # Add trend line
        if len(years) > 1:
            z = np.polyfit(years, counts, 1)
            p = np.poly1d(z)
            axs[1, 0].plot(years, p(years), "r--", 
                          label=f'Trend: {"+" if z[0]>0 else ""}{z[0]:.2f}/year')
            axs[1, 0].legend()
    
    # 4. Citation network growth over time
    if citation_years:
        # Extract and sort years
        cite_years = sorted(citation_years.values())
        year_counts = Counter(cite_years)
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        # Calculate cumulative counts
        cumulative = np.cumsum(counts)
        
        # Create the line plot
        axs[1, 1].plot(years, cumulative, marker='o', linestyle='-', color='orange')
        axs[1, 1].set_title('Cumulative Citation Network Growth')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Cumulative Number of Citations')
    
    # Add a super title
    plt.suptitle(f'Temporal Analysis for Author: {author_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved temporal analysis visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


def analyze_citation_bursts(publication_years, citation_years, author_id, output_path=None):
    """
    Analyze citation pattern bursts to detect potentially suspicious activity.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Optional path to save visualization
        
    Returns:
        dict: Burst analysis metrics
    """
    import numpy as np
    from collections import defaultdict, Counter
    import matplotlib.pyplot as plt
    
    # Organize citations by year
    citations_by_year = defaultdict(int)
    
    # For each citation, if we have the publication year, count it
    for (source_id, target_id), cite_year in citation_years.items():
        if source_id in publication_years:
            citations_by_year[publication_years[source_id]] += 1
    
    # Skip if we don't have enough data
    if len(citations_by_year) < 3:
        return {'error': 'Not enough citation year data'}
    
    # Get years and counts
    years = sorted(citations_by_year.keys())
    counts = [citations_by_year[year] for year in years]
    
    # Calculate moving average and standard deviation (window of 3 years)
    window = 3
    ma_counts = []
    std_counts = []
    burst_scores = []
    
    for i in range(len(counts)):
        # Get window indexes ensuring we don't go out of bounds
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_counts = counts[start_idx:end_idx]
        
        # Calculate stats
        ma = np.mean(window_counts)
        std = np.std(window_counts) if len(window_counts) > 1 else 0
        
        ma_counts.append(ma)
        std_counts.append(std)
        
        # Calculate burst score (how many std deviations above moving average)
        burst_score = (counts[i] - ma) / std if std > 0 else 0
        burst_scores.append(burst_score)
    
    # Identify burst years (years with burst score > 2)
    burst_years = [years[i] for i in range(len(years)) if burst_scores[i] > 2]
    burst_intensities = [burst_scores[i] for i in range(len(years)) if burst_scores[i] > 2]
    
    # Calculate metrics
    results = {
        'burst_count': len(burst_years),
        'burst_years': burst_years,
        'burst_intensities': burst_intensities,
        'max_burst_intensity': max(burst_scores) if burst_scores else 0,
        'suspicious': len(burst_years) >= 2  # Consider suspicious if multiple bursts
    }
    
    # Create visualization if output path is provided
    if output_path:
        plt.figure(figsize=(12, 8))
        
        # Plot citation counts
        plt.bar(years, counts, alpha=0.7, color='blue', label='Citation Count')
        
        # Plot moving average
        plt.plot(years, ma_counts, 'r-', label=f'{window}-Year Moving Average')
        
        # Highlight burst years
        for year, intensity in zip(burst_years, burst_intensities):
            plt.axvline(x=year, color='red', alpha=0.3)
            plt.annotate(f'Burst: {intensity:.1f}σ', 
                        xy=(year, max(counts)*0.9), 
                        xytext=(year, max(counts)*0.9),
                        ha='center')
        
        plt.title(f'Citation Pattern Analysis for Author: {author_id}')
        plt.xlabel('Year')
        plt.ylabel('Number of Citations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved citation burst analysis to {output_path}")
        except Exception as e:
            print(f"[WARNING] Could not save visualization: {e}")
            
        plt.close()
    
    return results

##############################################################################
# 11) Main Function
##############################################################################
def main():
    """Main entry point for the script."""
    try:
        # Connect to databases
        print("[INFO] Connecting to databases...")
        rolap_conn = sqlite3.connect(ROLAP_DB)
        impact_conn = sqlite3.connect(IMPACT_DB)
        
        # Detect schema
        schema = detect_schema(rolap_conn, impact_conn)
        
        # Create output directory if it doesn't exist
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True)
            print(f"[INFO] Created output directory: {OUTPUT_DIR}")
        
        # Load data for analysis
        print("[INFO] Loading data for analysis...")
        bad_authors = load_bad_authors(rolap_conn, schema)
        orcid_to_works, work_to_authors = load_work_author_mappings(rolap_conn, schema)
        work_to_cited, citation_years = load_citation_data(impact_conn, schema)
        publication_years = load_publication_years(impact_conn, schema)  # Use impact_conn
        author_metrics = load_author_metrics(rolap_conn, schema)
        
        # Run batch analysis for multiple authors
        batch_results = batch_analyze_authors(rolap_conn, impact_conn, schema, num_pairs=5)
        
        # Add temporal analysis for each author if we have results
        if batch_results:
            # Create a directory for temporal analysis
            temporal_dir = Path(f"{OUTPUT_DIR}/temporal_analysis")
            if not temporal_dir.exists():
                temporal_dir.mkdir()
                print(f"[INFO] Created directory for temporal analysis: {temporal_dir}")
            
            print("[INFO] Performing temporal analysis for all authors...")
            temporal_metrics = {}
            
            for result in batch_results:
                # Process bottom author
                bottom_author = result["bottom_author"]
                try:
                    # Build citation network for bottom author
                    bottom_graph = build_citation_network(
                        bottom_author, 
                        DEFAULT_DEPTH,
                        MAX_BFS_NODES,
                        orcid_to_works, 
                        work_to_authors, 
                        work_to_cited,
                        citation_years
                    )
                    
                    # Analyze temporal citation patterns
                    bottom_metrics = analyze_temporal_citation_patterns(
                        bottom_graph, 
                        work_to_authors, 
                        publication_years, 
                        citation_years
                    )
                    
                    # Create visualizations
                    visualize_temporal_patterns(
                        publication_years,
                        citation_years,
                        bottom_author,
                        f"{temporal_dir}/BOTTOM_{bottom_author}_temporal.png"
                    )
                    
                    # Analyze citation bursts
                    burst_metrics = analyze_citation_bursts(
                        publication_years,
                        citation_years,
                        bottom_author,
                        f"{temporal_dir}/BOTTOM_{bottom_author}_bursts.png"
                    )
                    
                    # Store metrics
                    temporal_metrics[f"BOTTOM_{bottom_author}"] = {
                        "temporal": bottom_metrics,
                        "bursts": burst_metrics
                    }
                except Exception as e:
                    print(f"[WARNING] Could not complete temporal analysis for BOTTOM author {bottom_author}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Process random author
                random_author = result["random_author"]
                try:
                    # Build citation network for random author
                    random_graph = build_citation_network(
                        random_author, 
                        DEFAULT_DEPTH,
                        MAX_BFS_NODES,
                        orcid_to_works, 
                        work_to_authors, 
                        work_to_cited,
                        citation_years
                    )
                    
                    # Analyze temporal citation patterns
                    random_metrics = analyze_temporal_citation_patterns(
                        random_graph, 
                        work_to_authors, 
                        publication_years, 
                        citation_years
                    )
                    
                    # Create visualizations
                    visualize_temporal_patterns(
                        publication_years,
                        citation_years,
                        random_author,
                        f"{temporal_dir}/RANDOM_{random_author}_temporal.png"
                    )
                    
                    # Analyze citation bursts
                    burst_metrics = analyze_citation_bursts(
                        publication_years,
                        citation_years,
                        random_author,
                        f"{temporal_dir}/RANDOM_{random_author}_bursts.png"
                    )
                    
                    # Store metrics
                    temporal_metrics[f"RANDOM_{random_author}"] = {
                        "temporal": random_metrics,
                        "bursts": burst_metrics
                    }
                except Exception as e:
                    print(f"[WARNING] Could not complete temporal analysis for RANDOM author {random_author}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save temporal metrics summary
            try:
                import json
                with open(f"{temporal_dir}/temporal_metrics_summary.json", 'w') as f:
                    json.dump(temporal_metrics, f, indent=4)
                print(f"[INFO] Saved temporal metrics summary to {temporal_dir}/temporal_metrics_summary.json")
                
                # Create a readable summary
                with open(f"{temporal_dir}/temporal_analysis_summary.txt", 'w') as f:
                    f.write("TEMPORAL CITATION ANALYSIS SUMMARY\n")
                    f.write("================================\n\n")
                    
                    # Compare burst patterns between bottom and random authors
                    bottom_burst_count = sum(1 for k, v in temporal_metrics.items() 
                                          if k.startswith("BOTTOM_") and 
                                          v.get("bursts", {}).get("suspicious", False))
                    
                    random_burst_count = sum(1 for k, v in temporal_metrics.items() 
                                          if k.startswith("RANDOM_") and 
                                          v.get("bursts", {}).get("suspicious", False))
                    
                    f.write(f"Bottom Authors with Suspicious Citation Bursts: {bottom_burst_count}\n")
                    f.write(f"Random Authors with Suspicious Citation Bursts: {random_burst_count}\n\n")
                    
                    # List authors with suspicious burst patterns
                    f.write("Authors with Suspicious Citation Burst Patterns:\n")
                    for author_key, metrics in temporal_metrics.items():
                        if metrics.get("bursts", {}).get("suspicious", False):
                            burst_years = metrics.get("bursts", {}).get("burst_years", [])
                            burst_intensities = metrics.get("bursts", {}).get("burst_intensities", [])
                            
                            f.write(f"  {author_key}:\n")
                            f.write(f"    - Burst Years: {', '.join(str(y) for y in burst_years)}\n")
                            f.write(f"    - Burst Intensities: {', '.join(f'{i:.1f}σ' for i in burst_intensities)}\n")
                            f.write(f"    - Max Burst Intensity: {metrics.get('bursts', {}).get('max_burst_intensity', 0):.1f}σ\n\n")
                    
                    # Compare citation ages
                    f.write("Citation Age Analysis:\n")
                    bottom_ages = [v.get("temporal", {}).get("median_citation_age", None) 
                                for k, v in temporal_metrics.items() if k.startswith("BOTTOM_")]
                    random_ages = [v.get("temporal", {}).get("median_citation_age", None) 
                                for k, v in temporal_metrics.items() if k.startswith("RANDOM_")]
                    
                    bottom_ages = [age for age in bottom_ages if age is not None]
                    random_ages = [age for age in random_ages if age is not None]
                    
                    if bottom_ages and random_ages:
                        import numpy as np
                        bottom_median = np.median(bottom_ages)
                        random_median = np.median(random_ages)
                        
                        f.write(f"  Bottom Authors - Median Citation Age: {bottom_median:.1f} years\n")
                        f.write(f"  Random Authors - Median Citation Age: {random_median:.1f} years\n")
                        
                        if bottom_median < random_median:
                            f.write(f"  NOTE: Bottom authors tend to cite more recent work ({bottom_median:.1f} vs {random_median:.1f} years),\n")
                            f.write(f"        which could indicate strategic citation behavior.\n\n")
                    
                    # Compare self-citation patterns
                    f.write("Self-Citation Analysis:\n")
                    bottom_self_diff = [v.get("temporal", {}).get("self_vs_other_diff", None) 
                                     for k, v in temporal_metrics.items() if k.startswith("BOTTOM_")]
                    random_self_diff = [v.get("temporal", {}).get("self_vs_other_diff", None) 
                                     for k, v in temporal_metrics.items() if k.startswith("RANDOM_")]
                    
                    bottom_self_diff = [diff for diff in bottom_self_diff if diff is not None]
                    random_self_diff = [diff for diff in random_self_diff if diff is not None]
                    
                    if bottom_self_diff and random_self_diff:
                        bottom_avg_diff = np.mean(bottom_self_diff)
                        random_avg_diff = np.mean(random_self_diff)
                        
                        f.write(f"  Bottom Authors - Self vs. Other Citation Age Difference: {bottom_avg_diff:.1f} years\n")
                        f.write(f"  Random Authors - Self vs. Other Citation Age Difference: {random_avg_diff:.1f} years\n")
                        
                        if bottom_avg_diff < random_avg_diff:
                            f.write(f"  NOTE: Bottom authors tend to self-cite more recent work compared to other citations\n")
                            f.write(f"        ({bottom_avg_diff:.1f} vs {random_avg_diff:.1f} years difference), which could indicate\n")
                            f.write(f"        strategic self-citation to boost metrics.\n\n")
                    
                    f.write("\nRECOMMENDATIONS:\n")
                    f.write("1. Authors with multiple high-intensity citation bursts should be investigated further.\n")
                    f.write("2. Authors with significant differences between self-citation age and other-citation age\n")
                    f.write("   may be strategically self-citing to boost metrics.\n")
                    f.write("3. Authors who predominantly cite very recent work compared to their peers may be\n")
                    f.write("   engaging in strategic citation behavior.\n")
                
                print(f"[INFO] Saved temporal analysis summary to {temporal_dir}/temporal_analysis_summary.txt")
            except Exception as e:
                print(f"[WARNING] Could not save temporal metrics summary: {e}")
                import traceback
                traceback.print_exc()
        
        # Report key statistical findings
        try:
            with open(f"{OUTPUT_DIR}/statistical_summary.txt", 'w') as f:
                f.write("CITATION NETWORK STATISTICAL ANALYSIS SUMMARY\n")
                f.write("===========================================\n\n")
                
                # Attempt to read and summarize the statistical results
                import csv
                significant_findings = []
                
                try:
                    with open(f"{OUTPUT_DIR}/statistical_analysis.csv", 'r') as stats_file:
                        reader = csv.reader(stats_file)
                        next(reader)  # Skip header
                        for row in reader:
                            if len(row) >= 4 and row[3].lower() == 'true':
                                test = row[0]
                                metric = row[1]
                                value = row[2]
                                significant_findings.append((test, metric, value))
                except:
                    pass
                
                if significant_findings:
                    f.write("Significant Statistical Findings:\n")
                    for test, metric, value in significant_findings:
                        f.write(f"- {test}: {metric} = {value}\n")
                else:
                    f.write("No statistically significant findings were detected.\n")
                    f.write("This may be due to sample size limitations or weakness in the patterns.\n")
                
                f.write("\nRecommendations:\n")
                f.write("1. Further investigation is needed for authors with high ring or clique counts.\n")
                f.write("2. Temporal analysis may reveal coordinated citation behavior.\n")
                f.write("3. Self-citation and exchange ratio analysis can identify additional suspicious patterns.\n")
                
                # Add new recommendation about temporal patterns
                f.write("4. Citation burst patterns and abnormal citation ages may indicate citation manipulation.\n")
        except Exception as e:
            print(f"[WARNING] Could not generate statistical summary: {e}")
        
        # Perform deeper analysis on the most suspicious author if we have results
        if batch_results:
            try:
                # Find the author with the most suspicious cliques
                most_suspicious = max(batch_results, 
                                    key=lambda x: x["bottom_metrics"]["suspicious_cliques"])
                
                bottom_author = most_suspicious["bottom_author"]
                print(f"\n[INFO] Performing deep analysis on most suspicious author: {bottom_author}")
                
                # Analyze this author in more detail
                analyze_author_network(
                    rolap_conn, 
                    impact_conn, 
                    bottom_author, 
                    "deep_analysis",
                    schema
                )
                
                # Add detailed temporal analysis for this author
                try:
                    # Additional detailed temporal analysis for most suspicious author
                    print(f"[INFO] Performing detailed temporal analysis for suspicious author: {bottom_author}")
                    
                    # Build more expanded network for this author
                    expanded_graph = build_citation_network(
                        bottom_author, 
                        DEFAULT_DEPTH + 1,  # Deeper exploration
                        MAX_BFS_NODES * 2,  # More nodes
                        orcid_to_works, 
                        work_to_authors, 
                        work_to_cited,
                        citation_years
                    )
                    
                    # Run temporal analysis
                    temporal_metrics = analyze_temporal_citation_patterns(
                        expanded_graph,
                        work_to_authors,
                        publication_years,
                        citation_years
                    )
                    
                    # More detailed visualizations
                    visualize_temporal_patterns(
                        publication_years,
                        citation_years,
                        bottom_author,
                        f"{OUTPUT_DIR}/deep_analysis_temporal.png"
                    )
                    
                    # Burst analysis
                    burst_metrics = analyze_citation_bursts(
                        publication_years,
                        citation_years,
                        bottom_author,
                        f"{OUTPUT_DIR}/deep_analysis_bursts.png"
                    )
                    
                    # Save the detailed findings
                    with open(f"{OUTPUT_DIR}/deep_analysis_temporal.txt", 'w') as f:
                        f.write(f"DETAILED TEMPORAL ANALYSIS FOR {bottom_author}\n")
                        f.write("===========================================\n\n")
                        
                        f.write("Temporal Citation Patterns:\n")
                        for key, value in temporal_metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {key}: {value:.2f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        
                        f.write("\nCitation Burst Analysis:\n")
                        for key, value in burst_metrics.items():
                            if key == 'burst_years':
                                f.write(f"  Burst Years: {', '.join(str(y) for y in value)}\n")
                            elif key == 'burst_intensities':
                                f.write(f"  Burst Intensities: {', '.join(f'{i:.1f}σ' for i in value)}\n")
                            elif isinstance(value, (int, float)):
                                f.write(f"  {key}: {value:.2f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        
                        # Add interpretation
                        f.write("\nInterpretation:\n")
                        if burst_metrics.get('suspicious', False):
                            f.write("  This author shows suspicious citation burst patterns that may indicate\n")
                            f.write("  coordinated citation behavior or strategic citation manipulation.\n")
                        else:
                            f.write("  This author's citation patterns appear normal from a temporal perspective.\n")
                        
                        if temporal_metrics.get('self_vs_other_diff', 0) < -1:
                            f.write("\n  The author shows a pattern of citing their own work more quickly than\n")
                            f.write("  the work of others, which could indicate strategic self-citation.\n")
                except Exception as e:
                    print(f"[WARNING] Could not perform detailed temporal analysis: {e}")
                    import traceback
                    traceback.print_exc()
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Could not select most suspicious author: {e}")
        
        # Close database connections
        rolap_conn.close()
        impact_conn.close()
        
        print("\n[INFO] Analysis complete! Results are available in the output directory.")
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()