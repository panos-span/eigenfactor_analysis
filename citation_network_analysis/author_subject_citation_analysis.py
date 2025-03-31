#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Citation Network Analysis for Detecting Citation Cliques

This script analyzes academic citation networks to identify potentially
problematic citation patterns among authors, with a focus on detecting
cliques and citation rings among low-eigenfactor ("bottom") authors.

The script now supports subject-based analysis, allowing for comparison of 
citation patterns across different academic fields.

The script works with two SQLite databases:
- ROLAP_DB: Contains author and publication data
- IMPACT_DB: Contains citation data

Author: Claude
Date: March 2025
"""

import sqlite3
import networkx as nx
import csv
import traceback
import statistics
import itertools
import math
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
DEFAULT_DEPTH = 6  # Depth of BFS traversal
SUSPICIOUS_CLIQUE_MIN_SIZE = 3  # Minimum size of suspicious cliques
SUSPICIOUS_CLIQUE_MIN_DENSITY = 0.7  # Minimum density for suspicious cliques
SUSPICIOUS_CLIQUE_MIN_BAD_FRACTION = 0.7  # Minimum fraction of bad authors
CITATION_RING_MIN_SIZE = 3  # Minimum size of citation rings
CITATION_RING_MAX_SIZE = 12  # Maximum size of citation rings
CITATION_RING_MIN_BAD_FRACTION = 0.6  # Minimum fraction of bad authors in ring

# Output directory
OUTPUT_DIR = Path("output_subject_analysis")


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
                
                # Check for subject-related tables
                if table == 'matched_authors':
                    schema['rolap']['has_matched_authors_with_subjects'] = True
                
                if table == 'top_bottom_authors':
                    schema['rolap']['has_top_bottom_authors'] = True
                
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
    
    if not schema['rolap'].get('has_matched_authors_with_subjects', False):
        print("[WARNING] matched_authors table with subject information not found. Subject-based analysis will be limited.")
    
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


def load_publication_years(rolap_conn, schema):
    """
    Load publication years for works.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        
    Returns:
        dict: Dictionary mapping work IDs to publication years
    """
    publication_years = {}
    
    if schema['rolap'].get('has_works', False):
        works_columns = schema['rolap'].get('works_columns', [])
        
        # Check if year column exists
        if 'year' in works_columns:
            try:
                cursor = rolap_conn.cursor()
                cursor.execute("SELECT id, year FROM works WHERE year IS NOT NULL")
                for work_id, year in cursor.fetchall():
                    try:
                        # Store the year as an integer
                        publication_years[work_id] = int(year)
                    except (ValueError, TypeError):
                        # Skip if year cannot be parsed
                        continue
                cursor.close()
                print(f"[INFO] Loaded {len(publication_years)} publication years.")
            except sqlite3.OperationalError as e:
                print(f"[WARNING] Could not load publication years: {e}")
        else:
            print("[WARNING] 'year' column not found in works table.")
    else:
        print("[WARNING] works table not found. No publication years loaded.")
    
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


def load_author_subjects(rolap_conn, schema):
    """
    Load subject areas for authors.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        
    Returns:
        dict: Dictionary mapping ORCIDs to subject areas
    """
    author_subjects = {}
    
    if schema['rolap'].get('has_matched_authors_with_subjects', False):
        try:
            cursor = rolap_conn.cursor()
            # Try to get subjects for bottom authors
            cursor.execute("""
                SELECT bottom_orcid, random_subject AS subject
                FROM matched_authors
                WHERE bottom_orcid IS NOT NULL AND random_subject IS NOT NULL
            """)
            for orcid, subject in cursor.fetchall():
                author_subjects[orcid] = subject
                
            # Get subjects for random authors
            cursor.execute("""
                SELECT random_orcid, random_subject AS subject
                FROM matched_authors
                WHERE random_orcid IS NOT NULL AND random_subject IS NOT NULL
            """)
            for orcid, subject in cursor.fetchall():
                author_subjects[orcid] = subject
                
            cursor.close()
            print(f"[INFO] Loaded subject areas for {len(author_subjects)} authors.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load author subjects: {e}")
    
    if schema['rolap'].get('has_top_bottom_authors', False) and not author_subjects:
        try:
            cursor = rolap_conn.cursor()
            cursor.execute("""
                SELECT orcid, subject
                FROM top_bottom_authors
                WHERE orcid IS NOT NULL AND subject IS NOT NULL
            """)
            for orcid, subject in cursor.fetchall():
                author_subjects[orcid] = subject
                
            cursor.close()
            print(f"[INFO] Loaded subject areas for {len(author_subjects)} authors from top_bottom_authors.")
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Could not load subjects from top_bottom_authors: {e}")
    
    if not author_subjects:
        print("[WARNING] No author-subject mappings found. Subject-based analysis will be limited.")
    
    return author_subjects


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


def load_author_pairs(rolap_conn, schema, num_pairs=5, by_subject=True):
    """
    Load pairs of bottom and random authors for comparison, optionally grouped by subject.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        schema: Database schema information
        num_pairs: Number of author pairs to load per subject (if by_subject=True),
                  or total number of pairs (if by_subject=False)
        by_subject: Whether to group pairs by subject
        
    Returns:
        list or dict: If by_subject=False, returns a list of (bottom_orcid, random_orcid, total_works, subject) tuples
                     If by_subject=True, returns a dict mapping subjects to lists of author pairs
    """
    if by_subject:
        # Group pairs by subject
        pairs_by_subject = {}
        
        if schema['rolap'].get('has_matched_authors', False):
            try:
                cursor = rolap_conn.cursor()
                
                # First, get all available subjects
                cursor.execute("""
                    SELECT DISTINCT random_subject
                    FROM matched_authors
                    WHERE random_subject IS NOT NULL
                """)
                
                subjects = [row[0] for row in cursor.fetchall()]
                
                # For each subject, get top pairs by total works
                for subject in subjects:
                    cursor.execute(f"""
                        SELECT ma.bottom_orcid, ma.random_orcid, 
                               (COALESCE(mac.bottom_n_works,0) + COALESCE(mac.random_n_works,0)) AS total_works,
                               ma.random_subject
                        FROM matched_authors ma
                        LEFT JOIN matched_authors_with_counts mac ON ma.bottom_orcid = mac.bottom_orcid 
                                                                 AND ma.random_orcid = mac.random_orcid
                        WHERE ma.random_subject = ?
                        ORDER BY total_works DESC
                        LIMIT {num_pairs}
                    """, (subject,))
                    
                    subject_pairs = cursor.fetchall()
                    if subject_pairs:
                        pairs_by_subject[subject] = subject_pairs
                        
                cursor.close()
                print(f"[INFO] Loaded pairs for {len(pairs_by_subject)} subjects from matched_authors.")
                
                # Report number of pairs per subject
                for subject, pairs in pairs_by_subject.items():
                    print(f"[INFO] Subject '{subject}': {len(pairs)} pairs")
                    
            except sqlite3.OperationalError as e:
                print(f"[WARNING] Could not load subject-based author pairs: {e}")
        
        # If no pairs were loaded, use fallback logic
        if not pairs_by_subject:
            print("[INFO] Using fallback pairs grouped by synthetic subjects.")
            
            # Create synthetic subject groupings with our fallback pairs
            fallback_pairs = [
                ("0000-0003-0094-1778", "0000-0001-5204-3465", 100, "Computer Science"),
                ("0000-0001-6645-8645", "0000-0001-5236-4592", 90, "Computer Science"),
                ("0000-0003-0094-1778", "0000-0002-8656-1444", 80, "Physics"),
                ("0000-0003-0094-1778", "0000-0001-9215-9737", 70, "Physics"),
                ("0000-0001-9872-8742", "0000-0002-1871-1850", 60, "Biology")
            ]
            
            # Group by subject
            for pair in fallback_pairs:
                subject = pair[3]
                if subject not in pairs_by_subject:
                    pairs_by_subject[subject] = []
                pairs_by_subject[subject].append(pair)
        
        return pairs_by_subject
        
    else:
        # Original behavior: flat list of pairs
        pairs = []
        
        if schema['rolap'].get('has_matched_authors', False):
            try:
                cursor = rolap_conn.cursor()
                sql_select = f"""
                    SELECT ma.bottom_orcid, ma.random_orcid,
                           (COALESCE(mac.bottom_n_works,0) + COALESCE(mac.random_n_works,0)) AS total_works,
                           ma.random_subject
                    FROM matched_authors ma
                    LEFT JOIN matched_authors_with_counts mac ON ma.bottom_orcid = mac.bottom_orcid 
                                                             AND ma.random_orcid = mac.random_orcid
                    ORDER BY total_works DESC
                    LIMIT {num_pairs}
                """
                cursor.execute(sql_select)
                pairs = cursor.fetchall()
                cursor.close()
                print(f"[INFO] Loaded {len(pairs)} author pairs from matched_authors.")
            except sqlite3.OperationalError as e:
                print(f"[WARNING] Could not load author pairs: {e}")
        
        # If no pairs were loaded, use fallback pairs from the CSV
        if not pairs:
            print("[INFO] Using fallback pairs from the CSV.")
            pairs = [
                ("0000-0003-0094-1778", "0000-0001-5204-3465", 100, "Computer Science"),
                ("0000-0001-6645-8645", "0000-0001-5236-4592", 90, "Computer Science"),
                ("0000-0003-0094-1778", "0000-0002-8656-1444", 80, "Physics"),
                ("0000-0003-0094-1778", "0000-0001-9215-9737", 70, "Physics"),
                ("0000-0001-9872-8742", "0000-0002-1871-1850", 60, "Biology")
            ]
            
            # Trim to requested number of pairs
            pairs = pairs[:num_pairs]
        
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
    max_cycles=100_000,  # Reduced from 5,000,000 to 100,000
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
def analyze_author_citation_behavior(graph, bad_authors, author_metrics=None, author_subjects=None):
    """
    Analyze individual author citation behavior to detect anomalies.
    
    This function calculates various metrics for each author in the network,
    such as citation patterns, connections to bad authors, and reciprocity.
    
    Args:
        graph: NetworkX directed graph of the citation network
        bad_authors: Set of author IDs considered 'bad'
        author_metrics: Dictionary of external author metrics (optional)
        author_subjects: Dictionary mapping authors to subjects (optional)
        
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
        
        # Add subject information if available
        if author_subjects and author in author_subjects:
            author_analysis[author]['subject'] = author_subjects[author]
        
        # Add external metrics if available
        if author_metrics and author in author_metrics:
            for key, value in author_metrics[author].items():
                author_analysis[author][key] = value
    
    print(f"[INFO] Analyzed {len(author_analysis)} authors in the network.")
    return author_analysis


##############################################################################
# 7) Visualization Functions
##############################################################################
def visualize_suspicious_clique(graph, clique_data, bad_authors, output_path, subject=None):
    """
    Create a detailed visualization of a suspicious clique.
    
    Args:
        graph: NetworkX directed graph of the citation network
        clique_data: Dictionary with clique metrics and node list
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
        subject: Subject area of the clique (optional)
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
    title_parts = [
        f"Suspicious Clique (Size: {clique_data['size']}",
        f"Bad Authors: {clique_data['bad_count']}/{clique_data['size']}",
        f"Density: {clique_data['density']:.2f})"
    ]
    
    if subject:
        title_parts.insert(0, f"Subject: {subject}")
    
    title = ", ".join(title_parts)
    
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


def visualize_citation_ring(graph, ring_data, bad_authors, output_path, subject=None):
    """
    Create a visualization of a suspicious citation ring.
    Enhanced to better display larger rings.
    
    Args:
        graph: NetworkX directed graph of the citation network
        ring_data: Dictionary with ring metrics and node list
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
        subject: Subject area of the ring (optional)
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
    title_parts = [
        f"Citation Ring (Size: {ring_data['size']}",
        f"Bad Authors: {ring_data['bad_count']}/{ring_data['size']}",
        f"Avg. Citation Frequency: {ring_data['avg_edge_weight']:.1f})"
    ]
    
    if subject:
        title_parts.insert(0, f"Subject: {subject}")
    
    title = ", ".join(title_parts)
    
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
                           max_nodes=150, highlight_top=20, subject=None):
    """
    Visualize the author network highlighting suspicious authors.
    
    Args:
        graph: NetworkX directed graph of the citation network
        author_analysis: Dictionary of author analysis results
        bad_authors: Set of author IDs considered 'bad'
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to include
        highlight_top: Number of top suspicious authors to highlight
        subject: Subject area of the network (optional)
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
    title = "Citation Network with Suspicious Authors Highlighted"
    if subject:
        title = f"{title} (Subject: {subject})"
    plt.title(title, fontsize=16)
    
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
def save_clique_metrics(cliques, output_path, subject=None):
    """
    Save clique metrics to a CSV file.
    
    Args:
        cliques: List of clique dictionaries
        output_path: Path to save the CSV file
        subject: Subject area for these cliques (optional)
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
            header = ['clique_id']
            
            # Add subject if provided
            if subject:
                header.append('subject')
                
            header.extend(sorted_fields + ['nodes_sample'])
            
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
                
                # Add subject if provided
                if subject:
                    row.append(subject)
                
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


def save_ring_metrics(rings, output_path, subject=None):
    """
    Save ring metrics to a CSV file with additional summary for larger rings.
    
    Args:
        rings: List of ring dictionaries
        output_path: Path to save the CSV file
        subject: Subject area for these rings (optional)
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
            header = ['ring_id']
            
            # Add subject if provided
            if subject:
                header.append('subject')
                
            header.extend(sorted_fields + ['nodes_sample'])
            
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
                
                # Add subject if provided
                if subject:
                    row.append(subject)
                
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
            f.write(f"Ring Analysis Summary{' for ' + subject if subject else ''}\n")
            f.write("=" * 50 + "\n\n")
            
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
            
            # Additional temporal analysis if available
            rings_with_years = [r for r in rings if 'year_range' in r]
            if rings_with_years:
                avg_year_range = statistics.mean([r['year_range'] for r in rings_with_years])
                f.write(f"\nTemporal Analysis:\n")
                f.write(f"  Average year range: {avg_year_range:.2f} years\n")
                
                # List rings with very tight year ranges (potential coordinated behavior)
                tight_rings = [r for r in rings_with_years if r['year_range'] <= 2]
                if tight_rings:
                    f.write(f"  Rings with tight temporal coordination (≤ 2 years): {len(tight_rings)}\n")
        
        print(f"[INFO] Saved ring metrics to {output_path}")
        print(f"[INFO] Saved ring size summary to {Path(output_path).parent}/{Path(output_path).stem}_summary.txt")
    except Exception as e:
        print(f"[WARNING] Could not save ring metrics: {e}")


def save_author_metrics(author_analysis, output_path, subject=None):
    """
    Save author metrics to a CSV file.
    
    Args:
        author_analysis: Dictionary of author analysis results
        output_path: Path to save the CSV file
        subject: Subject area for these authors (optional)
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
            header = ['author_id']
            
            # Add explicit subject column if provided
            if subject and 'subject' not in sorted_fields:
                header.append('subject')
                
            header.extend(sorted_fields)
            
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
                
                # Add explicit subject if provided and not in metrics
                if subject and 'subject' not in sorted_fields:
                    row.append(subject)
                
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
                'subject',  # Added subject
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
                    result.get("subject", "Unknown"),  # Added subject with fallback
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
# 9) Subject-Based Analysis
##############################################################################
def analyze_subject_patterns(results_by_subject, output_dir):
    """
    Analyze patterns across different subjects and generate summary reports.
    
    Args:
        results_by_subject: Dictionary mapping subjects to lists of analysis results
        output_dir: Directory to save output files
    """
    print("[INFO] Analyzing patterns across subjects...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data structures for analysis
    subject_summaries = {}
    
    for subject, results in results_by_subject.items():
        # Skip subjects with no results
        if not results:
            continue
            
        # Calculate averages for bottom and random authors
        bottom_metrics = {
            "nodes": [],
            "edges": [],
            "density": [],
            "bad_node_fraction": [],
            "bad_edge_fraction": [],
            "suspicious_cliques": [],
            "citation_rings": [],
            "top_clique_size": [],
            "top_clique_score": [],
            "top_ring_size": [],
            "top_ring_score": []
        }
        
        random_metrics = {
            "nodes": [],
            "edges": [],
            "density": [],
            "bad_node_fraction": [],
            "bad_edge_fraction": [],
            "suspicious_cliques": [],
            "citation_rings": [],
            "top_clique_size": [],
            "top_clique_score": [],
            "top_ring_size": [],
            "top_ring_score": []
        }
        
        # Collect metrics from all pairs in this subject
        for result in results:
            for metric in bottom_metrics:
                bottom_metrics[metric].append(result["bottom_metrics"][metric])
                random_metrics[metric].append(result["random_metrics"][metric])
        
        # Calculate summary statistics
        bottom_summary = {}
        random_summary = {}
        
        for metric in bottom_metrics:
            # Average
            bottom_summary[f"avg_{metric}"] = statistics.mean(bottom_metrics[metric])
            random_summary[f"avg_{metric}"] = statistics.mean(random_metrics[metric])
            
            # Standard deviation (if more than one value)
            if len(bottom_metrics[metric]) > 1:
                bottom_summary[f"std_{metric}"] = statistics.stdev(bottom_metrics[metric])
                random_summary[f"std_{metric}"] = statistics.stdev(random_metrics[metric])
            else:
                bottom_summary[f"std_{metric}"] = 0
                random_summary[f"std_{metric}"] = 0
            
            # Median
            bottom_summary[f"median_{metric}"] = statistics.median(bottom_metrics[metric])
            random_summary[f"median_{metric}"] = statistics.median(random_metrics[metric])
        
        # Calculate the average difference between bottom and random authors
        diff_summary = {}
        for metric in bottom_metrics:
            diffs = [b - r for b, r in zip(bottom_metrics[metric], random_metrics[metric])]
            diff_summary[f"avg_diff_{metric}"] = statistics.mean(diffs)
            
            # Calculate ratio of bottom to random (for non-zero values)
            ratios = []
            for b, r in zip(bottom_metrics[metric], random_metrics[metric]):
                if r != 0:
                    ratios.append(b / r)
            
            if ratios:
                diff_summary[f"avg_ratio_{metric}"] = statistics.mean(ratios)
            else:
                diff_summary[f"avg_ratio_{metric}"] = float('nan')
        
        # Store summaries for this subject
        subject_summaries[subject] = {
            "num_pairs": len(results),
            "bottom": bottom_summary,
            "random": random_summary,
            "diff": diff_summary
        }
    
    # Save subject summaries to CSV
    with open(f"{output_dir}/subject_summaries.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Define header
        header = ['subject', 'num_pairs']
        
        # Add all metrics for bottom, random, and diff
        metric_types = ['avg', 'median', 'std']
        all_metrics = [
            "nodes", "edges", "density", 
            "bad_node_fraction", "bad_edge_fraction",
            "suspicious_cliques", "citation_rings",
            "top_clique_size", "top_clique_score",
            "top_ring_size", "top_ring_score"
        ]
        
        for author_type in ['bottom', 'random']:
            for metric_type in metric_types:
                for metric in all_metrics:
                    header.append(f"{author_type}_{metric_type}_{metric}")
        
        # Add difference metrics
        for metric in all_metrics:
            header.append(f"diff_avg_{metric}")
            header.append(f"diff_ratio_{metric}")
            
        writer.writerow(header)
        
        # Write rows for each subject
        for subject, summary in subject_summaries.items():
            row = [subject, summary["num_pairs"]]
            
            # Add metrics for bottom and random
            for author_type in ['bottom', 'random']:
                for metric_type in metric_types:
                    for metric in all_metrics:
                        key = f"{metric_type}_{metric}"
                        value = summary[author_type].get(key, 0)
                        if isinstance(value, float):
                            value = f"{value:.4f}"
                        row.append(value)
            
            # Add difference metrics
            for metric in all_metrics:
                diff_key = f"avg_diff_{metric}"
                ratio_key = f"avg_ratio_{metric}"
                
                diff_value = summary["diff"].get(diff_key, 0)
                if isinstance(diff_value, float):
                    diff_value = f"{diff_value:.4f}"
                row.append(diff_value)
                
                ratio_value = summary["diff"].get(ratio_key, 0)
                if isinstance(ratio_value, float) and not math.isnan(ratio_value):
                    ratio_value = f"{ratio_value:.4f}"
                else:
                    ratio_value = "N/A"
                row.append(ratio_value)
                
            writer.writerow(row)
    
    print(f"[INFO] Saved subject summaries to {output_dir}/subject_summaries.csv")
    
    # Create visualizations comparing subjects
    # Bar chart of clique and ring counts by subject
    plt.figure(figsize=(14, 8))
    
    subjects = list(subject_summaries.keys())
    x = np.arange(len(subjects))
    width = 0.35
    
    bottom_cliques = [subject_summaries[s]["bottom"]["avg_suspicious_cliques"] for s in subjects]
    random_cliques = [subject_summaries[s]["random"]["avg_suspicious_cliques"] for s in subjects]
    
    plt.bar(x - width/2, bottom_cliques, width, label='Bottom Authors')
    plt.bar(x + width/2, random_cliques, width, label='Random Authors')
    
    plt.xlabel('Subject')
    plt.ylabel('Average Number of Suspicious Cliques')
    plt.title('Comparison of Suspicious Cliques by Subject')
    plt.xticks(x, subjects, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/subject_clique_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart of bad node fractions by subject
    plt.figure(figsize=(14, 8))
    
    bottom_bad = [subject_summaries[s]["bottom"]["avg_bad_node_fraction"] for s in subjects]
    random_bad = [subject_summaries[s]["random"]["avg_bad_node_fraction"] for s in subjects]
    
    plt.bar(x - width/2, bottom_bad, width, label='Bottom Authors')
    plt.bar(x + width/2, random_bad, width, label='Random Authors')
    
    plt.xlabel('Subject')
    plt.ylabel('Average Bad Node Fraction')
    plt.title('Comparison of Bad Node Fraction by Subject')
    plt.xticks(x, subjects, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/subject_bad_node_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a ranked list of subjects by citation manipulation metrics
    manipulation_scores = {}
    for subject, summary in subject_summaries.items():
        # Compute a composite score based on multiple metrics
        clique_ratio = summary["diff"].get("avg_ratio_suspicious_cliques", 1)
        ring_ratio = summary["diff"].get("avg_ratio_citation_rings", 1)
        bad_node_diff = summary["diff"].get("avg_diff_bad_node_fraction", 0)
        
        # Handle NaN values
        if isinstance(clique_ratio, float) and math.isnan(clique_ratio):
            clique_ratio = 1
        if isinstance(ring_ratio, float) and math.isnan(ring_ratio):
            ring_ratio = 1
            
        # Compute weighted score
        manipulation_score = (
            0.4 * clique_ratio + 
            0.4 * ring_ratio + 
            0.2 * (bad_node_diff * 10)  # Scale up the bad node fraction difference
        )
        
        manipulation_scores[subject] = manipulation_score
    
    # Create a bar chart of manipulation scores
    plt.figure(figsize=(14, 8))
    
    # Sort subjects by manipulation score
    sorted_subjects = sorted(manipulation_scores.items(), key=lambda x: x[1], reverse=True)
    subjects = [s[0] for s in sorted_subjects]
    scores = [s[1] for s in sorted_subjects]
    
    plt.bar(subjects, scores, color='darkred')
    plt.xlabel('Subject')
    plt.ylabel('Citation Manipulation Score')
    plt.title('Subjects Ranked by Citation Manipulation Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/subject_manipulation_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the ranking to a text file
    with open(f"{output_dir}/subject_manipulation_ranking.txt", 'w') as f:
        f.write("Subjects Ranked by Citation Manipulation Score\n")
        f.write("===========================================\n\n")
        
        for i, (subject, score) in enumerate(sorted_subjects, 1):
            f.write(f"{i}. {subject}: {score:.4f}\n")
            
            # Add key metrics
            summary = subject_summaries[subject]
            f.write(f"   - Bottom author cliques: {summary['bottom']['avg_suspicious_cliques']:.1f} vs Random: {summary['random']['avg_suspicious_cliques']:.1f}\n")
            f.write(f"   - Bottom author rings: {summary['bottom']['avg_citation_rings']:.1f} vs Random: {summary['random']['avg_citation_rings']:.1f}\n")
            f.write(f"   - Bad node fraction diff: {summary['diff']['avg_diff_bad_node_fraction']:.4f}\n")
            f.write("\n")
    
    print(f"[INFO] Created subject comparison visualizations and rankings")


##############################################################################
# 10) Single Author Analysis
##############################################################################
def analyze_author_network(rolap_conn, impact_conn, author_id, output_prefix, schema, author_subjects=None):
    """
    Perform comprehensive analysis on a single author's citation network.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        author_id: The author's ORCID to analyze
        output_prefix: Prefix for output files
        schema: Database schema information
        author_subjects: Dictionary mapping authors to subjects (optional)
        
    Returns:
        tuple: (graph, cliques, rings, author_analysis)
    """
    subject = author_subjects.get(author_id, "Unknown") if author_subjects else None
    subject_str = f" (Subject: {subject})" if subject else ""
    
    print(f"[INFO] Starting comprehensive analysis for author {author_id}{subject_str}")
    
    # Step 1: Load data
    bad_authors = load_bad_authors(rolap_conn, schema)
    #publication_years = load_publication_years(rolap_conn, schema)
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
            if subject:
                f.write(f"Subject: {subject}\n")
            f.write(f"Number of nodes: {graph.number_of_nodes()}\n")
            f.write(f"Number of edges: {graph.number_of_edges()}\n")
            if graph.number_of_nodes() > 0:
                f.write(f"Density: {nx.density(graph):.6f}\n")
                f.write(f"Bad authors in network: {sum(1 for n in graph.nodes() if n in bad_authors)}\n")
                f.write(f"Fraction of bad authors: {sum(1 for n in graph.nodes() if n in bad_authors) / graph.number_of_nodes():.4f}\n")
    except Exception as e:
        print(f"[WARNING] Could not save graph metrics: {e}")
    
    # Step 3: Find suspicious cliques
    suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
    
    # Step 4: Find citation rings
    citation_rings = find_citation_rings(graph, bad_authors)
    
    # Step 5: Analyze author citation behavior
    author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_metrics, author_subjects)
    
    # Step 6: Generate visualizations
    if suspicious_cliques:
        for i, clique in enumerate(suspicious_cliques[:5]):
            output_path = f"{OUTPUT_DIR}/{output_prefix}_suspicious_clique_{i+1}.png"
            visualize_suspicious_clique(graph, clique, bad_authors, output_path, subject)
    
    if citation_rings:
        for i, ring in enumerate(citation_rings[:5]):
            output_path = f"{OUTPUT_DIR}/{output_prefix}_citation_ring_{i+1}.png"
            visualize_citation_ring(graph, ring, bad_authors, output_path, subject)
    
    if author_analysis:
        visualize_author_network(
            graph, 
            author_analysis, 
            bad_authors, 
            f"{OUTPUT_DIR}/{output_prefix}_author_network.png",
            subject=subject
        )
    
    # Step 7: Save metrics to CSV
    save_clique_metrics(suspicious_cliques, f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv", subject)
    save_ring_metrics(citation_rings, f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv", subject)
    save_author_metrics(author_analysis, f"{OUTPUT_DIR}/{output_prefix}_author_analysis.csv", subject)
    
    print(f"[INFO] Analysis complete for author {author_id}")
    return graph, suspicious_cliques, citation_rings, author_analysis

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


def visualize_temporal_patterns(publication_years, citation_years, author_id, output_path, subject=None):
    """
    Create visualizations for temporal citation patterns.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Path to save the visualization
        subject: Subject area of the author (optional)
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
    title = f'Temporal Analysis for Author: {author_id}'
    if subject:
        title += f' (Subject: {subject})'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved temporal analysis visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


def analyze_citation_bursts(publication_years, citation_years, author_id, output_path=None, subject=None):
    """
    Analyze citation pattern bursts to detect potentially suspicious activity.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Optional path to save visualization
        subject: Subject area of the author (optional)
        
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
        
        # Add title
        title = f'Citation Pattern Analysis for Author: {author_id}'
        if subject:
            title += f' (Subject: {subject})'
        plt.title(title)
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


def analyze_subject_temporal_patterns(results_by_subject, output_dir, publication_years, citation_years):
    """
    Analyze temporal patterns across subjects and generate comparative reports.
    
    Args:
        results_by_subject: Dictionary mapping subjects to lists of analysis results
        output_dir: Directory to save output files
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import csv
    
    # Prepare data structures
    subject_temporal_metrics = {}
    
    # Create a directory for temporal analysis by subject
    temporal_dir = Path(f"{output_dir}/subject_temporal_analysis")
    if not temporal_dir.exists():
        temporal_dir.mkdir(parents=True)
        print(f"[INFO] Created directory for subject temporal analysis: {temporal_dir}")
    
    # Process each subject
    for subject, results in results_by_subject.items():
        subject_metrics = {
            'bottom': {
                'citation_age': [],
                'self_citation_age': [],
                'other_citation_age': [],
                'burst_counts': [],
                'max_burst_intensities': []
            },
            'random': {
                'citation_age': [],
                'self_citation_age': [],
                'other_citation_age': [],
                'burst_counts': [],
                'max_burst_intensities': []
            }
        }
        
        # Process each pair in the subject
        for result in results:
            bottom_author = result["bottom_author"]
            random_author = result["random_author"]
            
            # Process bottom author
            try:
                # Build citation ages for bottom author
                bottom_citation_ages = []
                for (source_id, target_id), cite_year in citation_years.items():
                    if source_id in publication_years and target_id in publication_years:
                        source_year = publication_years[source_id]
                        target_year = publication_years[target_id]
                        citation_age = source_year - target_year
                        bottom_citation_ages.append(citation_age)
                
                # Analyze citation bursts for bottom author
                bottom_burst_analysis = analyze_citation_bursts(
                    publication_years,
                    citation_years,
                    bottom_author,
                    f"{temporal_dir}/{subject}_BOTTOM_{bottom_author}_bursts.png",
                    subject=subject
                )
                
                # Store metrics for bottom author
                if bottom_citation_ages:
                    subject_metrics['bottom']['citation_age'].extend(bottom_citation_ages)
                
                if bottom_burst_analysis.get('burst_count', 0) > 0:
                    subject_metrics['bottom']['burst_counts'].append(bottom_burst_analysis['burst_count'])
                    subject_metrics['bottom']['max_burst_intensities'].append(bottom_burst_analysis['max_burst_intensity'])
                
            except Exception as e:
                print(f"[WARNING] Error processing temporal metrics for bottom author {bottom_author}: {e}")
            
            # Process random author
            try:
                # Build citation ages for random author
                random_citation_ages = []
                for (source_id, target_id), cite_year in citation_years.items():
                    if source_id in publication_years and target_id in publication_years:
                        source_year = publication_years[source_id]
                        target_year = publication_years[target_id]
                        citation_age = source_year - target_year
                        random_citation_ages.append(citation_age)
                
                # Analyze citation bursts for random author
                random_burst_analysis = analyze_citation_bursts(
                    publication_years,
                    citation_years,
                    random_author,
                    f"{temporal_dir}/{subject}_RANDOM_{random_author}_bursts.png",
                    subject=subject
                )
                
                # Store metrics for random author
                if random_citation_ages:
                    subject_metrics['random']['citation_age'].extend(random_citation_ages)
                
                if random_burst_analysis.get('burst_count', 0) > 0:
                    subject_metrics['random']['burst_counts'].append(random_burst_analysis['burst_count'])
                    subject_metrics['random']['max_burst_intensities'].append(random_burst_analysis['max_burst_intensity'])
                
            except Exception as e:
                print(f"[WARNING] Error processing temporal metrics for random author {random_author}: {e}")
        
        # Calculate aggregate metrics for this subject
        subject_summary = {}
        
        for author_type in ['bottom', 'random']:
            summary = {}
            
            # Citation age metrics
            if subject_metrics[author_type]['citation_age']:
                summary['median_citation_age'] = np.median(subject_metrics[author_type]['citation_age'])
                summary['mean_citation_age'] = np.mean(subject_metrics[author_type]['citation_age'])
                summary['recent_citation_pct'] = sum(1 for age in subject_metrics[author_type]['citation_age'] if age <= 2) / len(subject_metrics[author_type]['citation_age']) * 100
            
            # Burst metrics
            if subject_metrics[author_type]['burst_counts']:
                summary['avg_burst_count'] = np.mean(subject_metrics[author_type]['burst_counts'])
                summary['max_burst_intensity'] = np.mean(subject_metrics[author_type]['max_burst_intensities'])
                summary['suspicious_burst_pct'] = sum(1 for count in subject_metrics[author_type]['burst_counts'] if count >= 2) / len(subject_metrics[author_type]['burst_counts']) * 100
            
            subject_summary[author_type] = summary
        
        # Calculate differences between bottom and random
        diff_summary = {}
        
        # Citation age difference
        if 'median_citation_age' in subject_summary['bottom'] and 'median_citation_age' in subject_summary['random']:
            diff_summary['median_citation_age_diff'] = subject_summary['bottom']['median_citation_age'] - subject_summary['random']['median_citation_age']
        
        # Burst metrics differences
        if 'avg_burst_count' in subject_summary['bottom'] and 'avg_burst_count' in subject_summary['random']:
            diff_summary['avg_burst_count_diff'] = subject_summary['bottom']['avg_burst_count'] - subject_summary['random']['avg_burst_count']
        
        if 'suspicious_burst_pct' in subject_summary['bottom'] and 'suspicious_burst_pct' in subject_summary['random']:
            diff_summary['suspicious_burst_pct_diff'] = subject_summary['bottom']['suspicious_burst_pct'] - subject_summary['random']['suspicious_burst_pct']
        
        # Store complete summary
        subject_summary['diff'] = diff_summary
        subject_temporal_metrics[subject] = subject_summary
    
    # Create visualizations comparing subjects
    # Bar chart of citation age by subject
    if subject_temporal_metrics:
        plt.figure(figsize=(14, 8))
        
        subjects = list(subject_temporal_metrics.keys())
        x = np.arange(len(subjects))
        width = 0.35
        
        # Extract median citation ages for bottom and random authors
        bottom_ages = []
        random_ages = []
        
        for subject in subjects:
            bottom_age = subject_temporal_metrics[subject]['bottom'].get('median_citation_age', 0)
            random_age = subject_temporal_metrics[subject]['random'].get('median_citation_age', 0)
            bottom_ages.append(bottom_age)
            random_ages.append(random_age)
        
        plt.bar(x - width/2, bottom_ages, width, label='Bottom Authors')
        plt.bar(x + width/2, random_ages, width, label='Random Authors')
        
        plt.xlabel('Subject')
        plt.ylabel('Median Citation Age (Years)')
        plt.title('Comparison of Citation Age by Subject')
        plt.xticks(x, subjects, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{temporal_dir}/subject_citation_age_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar chart of suspicious burst percentage by subject
        plt.figure(figsize=(14, 8))
        
        # Extract suspicious burst percentages for bottom and random authors
        bottom_burst_pct = []
        random_burst_pct = []
        
        for subject in subjects:
            bottom_pct = subject_temporal_metrics[subject]['bottom'].get('suspicious_burst_pct', 0)
            random_pct = subject_temporal_metrics[subject]['random'].get('suspicious_burst_pct', 0)
            bottom_burst_pct.append(bottom_pct)
            random_burst_pct.append(random_pct)
        
        plt.bar(x - width/2, bottom_burst_pct, width, label='Bottom Authors')
        plt.bar(x + width/2, random_burst_pct, width, label='Random Authors')
        
        plt.xlabel('Subject')
        plt.ylabel('Suspicious Burst Pattern (%)')
        plt.title('Comparison of Suspicious Burst Patterns by Subject')
        plt.xticks(x, subjects, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{temporal_dir}/subject_burst_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save temporal metrics to CSV
        with open(f"{temporal_dir}/subject_temporal_metrics.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Define header
            header = ['subject']
            
            # Add metrics for bottom and random
            for author_type in ['bottom', 'random']:
                for metric in ['median_citation_age', 'mean_citation_age', 'recent_citation_pct', 
                               'avg_burst_count', 'max_burst_intensity', 'suspicious_burst_pct']:
                    header.append(f"{author_type}_{metric}")
            
            # Add difference metrics
            for metric in ['median_citation_age_diff', 'avg_burst_count_diff', 'suspicious_burst_pct_diff']:
                header.append(f"diff_{metric}")
                
            writer.writerow(header)
            
            # Write rows for each subject
            for subject, summary in subject_temporal_metrics.items():
                row = [subject]
                
                # Add metrics for bottom and random
                for author_type in ['bottom', 'random']:
                    for metric in ['median_citation_age', 'mean_citation_age', 'recent_citation_pct', 
                                  'avg_burst_count', 'max_burst_intensity', 'suspicious_burst_pct']:
                        value = summary[author_type].get(metric, '')
                        if isinstance(value, float):
                            value = f"{value:.4f}"
                        row.append(value)
                
                # Add difference metrics
                for metric in ['median_citation_age_diff', 'avg_burst_count_diff', 'suspicious_burst_pct_diff']:
                    value = summary['diff'].get(metric, '')
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(value)
                    
                writer.writerow(row)
        
        # Create a summary report
        with open(f"{temporal_dir}/subject_temporal_summary.txt", 'w') as f:
            f.write("TEMPORAL CITATION ANALYSIS BY SUBJECT\n")
            f.write("====================================\n\n")
            
            # Calculate differences across subjects
            bottom_suspicious_pct = {}
            random_suspicious_pct = {}
            
            for subject, summary in subject_temporal_metrics.items():
                bottom_suspicious_pct[subject] = summary['bottom'].get('suspicious_burst_pct', 0)
                random_suspicious_pct[subject] = summary['random'].get('suspicious_burst_pct', 0)
            
            # Identify subjects with largest differences in suspicious patterns
            diff_pct = {subject: bottom_suspicious_pct[subject] - random_suspicious_pct[subject] 
                        for subject in subjects if subject in bottom_suspicious_pct and subject in random_suspicious_pct}
            
            sorted_diffs = sorted(diff_pct.items(), key=lambda x: x[1], reverse=True)
            
            f.write("Subjects Ranked by Difference in Suspicious Burst Patterns:\n")
            for subject, diff in sorted_diffs:
                bottom_pct = bottom_suspicious_pct[subject]
                random_pct = random_suspicious_pct[subject]
                f.write(f"  * {subject}: {diff:.1f}% difference (Bottom: {bottom_pct:.1f}%, Random: {random_pct:.1f}%)\n")
            
            f.write("\nSubjects Ranked by Overall Suspicious Burst Patterns in Bottom Authors:\n")
            for subject, pct in sorted(bottom_suspicious_pct.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  * {subject}: {pct:.1f}%\n")
            
            f.write("\nCitation Age Analysis by Subject:\n")
            for subject, summary in subject_temporal_metrics.items():
                if 'median_citation_age' in summary['bottom'] and 'median_citation_age' in summary['random']:
                    bottom_age = summary['bottom']['median_citation_age']
                    random_age = summary['random']['median_citation_age']
                    diff = summary['diff']['median_citation_age_diff']
                    
                    f.write(f"  * {subject}: Bottom authors cite {abs(diff):.1f} years {'older' if diff > 0 else 'newer'} work ")
                    f.write(f"than random authors (Bottom: {bottom_age:.1f} years, Random: {random_age:.1f} years)\n")
            
            f.write("\nSUSPICIOUS TEMPORAL PATTERNS BY SUBJECT:\n")
            for subject, summary in subject_temporal_metrics.items():
                # Define criteria for suspicious patterns
                if (summary['diff'].get('suspicious_burst_pct_diff', 0) > 20 or
                    summary['diff'].get('median_citation_age_diff', 0) < -2):
                    
                    f.write(f"\n{subject}:\n")
                    f.write(f"  - {'Higher' if summary['diff'].get('suspicious_burst_pct_diff', 0) > 0 else 'Lower'} rate of suspicious burst patterns: ")
                    f.write(f"{abs(summary['diff'].get('suspicious_burst_pct_diff', 0)):.1f}% difference\n")
                    
                    if 'median_citation_age_diff' in summary['diff']:
                        f.write(f"  - {'Cites older' if summary['diff']['median_citation_age_diff'] > 0 else 'Cites newer'} work: ")
                        f.write(f"{abs(summary['diff']['median_citation_age_diff']):.1f} years difference\n")
    
    print(f"[INFO] Completed temporal analysis by subject")
    return subject_temporal_metrics

# Function to integrate temporal analysis into subject batch analysis
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


def visualize_temporal_patterns(publication_years, citation_years, author_id, output_path, subject=None):
    """
    Create visualizations for temporal citation patterns.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Path to save the visualization
        subject: Subject area of the author (optional)
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
    title = f'Temporal Analysis for Author: {author_id}'
    if subject:
        title += f' (Subject: {subject})'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved temporal analysis visualization to {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save visualization: {e}")
        
    plt.close()


def analyze_citation_bursts(publication_years, citation_years, author_id, output_path=None, subject=None):
    """
    Analyze citation pattern bursts to detect potentially suspicious activity.
    
    Args:
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
        author_id: ORCID of the author being analyzed
        output_path: Optional path to save visualization
        subject: Subject area of the author (optional)
        
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
        
        # Add title
        title = f'Citation Pattern Analysis for Author: {author_id}'
        if subject:
            title += f' (Subject: {subject})'
        plt.title(title)
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


def analyze_subject_temporal_patterns(results_by_subject, output_dir, publication_years, citation_years):
    """
    Analyze temporal patterns across subjects and generate comparative reports.
    
    Args:
        results_by_subject: Dictionary mapping subjects to lists of analysis results
        output_dir: Directory to save output files
        publication_years: Dictionary mapping work IDs to publication years
        citation_years: Dictionary mapping (source_id, target_id) to citation years
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import csv
    
    # Prepare data structures
    subject_temporal_metrics = {}
    
    # Create a directory for temporal analysis by subject
    temporal_dir = Path(f"{output_dir}/subject_temporal_analysis")
    if not temporal_dir.exists():
        temporal_dir.mkdir(parents=True)
        print(f"[INFO] Created directory for subject temporal analysis: {temporal_dir}")
    
    # Process each subject
    for subject, results in results_by_subject.items():
        subject_metrics = {
            'bottom': {
                'citation_age': [],
                'self_citation_age': [],
                'other_citation_age': [],
                'burst_counts': [],
                'max_burst_intensities': []
            },
            'random': {
                'citation_age': [],
                'self_citation_age': [],
                'other_citation_age': [],
                'burst_counts': [],
                'max_burst_intensities': []
            }
        }
        
        # Process each pair in the subject
        for result in results:
            bottom_author = result["bottom_author"]
            random_author = result["random_author"]
            
            # Process bottom author
            try:
                # Build citation ages for bottom author
                bottom_citation_ages = []
                for (source_id, target_id), cite_year in citation_years.items():
                    if source_id in publication_years and target_id in publication_years:
                        source_year = publication_years[source_id]
                        target_year = publication_years[target_id]
                        citation_age = source_year - target_year
                        bottom_citation_ages.append(citation_age)
                
                # Analyze citation bursts for bottom author
                bottom_burst_analysis = analyze_citation_bursts(
                    publication_years,
                    citation_years,
                    bottom_author,
                    f"{temporal_dir}/{subject}_BOTTOM_{bottom_author}_bursts.png",
                    subject=subject
                )
                
                # Store metrics for bottom author
                if bottom_citation_ages:
                    subject_metrics['bottom']['citation_age'].extend(bottom_citation_ages)
                
                if bottom_burst_analysis.get('burst_count', 0) > 0:
                    subject_metrics['bottom']['burst_counts'].append(bottom_burst_analysis['burst_count'])
                    subject_metrics['bottom']['max_burst_intensities'].append(bottom_burst_analysis['max_burst_intensity'])
                
            except Exception as e:
                print(f"[WARNING] Error processing temporal metrics for bottom author {bottom_author}: {e}")
            
            # Process random author
            try:
                # Build citation ages for random author
                random_citation_ages = []
                for (source_id, target_id), cite_year in citation_years.items():
                    if source_id in publication_years and target_id in publication_years:
                        source_year = publication_years[source_id]
                        target_year = publication_years[target_id]
                        citation_age = source_year - target_year
                        random_citation_ages.append(citation_age)
                
                # Analyze citation bursts for random author
                random_burst_analysis = analyze_citation_bursts(
                    publication_years,
                    citation_years,
                    random_author,
                    f"{temporal_dir}/{subject}_RANDOM_{random_author}_bursts.png",
                    subject=subject
                )
                
                # Store metrics for random author
                if random_citation_ages:
                    subject_metrics['random']['citation_age'].extend(random_citation_ages)
                
                if random_burst_analysis.get('burst_count', 0) > 0:
                    subject_metrics['random']['burst_counts'].append(random_burst_analysis['burst_count'])
                    subject_metrics['random']['max_burst_intensities'].append(random_burst_analysis['max_burst_intensity'])
                
            except Exception as e:
                print(f"[WARNING] Error processing temporal metrics for random author {random_author}: {e}")
        
        # Calculate aggregate metrics for this subject
        subject_summary = {}
        
        for author_type in ['bottom', 'random']:
            summary = {}
            
            # Citation age metrics
            if subject_metrics[author_type]['citation_age']:
                summary['median_citation_age'] = np.median(subject_metrics[author_type]['citation_age'])
                summary['mean_citation_age'] = np.mean(subject_metrics[author_type]['citation_age'])
                summary['recent_citation_pct'] = sum(1 for age in subject_metrics[author_type]['citation_age'] if age <= 2) / len(subject_metrics[author_type]['citation_age']) * 100
            
            # Burst metrics
            if subject_metrics[author_type]['burst_counts']:
                summary['avg_burst_count'] = np.mean(subject_metrics[author_type]['burst_counts'])
                summary['max_burst_intensity'] = np.mean(subject_metrics[author_type]['max_burst_intensities'])
                summary['suspicious_burst_pct'] = sum(1 for count in subject_metrics[author_type]['burst_counts'] if count >= 2) / len(subject_metrics[author_type]['burst_counts']) * 100
            
            subject_summary[author_type] = summary
        
        # Calculate differences between bottom and random
        diff_summary = {}
        
        # Citation age difference
        if 'median_citation_age' in subject_summary['bottom'] and 'median_citation_age' in subject_summary['random']:
            diff_summary['median_citation_age_diff'] = subject_summary['bottom']['median_citation_age'] - subject_summary['random']['median_citation_age']
        
        # Burst metrics differences
        if 'avg_burst_count' in subject_summary['bottom'] and 'avg_burst_count' in subject_summary['random']:
            diff_summary['avg_burst_count_diff'] = subject_summary['bottom']['avg_burst_count'] - subject_summary['random']['avg_burst_count']
        
        if 'suspicious_burst_pct' in subject_summary['bottom'] and 'suspicious_burst_pct' in subject_summary['random']:
            diff_summary['suspicious_burst_pct_diff'] = subject_summary['bottom']['suspicious_burst_pct'] - subject_summary['random']['suspicious_burst_pct']
        
        # Store complete summary
        subject_summary['diff'] = diff_summary
        subject_temporal_metrics[subject] = subject_summary
    
    # Create visualizations comparing subjects
    # Bar chart of citation age by subject
    if subject_temporal_metrics:
        plt.figure(figsize=(14, 8))
        
        subjects = list(subject_temporal_metrics.keys())
        x = np.arange(len(subjects))
        width = 0.35
        
        # Extract median citation ages for bottom and random authors
        bottom_ages = []
        random_ages = []
        
        for subject in subjects:
            bottom_age = subject_temporal_metrics[subject]['bottom'].get('median_citation_age', 0)
            random_age = subject_temporal_metrics[subject]['random'].get('median_citation_age', 0)
            bottom_ages.append(bottom_age)
            random_ages.append(random_age)
        
        plt.bar(x - width/2, bottom_ages, width, label='Bottom Authors')
        plt.bar(x + width/2, random_ages, width, label='Random Authors')
        
        plt.xlabel('Subject')
        plt.ylabel('Median Citation Age (Years)')
        plt.title('Comparison of Citation Age by Subject')
        plt.xticks(x, subjects, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{temporal_dir}/subject_citation_age_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar chart of suspicious burst percentage by subject
        plt.figure(figsize=(14, 8))
        
        # Extract suspicious burst percentages for bottom and random authors
        bottom_burst_pct = []
        random_burst_pct = []
        
        for subject in subjects:
            bottom_pct = subject_temporal_metrics[subject]['bottom'].get('suspicious_burst_pct', 0)
            random_pct = subject_temporal_metrics[subject]['random'].get('suspicious_burst_pct', 0)
            bottom_burst_pct.append(bottom_pct)
            random_burst_pct.append(random_pct)
        
        plt.bar(x - width/2, bottom_burst_pct, width, label='Bottom Authors')
        plt.bar(x + width/2, random_burst_pct, width, label='Random Authors')
        
        plt.xlabel('Subject')
        plt.ylabel('Suspicious Burst Pattern (%)')
        plt.title('Comparison of Suspicious Burst Patterns by Subject')
        plt.xticks(x, subjects, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{temporal_dir}/subject_burst_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save temporal metrics to CSV
        with open(f"{temporal_dir}/subject_temporal_metrics.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Define header
            header = ['subject']
            
            # Add metrics for bottom and random
            for author_type in ['bottom', 'random']:
                for metric in ['median_citation_age', 'mean_citation_age', 'recent_citation_pct', 
                               'avg_burst_count', 'max_burst_intensity', 'suspicious_burst_pct']:
                    header.append(f"{author_type}_{metric}")
            
            # Add difference metrics
            for metric in ['median_citation_age_diff', 'avg_burst_count_diff', 'suspicious_burst_pct_diff']:
                header.append(f"diff_{metric}")
                
            writer.writerow(header)
            
            # Write rows for each subject
            for subject, summary in subject_temporal_metrics.items():
                row = [subject]
                
                # Add metrics for bottom and random
                for author_type in ['bottom', 'random']:
                    for metric in ['median_citation_age', 'mean_citation_age', 'recent_citation_pct', 
                                  'avg_burst_count', 'max_burst_intensity', 'suspicious_burst_pct']:
                        value = summary[author_type].get(metric, '')
                        if isinstance(value, float):
                            value = f"{value:.4f}"
                        row.append(value)
                
                # Add difference metrics
                for metric in ['median_citation_age_diff', 'avg_burst_count_diff', 'suspicious_burst_pct_diff']:
                    value = summary['diff'].get(metric, '')
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(value)
                    
                writer.writerow(row)
        
        # Create a summary report
        with open(f"{temporal_dir}/subject_temporal_summary.txt", 'w') as f:
            f.write("TEMPORAL CITATION ANALYSIS BY SUBJECT\n")
            f.write("====================================\n\n")
            
            # Calculate differences across subjects
            bottom_suspicious_pct = {}
            random_suspicious_pct = {}
            
            for subject, summary in subject_temporal_metrics.items():
                bottom_suspicious_pct[subject] = summary['bottom'].get('suspicious_burst_pct', 0)
                random_suspicious_pct[subject] = summary['random'].get('suspicious_burst_pct', 0)
            
            # Identify subjects with largest differences in suspicious patterns
            diff_pct = {subject: bottom_suspicious_pct[subject] - random_suspicious_pct[subject] 
                        for subject in subjects if subject in bottom_suspicious_pct and subject in random_suspicious_pct}
            
            sorted_diffs = sorted(diff_pct.items(), key=lambda x: x[1], reverse=True)
            
            f.write("Subjects Ranked by Difference in Suspicious Burst Patterns:\n")
            for subject, diff in sorted_diffs:
                bottom_pct = bottom_suspicious_pct[subject]
                random_pct = random_suspicious_pct[subject]
                f.write(f"  * {subject}: {diff:.1f}% difference (Bottom: {bottom_pct:.1f}%, Random: {random_pct:.1f}%)\n")
            
            f.write("\nSubjects Ranked by Overall Suspicious Burst Patterns in Bottom Authors:\n")
            for subject, pct in sorted(bottom_suspicious_pct.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  * {subject}: {pct:.1f}%\n")
            
            f.write("\nCitation Age Analysis by Subject:\n")
            for subject, summary in subject_temporal_metrics.items():
                if 'median_citation_age' in summary['bottom'] and 'median_citation_age' in summary['random']:
                    bottom_age = summary['bottom']['median_citation_age']
                    random_age = summary['random']['median_citation_age']
                    diff = summary['diff']['median_citation_age_diff']
                    
                    f.write(f"  * {subject}: Bottom authors cite {abs(diff):.1f} years {'older' if diff > 0 else 'newer'} work ")
                    f.write(f"than random authors (Bottom: {bottom_age:.1f} years, Random: {random_age:.1f} years)\n")
            
            f.write("\nSUSPICIOUS TEMPORAL PATTERNS BY SUBJECT:\n")
            for subject, summary in subject_temporal_metrics.items():
                # Define criteria for suspicious patterns
                if (summary['diff'].get('suspicious_burst_pct_diff', 0) > 20 or
                    summary['diff'].get('median_citation_age_diff', 0) < -2):
                    
                    f.write(f"\n{subject}:\n")
                    f.write(f"  - {'Higher' if summary['diff'].get('suspicious_burst_pct_diff', 0) > 0 else 'Lower'} rate of suspicious burst patterns: ")
                    f.write(f"{abs(summary['diff'].get('suspicious_burst_pct_diff', 0)):.1f}% difference\n")
                    
                    if 'median_citation_age_diff' in summary['diff']:
                        f.write(f"  - {'Cites older' if summary['diff']['median_citation_age_diff'] > 0 else 'Cites newer'} work: ")
                        f.write(f"{abs(summary['diff']['median_citation_age_diff']):.1f} years difference\n")
    
    print(f"[INFO] Completed temporal analysis by subject")
    return subject_temporal_metrics

# Function to integrate temporal analysis into subject batch analysis
def batch_analyze_authors_with_temporal(rolap_conn, impact_conn, schema, num_pairs=5, by_subject=True):
    """
    Analyze multiple author pairs with temporal analysis, optionally grouped by subject.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        schema: Database schema information
        num_pairs: Number of author pairs to analyze per subject (if by_subject=True)
                  or total (if by_subject=False)
        by_subject: Whether to group analysis by subject
        
    Returns:
        dict: Analysis results, with by_subject=True: {subject: [results]}, 
              with by_subject=False: [results]
    """
    print(f"[INFO] Starting batch analysis for {'subject-based' if by_subject else 'cross-subject'} author pairs with temporal analysis")
    
    # Step 1: Load data
    bad_authors = load_bad_authors(rolap_conn, schema)
    author_subjects = load_author_subjects(rolap_conn, schema)
    publication_years = load_publication_years(rolap_conn, schema)
    orcid_to_works, work_to_authors = load_work_author_mappings(rolap_conn, schema)
    work_to_cited, citation_years = load_citation_data(impact_conn, schema)
    
    # Load author pairs, potentially grouped by subject
    author_pairs = load_author_pairs(rolap_conn, schema, num_pairs, by_subject)
    
    if by_subject:
        if not author_pairs:
            print("[WARNING] No subject-based author pairs to analyze.")
            return {}
            
        # Prepare results collection by subject
        results_by_subject = {}
        
        # Create temporal analysis directory
        temporal_dir = Path(f"{OUTPUT_DIR}/temporal_analysis")
        if not temporal_dir.exists():
            temporal_dir.mkdir(parents=True)
            print(f"[INFO] Created directory for temporal analysis: {temporal_dir}")
        
        # Process each subject
        for subject, pairs in author_pairs.items():
            print(f"\n[INFO] Processing pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM) [Subject: {subject}]")
            
            # Process the pair as in the original code...
            pair_results = {}
            
            # Add temporal data dictionary to store results
            pair_temporal_data = {
                "BOTTOM": {},
                "RANDOM": {}
            }
            
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
                
                # Find suspicious patterns
                suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
                citation_rings = find_citation_rings(graph, bad_authors)
                author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_subjects=author_subjects)
                
                # TEMPORAL ANALYSIS: Add temporal citation pattern analysis
                temporal_metrics = analyze_temporal_citation_patterns(
                    graph, 
                    work_to_authors, 
                    publication_years, 
                    citation_years
                )
                
                # Store temporal metrics in the pair data
                pair_temporal_data[category] = temporal_metrics
                
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
                
                # Add temporal metrics to network metrics if available
                if temporal_metrics:
                    # Add key temporal metrics
                    if 'median_citation_age' in temporal_metrics:
                        network_metrics['median_citation_age'] = temporal_metrics['median_citation_age']
                    if 'self_vs_other_diff' in temporal_metrics:
                        network_metrics['self_vs_other_diff'] = temporal_metrics['self_vs_other_diff']
                
                # Save visualization for this author
                output_prefix = f"pair{pair_idx}_{category}_{author_id}"
                
                # TEMPORAL ANALYSIS: Create temporal visualizations
                visualize_temporal_patterns(
                    publication_years,
                    citation_years,
                    author_id,
                    f"{temporal_dir}/{output_prefix}_temporal.png",
                    subject=subject
                )
                
                # TEMPORAL ANALYSIS: Analyze citation bursts
                burst_analysis = analyze_citation_bursts(
                    publication_years,
                    citation_years,
                    author_id,
                    f"{temporal_dir}/{output_prefix}_bursts.png",
                    subject=subject
                )
                
                # Store burst metrics in network metrics
                if 'suspicious' in burst_analysis:
                    network_metrics['has_suspicious_bursts'] = burst_analysis['suspicious']
                if 'burst_count' in burst_analysis:
                    network_metrics['burst_count'] = burst_analysis['burst_count']
                if 'max_burst_intensity' in burst_analysis:
                    network_metrics['max_burst_intensity'] = burst_analysis['max_burst_intensity']
                
                # Visualize network overview
                if author_analysis:
                    visualize_author_network(
                        graph, 
                        author_analysis, 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_network.png",
                        subject=subject
                    )
                
                # Visualize top clique and ring if they exist
                if suspicious_cliques:
                    visualize_suspicious_clique(
                        graph, 
                        suspicious_cliques[0], 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_top_clique.png",
                        subject=subject
                    )
                    
                    # Save clique metrics
                    save_clique_metrics(
                        suspicious_cliques[:10],  # Save top 10
                        f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv",
                        subject=subject
                    )
                
                if citation_rings:
                    visualize_citation_ring(
                        graph, 
                        citation_rings[0], 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_top_ring.png",
                        subject=subject
                    )
                    
                    # Save ring metrics
                    save_ring_metrics(
                        citation_rings[:10],  # Save top 10
                        f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv",
                        subject=subject
                    )
                
                # Store results
                pair_results[category] = {
                    "author_id": author_id,
                    "metrics": network_metrics,
                    "suspicious_cliques": suspicious_cliques[:10],  # Store top 10
                    "citation_rings": citation_rings[:10],  # Store top 10
                    "author_analysis": author_analysis,
                    "temporal_metrics": temporal_metrics,  # Add temporal metrics
                    "burst_analysis": burst_analysis  # Add burst analysis
                }
            
            # Calculate and store temporal differences
            temporal_diff = {}
            
            # Compare citation ages
            if ('median_citation_age' in pair_temporal_data['BOTTOM'] and 
                'median_citation_age' in pair_temporal_data['RANDOM']):
                temporal_diff['citation_age_diff'] = (
                    pair_temporal_data['BOTTOM']['median_citation_age'] - 
                    pair_temporal_data['RANDOM']['median_citation_age']
                )
            
            # Compare self vs other citation differences
            if ('self_vs_other_diff' in pair_temporal_data['BOTTOM'] and 
                'self_vs_other_diff' in pair_temporal_data['RANDOM']):
                temporal_diff['self_other_citation_diff'] = (
                    pair_temporal_data['BOTTOM']['self_vs_other_diff'] - 
                    pair_temporal_data['RANDOM']['self_vs_other_diff']
                )
            
            # Add to overall results
            results.append({
                "pair_idx": pair_idx,
                "subject": subject,
                "bottom_author": pair_results["BOTTOM"]["author_id"],
                "random_author": pair_results["RANDOM"]["author_id"],
                "total_works": total_works,
                "bottom_metrics": pair_results["BOTTOM"]["metrics"],
                "random_metrics": pair_results["RANDOM"]["metrics"],
                "temporal_diff": temporal_diff  # Add temporal differences
            })
        
        # Save comparative report
        save_comparative_analysis(results, f"{OUTPUT_DIR}/comparative_analysis.csv")
        
        # Create a summary of temporal analysis
        with open(f"{temporal_dir}/temporal_analysis_summary.txt", 'w') as f:
            f.write("TEMPORAL CITATION ANALYSIS SUMMARY\n")
            f.write("=================================\n\n")
            
            # Compare citation age patterns
            median_bottom_age = np.median([r['bottom_metrics'].get('median_citation_age', 0) for r in results if 'median_citation_age' in r['bottom_metrics']])
            median_random_age = np.median([r['random_metrics'].get('median_citation_age', 0) for r in results if 'median_citation_age' in r['random_metrics']])
            
            f.write(f"Median Citation Age:\n")
            f.write(f"  Bottom Authors: {median_bottom_age:.2f} years\n")
            f.write(f"  Random Authors: {median_random_age:.2f} years\n")
            f.write(f"  Difference: {median_bottom_age - median_random_age:.2f} years\n\n")
            
            # Compare burst patterns
            bottom_suspicious = sum(1 for r in results if r['bottom_metrics'].get('has_suspicious_bursts', False))
            random_suspicious = sum(1 for r in results if r['random_metrics'].get('has_suspicious_bursts', False))
            
            f.write(f"Authors with Suspicious Burst Patterns:\n")
            f.write(f"  Bottom Authors: {bottom_suspicious} ({bottom_suspicious / len(results) * 100:.1f}%)\n")
            f.write(f"  Random Authors: {random_suspicious} ({random_suspicious / len(results) * 100:.1f}%)\n\n")
            
            # List pairs with largest temporal differences
            f.write("Pairs with Largest Citation Age Differences:\n")
            age_diff_pairs = [(r['pair_idx'], r['subject'], r['temporal_diff'].get('citation_age_diff', 0)) 
                            for r in results if 'citation_age_diff' in r['temporal_diff']]
            
            for pair_idx, subject, diff in sorted(age_diff_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                f.write(f"  Pair #{pair_idx} ({subject}): {diff:.2f} years {('older' if diff > 0 else 'newer')}\n")
            
            f.write("\nINTERPRETATION:\n")
            if median_bottom_age < median_random_age:
                f.write("- Bottom authors tend to cite more recent work compared to random authors.\n")
                f.write("  This could indicate strategic citation behavior to boost metrics.\n")
            
            if bottom_suspicious > random_suspicious:
                f.write("- Bottom authors show more suspicious burst patterns than random authors.\n")
                f.write("  This could indicate coordinated citation behavior.\n")
        
        print(f"[INFO] Batch analysis complete for {len(results)} author pairs")
        return results subject: {subject}")
            subject_results = []
            
            # Process each pair in this subject
            for pair_idx, (bottom_orcid, random_orcid, total_works, _) in enumerate(pairs, 1):
                print(f"\n[INFO] Processing {subject} pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM)")
                
                # Process this pair (similar to the original code)
                pair_results = {}
                
                # Add temporal data dictionary to store results
                pair_temporal_data = {
                    "BOTTOM": {},
                    "RANDOM": {}
                }
                
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
                    
                    # Find suspicious patterns
                    suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
                    citation_rings = find_citation_rings(graph, bad_authors)
                    author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_subjects=author_subjects)
                    
                    # TEMPORAL ANALYSIS: Add temporal citation pattern analysis
                    temporal_metrics = analyze_temporal_citation_patterns(
                        graph, 
                        work_to_authors, 
                        publication_years, 
                        citation_years
                    )
                    
                    # Store temporal metrics in the pair data
                    pair_temporal_data[category] = temporal_metrics
                    
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
                    
                    # Add temporal metrics to network metrics if available
                    if temporal_metrics:
                        # Add key temporal metrics
                        if 'median_citation_age' in temporal_metrics:
                            network_metrics['median_citation_age'] = temporal_metrics['median_citation_age']
                        if 'self_vs_other_diff' in temporal_metrics:
                            network_metrics['self_vs_other_diff'] = temporal_metrics['self_vs_other_diff']
                    
                    # Save visualization for this author with subject prefix
                    output_prefix = f"{subject}_pair{pair_idx}_{category}_{author_id}"
                    
                    # TEMPORAL ANALYSIS: Create temporal visualizations
                    visualize_temporal_patterns(
                        publication_years,
                        citation_years,
                        author_id,
                        f"{temporal_dir}/{output_prefix}_temporal.png",
                        subject=subject
                    )
                    
                    # TEMPORAL ANALYSIS: Analyze citation bursts
                    burst_analysis = analyze_citation_bursts(
                        publication_years,
                        citation_years,
                        author_id,
                        f"{temporal_dir}/{output_prefix}_bursts.png",
                        subject=subject
                    )
                    
                    # Store burst metrics in network metrics
                    if 'suspicious' in burst_analysis:
                        network_metrics['has_suspicious_bursts'] = burst_analysis['suspicious']
                    if 'burst_count' in burst_analysis:
                        network_metrics['burst_count'] = burst_analysis['burst_count']
                    if 'max_burst_intensity' in burst_analysis:
                        network_metrics['max_burst_intensity'] = burst_analysis['max_burst_intensity']
                    
                    # Visualize network overview
                    if author_analysis:
                        visualize_author_network(
                            graph, 
                            author_analysis, 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_network.png",
                            subject=subject
                        )
                    
                    # Visualize top clique and ring if they exist
                    if suspicious_cliques:
                        visualize_suspicious_clique(
                            graph, 
                            suspicious_cliques[0], 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_top_clique.png",
                            subject=subject
                        )
                        
                        # Save clique metrics
                        save_clique_metrics(
                            suspicious_cliques[:10],  # Save top 10
                            f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv",
                            subject=subject
                        )
                    
                    if citation_rings:
                        visualize_citation_ring(
                            graph, 
                            citation_rings[0], 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_top_ring.png",
                            subject=subject
                        )
                        
                        # Save ring metrics
                        save_ring_metrics(
                            citation_rings[:10],  # Save top 10
                            f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv",
                            subject=subject
                        )
                    
                    # Store results
                    pair_results[category] = {
                        "author_id": author_id,
                        "metrics": network_metrics,
                        "suspicious_cliques": suspicious_cliques[:10],  # Store top 10
                        "citation_rings": citation_rings[:10],  # Store top 10
                        "author_analysis": author_analysis,
                        "temporal_metrics": temporal_metrics,  # Add temporal metrics
                        "burst_analysis": burst_analysis  # Add burst analysis
                    }
                
                # Calculate and store temporal differences
                temporal_diff = {}
                
                # Compare citation ages
                if ('median_citation_age' in pair_temporal_data['BOTTOM'] and 
                    'median_citation_age' in pair_temporal_data['RANDOM']):
                    temporal_diff['citation_age_diff'] = (
                        pair_temporal_data['BOTTOM']['median_citation_age'] - 
                        pair_temporal_data['RANDOM']['median_citation_age']
                    )
                
                # Compare self vs other citation differences
                if ('self_vs_other_diff' in pair_temporal_data['BOTTOM'] and 
                    'self_vs_other_diff' in pair_temporal_data['RANDOM']):
                    temporal_diff['self_other_citation_diff'] = (
                        pair_temporal_data['BOTTOM']['self_vs_other_diff'] - 
                        pair_temporal_data['RANDOM']['self_vs_other_diff']
                    )
                
                # Add to subject results
                subject_results.append({
                    "pair_idx": pair_idx,
                    "subject": subject,
                    "bottom_author": pair_results["BOTTOM"]["author_id"],
                    "random_author": pair_results["RANDOM"]["author_id"],
                    "total_works": total_works,
                    "bottom_metrics": pair_results["BOTTOM"]["metrics"],
                    "random_metrics": pair_results["RANDOM"]["metrics"],
                    "temporal_diff": temporal_diff  # Add temporal differences
                })
            
            # Store all results for this subject
            results_by_subject[subject] = subject_results
            
            # Save per-subject comparative report
            save_comparative_analysis(
                subject_results, 
                f"{OUTPUT_DIR}/{subject}_comparative_analysis.csv"
            )
            
        # After all subjects are processed, create an overall comparative analysis
        all_results = []
        for subject, results in results_by_subject.items():
            all_results.extend(results)
            
        # Save overall comparative report
        save_comparative_analysis(all_results, f"{OUTPUT_DIR}/all_subjects_comparative_analysis.csv")
        
        # Analyze cross-subject patterns
        analyze_subject_patterns(results_by_subject, OUTPUT_DIR)
        
        # TEMPORAL ANALYSIS: Analyze temporal patterns across subjects
        analyze_subject_temporal_patterns(
            results_by_subject, 
            OUTPUT_DIR, 
            publication_years, 
            citation_years
        )
        
        print(f"[INFO] Batch analysis complete for {len(all_results)} author pairs across {len(results_by_subject)} subjects")
        return results_by_subject
        
    else:
        # Original behavior: flat list of results
        if not author_pairs:
            print("[WARNING] No author pairs to analyze.")
            return []
            
        # Step 2: Prepare results collection
        results = []
        
        # Create temporal analysis directory
        temporal_dir = Path(f"{OUTPUT_DIR}/temporal_analysis")
        if not temporal_dir.exists():
            temporal_dir.mkdir(parents=True)
            print(f"[INFO] Created directory for temporal analysis: {temporal_dir}")
            
        # Step 3: Process each author pair
        for pair_idx, (bottom_orcid, random_orcid, total_works, subject) in enumerate(author_pairs, 1):
            print(f"\n[INFO] Processing pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM) [Subject: {subject}]")

##############################################################################
# 11) Batch Analysis for Multiple Author Pairs
##############################################################################
def batch_analyze_authors(rolap_conn, impact_conn, schema, num_pairs=5, by_subject=True):
    """
    Analyze multiple author pairs and generate comparative reports, optionally grouped by subject.
    
    Args:
        rolap_conn: SQLite connection to ROLAP database
        impact_conn: SQLite connection to IMPACT database
        schema: Database schema information
        num_pairs: Number of author pairs to analyze per subject (if by_subject=True)
                  or total (if by_subject=False)
        by_subject: Whether to group analysis by subject
        
    Returns:
        dict: Analysis results, with by_subject=True: {subject: [results]}, 
              with by_subject=False: [results]
    """
    print(f"[INFO] Starting batch analysis for {'subject-based' if by_subject else 'cross-subject'} author pairs")
    
    # Step 1: Load data
    bad_authors = load_bad_authors(rolap_conn, schema)
    author_subjects = load_author_subjects(rolap_conn, schema)
    orcid_to_works, work_to_authors = load_work_author_mappings(rolap_conn, schema)
    work_to_cited, citation_years = load_citation_data(impact_conn, schema)
    
    # Load author pairs, potentially grouped by subject
    author_pairs = load_author_pairs(rolap_conn, schema, num_pairs, by_subject)
    
    if by_subject:
        if not author_pairs:
            print("[WARNING] No subject-based author pairs to analyze.")
            return {}
            
        # Prepare results collection by subject
        results_by_subject = {}
        
        # Process each subject
        for subject, pairs in author_pairs.items():
            print(f"\n[INFO] Processing subject: {subject}")
            subject_results = []
            
            # Process each pair in this subject
            for pair_idx, (bottom_orcid, random_orcid, total_works, _) in enumerate(pairs, 1):
                print(f"\n[INFO] Processing {subject} pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM)")
                
                # Process this pair (similar to the original code)
                pair_results = {}
                
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
                    
                    # Find suspicious patterns
                    suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
                    citation_rings = find_citation_rings(graph, bad_authors)
                    author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_subjects=author_subjects)
                    
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
                    
                    # Save visualization for this author with subject prefix
                    output_prefix = f"{subject}_pair{pair_idx}_{category}_{author_id}"
                    
                    # Visualize network overview
                    if author_analysis:
                        visualize_author_network(
                            graph, 
                            author_analysis, 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_network.png",
                            subject=subject
                        )
                    
                    # Visualize top clique and ring if they exist
                    if suspicious_cliques:
                        visualize_suspicious_clique(
                            graph, 
                            suspicious_cliques[0], 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_top_clique.png",
                            subject=subject
                        )
                        
                        # Save clique metrics
                        save_clique_metrics(
                            suspicious_cliques[:10],  # Save top 10
                            f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv",
                            subject=subject
                        )
                    
                    if citation_rings:
                        visualize_citation_ring(
                            graph, 
                            citation_rings[0], 
                            bad_authors, 
                            f"{OUTPUT_DIR}/{output_prefix}_top_ring.png",
                            subject=subject
                        )
                        
                        # Save ring metrics
                        save_ring_metrics(
                            citation_rings[:10],  # Save top 10
                            f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv",
                            subject=subject
                        )
                    
                    # Store results
                    pair_results[category] = {
                        "author_id": author_id,
                        "metrics": network_metrics,
                        "suspicious_cliques": suspicious_cliques[:10],  # Store top 10
                        "citation_rings": citation_rings[:10],  # Store top 10
                        "author_analysis": author_analysis
                    }
                
                # Add to subject results
                subject_results.append({
                    "pair_idx": pair_idx,
                    "subject": subject,
                    "bottom_author": pair_results["BOTTOM"]["author_id"],
                    "random_author": pair_results["RANDOM"]["author_id"],
                    "total_works": total_works,
                    "bottom_metrics": pair_results["BOTTOM"]["metrics"],
                    "random_metrics": pair_results["RANDOM"]["metrics"]
                })
            
            # Store all results for this subject
            results_by_subject[subject] = subject_results
            
            # Save per-subject comparative report
            save_comparative_analysis(
                subject_results, 
                f"{OUTPUT_DIR}/{subject}_comparative_analysis.csv"
            )
            
        # After all subjects are processed, create an overall comparative analysis
        all_results = []
        for subject, results in results_by_subject.items():
            all_results.extend(results)
            
        # Save overall comparative report
        save_comparative_analysis(all_results, f"{OUTPUT_DIR}/all_subjects_comparative_analysis.csv")
        
        # Analyze cross-subject patterns
        analyze_subject_patterns(results_by_subject, OUTPUT_DIR)
        
        print(f"[INFO] Batch analysis complete for {len(all_results)} author pairs across {len(results_by_subject)} subjects")
        return results_by_subject
        
    else:
        # Original behavior: flat list of results
        if not author_pairs:
            print("[WARNING] No author pairs to analyze.")
            return []
            
        # Step 2: Prepare results collection
        results = []
        
        # Step 3: Process each author pair
        for pair_idx, (bottom_orcid, random_orcid, total_works, subject) in enumerate(author_pairs, 1):
            print(f"\n[INFO] Processing pair #{pair_idx}: {bottom_orcid} (BOTTOM) vs {random_orcid} (RANDOM) [Subject: {subject}]")
            
            # Process the pair as in the original code...
            pair_results = {}
            
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
                
                # Find suspicious patterns
                suspicious_cliques = find_suspicious_cliques(graph, bad_authors)
                citation_rings = find_citation_rings(graph, bad_authors)
                author_analysis = analyze_author_citation_behavior(graph, bad_authors, author_subjects=author_subjects)
                
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
                
                # Save visualization for this author
                output_prefix = f"pair{pair_idx}_{category}_{author_id}"
                
                # Visualize network overview
                if author_analysis:
                    visualize_author_network(
                        graph, 
                        author_analysis, 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_network.png",
                        subject=subject
                    )
                
                # Visualize top clique and ring if they exist
                if suspicious_cliques:
                    visualize_suspicious_clique(
                        graph, 
                        suspicious_cliques[0], 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_top_clique.png",
                        subject=subject
                    )
                    
                    # Save clique metrics
                    save_clique_metrics(
                        suspicious_cliques[:10],  # Save top 10
                        f"{OUTPUT_DIR}/{output_prefix}_suspicious_cliques.csv",
                        subject=subject
                    )
                
                if citation_rings:
                    visualize_citation_ring(
                        graph, 
                        citation_rings[0], 
                        bad_authors, 
                        f"{OUTPUT_DIR}/{output_prefix}_top_ring.png",
                        subject=subject
                    )
                    
                    # Save ring metrics
                    save_ring_metrics(
                        citation_rings[:10],  # Save top 10
                        f"{OUTPUT_DIR}/{output_prefix}_citation_rings.csv",
                        subject=subject
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
                "subject": subject,
                "bottom_author": pair_results["BOTTOM"]["author_id"],
                "random_author": pair_results["RANDOM"]["author_id"],
                "total_works": total_works,
                "bottom_metrics": pair_results["BOTTOM"]["metrics"],
                "random_metrics": pair_results["RANDOM"]["metrics"]
            })
        
        # Save comparative report
        save_comparative_analysis(results, f"{OUTPUT_DIR}/comparative_analysis.csv")
        
        print(f"[INFO] Batch analysis complete for {len(results)} author pairs")
        return results


##############################################################################
# 12) Command Line Interface and Main Function
##############################################################################
def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Citation Network Analysis Tool')
    
    parser.add_argument('--rolap-db', type=str, default=ROLAP_DB,
                      help='Path to ROLAP database (default: rolap.db)')
    
    parser.add_argument('--impact-db', type=str, default=IMPACT_DB,
                      help='Path to IMPACT database (default: impact.db)')
    
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                      help=f'Output directory (default: {OUTPUT_DIR})')
    
    parser.add_argument('--by-subject', action='store_true', default=True,
                      help='Group analysis by subject (default: True)')
    
    parser.add_argument('--num-pairs', type=int, default=3,
                      help='Number of author pairs to analyze per subject (default: 3)')
    
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                      help=f'Depth of BFS traversal (default: {DEFAULT_DEPTH})')
    
    parser.add_argument('--max-nodes', type=int, default=MAX_BFS_NODES,
                      help=f'Maximum number of nodes in network (default: {MAX_BFS_NODES})')
    
    parser.add_argument('--min-clique-size', type=int, default=SUSPICIOUS_CLIQUE_MIN_SIZE,
                      help=f'Minimum size of suspicious cliques (default: {SUSPICIOUS_CLIQUE_MIN_SIZE})')
    
    parser.add_argument('--min-ring-size', type=int, default=CITATION_RING_MIN_SIZE,
                      help=f'Minimum size of citation rings (default: {CITATION_RING_MIN_SIZE})')
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Update global parameters
        global ROLAP_DB, IMPACT_DB, OUTPUT_DIR, DEFAULT_DEPTH, MAX_BFS_NODES
        global SUSPICIOUS_CLIQUE_MIN_SIZE, CITATION_RING_MIN_SIZE
        
        ROLAP_DB = args.rolap_db
        IMPACT_DB = args.impact_db
        OUTPUT_DIR = Path(args.output_dir)
        DEFAULT_DEPTH = args.depth
        MAX_BFS_NODES = args.max_nodes
        SUSPICIOUS_CLIQUE_MIN_SIZE = args.min_clique_size
        CITATION_RING_MIN_SIZE = args.min_ring_size
        
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
        
        # Run batch analysis for multiple authors grouped by subject
        batch_results = batch_analyze_authors(
            rolap_conn, 
            impact_conn, 
            schema, 
            num_pairs=args.num_pairs,
            by_subject=args.by_subject
        )
        
        # Perform deeper analysis on the most suspicious author if we have results
        if batch_results:
            try:
                # Load subjects if available
                author_subjects = load_author_subjects(rolap_conn, schema)
                
                # Find the author with the most suspicious cliques across all subjects
                if args.by_subject:
                    # Find most suspicious author across all subjects
                    most_suspicious = None
                    max_cliques = 0
                    most_suspicious_subject = None
                    
                    for subject, results in batch_results.items():
                        for result in results:
                            if result["bottom_metrics"]["suspicious_cliques"] > max_cliques:
                                max_cliques = result["bottom_metrics"]["suspicious_cliques"]
                                most_suspicious = result
                                most_suspicious_subject = subject
                else:
                    # Original behavior
                    most_suspicious = max(
                        batch_results,
                        key=lambda x: x["bottom_metrics"]["suspicious_cliques"]
                    )
                    most_suspicious_subject = most_suspicious.get("subject", "Unknown")
                
                if most_suspicious:
                    bottom_author = most_suspicious["bottom_author"]
                    
                    print(f"\n[INFO] Performing deep analysis on most suspicious author: {bottom_author} (Subject: {most_suspicious_subject})")
                    
                    # Analyze this author in more detail
                    analyze_author_network(
                        rolap_conn, 
                        impact_conn, 
                        bottom_author, 
                        f"deep_analysis_{most_suspicious_subject}",
                        schema,
                        author_subjects
                    )
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
    