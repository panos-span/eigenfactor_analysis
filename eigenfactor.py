import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm
import multiprocessing as mp
from functools import partial

# Constants
ALPHA = 0.85
EPSILON = 1e-6
MAX_ITER = 1000

def calculate_eigenfactor_sparse(group: pd.DataFrame, journal_article_counts: dict, 
                               alpha=0.85, epsilon=1e-6, max_iter=1000):
    """
    Optimized Eigenfactor calculation using sparse matrices and vectorized operations.
    
    Mathematical Foundation:
    - Eigenfactor solves: π = α * H * π + (α * d^T * π + (1-α)) * a
    - Where H is column-stochastic transition matrix, d is dangling nodes vector, a is article vector
    
    Optimization Techniques:
    1. Sparse matrix representation (CSR format)
    2. Efficient handling of dangling nodes
    3. Vectorized operations for teleportation
    4. Early convergence detection
    """
    
    # Step 1: Create journal mapping and validate data
    journals = sorted(list(set(group['citing_issn']).union(set(group['cited_issn']))))
    journal_to_idx = {journal: i for i, journal in enumerate(journals)}
    n = len(journals)
    
    if n == 0:
        return pd.DataFrame(columns=['issn', 'eigenfactor_score'])
    
    # Step 2: Build sparse adjacency matrix efficiently
    citing_indices = group['citing_issn'].map(journal_to_idx).values
    cited_indices = group['cited_issn'].map(journal_to_idx).values
    citation_counts = group['citation_count'].values
    
    # Create sparse matrix (citations FROM citing TO cited)
    # Using COO format for construction, then convert to CSR for efficiency
    Z_coo = csr_matrix((citation_counts, (citing_indices, cited_indices)), 
                       shape=(n, n), dtype=np.float32)
    
    # Remove self-citations efficiently
    Z_coo.setdiag(0)
    Z_coo.eliminate_zeros()
    
    # Transpose for column-stochastic operations (Z_T[i,j] = citation from j to i)
    H = Z_coo.T.tocsr()
    
    # Step 3: Create article vector with error handling
    article_counts = np.array([journal_article_counts.get(j, 0) for j in journals], dtype=np.float32)
    total_articles = article_counts.sum()
    
    if total_articles > 0:
        article_vector = article_counts / total_articles
    else:
        # Fallback for missing article data
        print(f"WARNING: No article data for subject. Using uniform distribution.")
        article_vector = np.ones(n, dtype=np.float32) / n
    
    # Step 4: Identify dangling nodes and normalize transition matrix
    column_sums = np.array(H.sum(axis=0)).flatten()
    dangling_mask = column_sums == 0
    
    # Normalize non-dangling columns
    non_zero_mask = column_sums > 0
    if np.any(non_zero_mask):
        # Use diagonal matrix for efficient column normalization
        inv_col_sums = np.zeros(n, dtype=np.float32)
        inv_col_sums[non_zero_mask] = 1.0 / column_sums[non_zero_mask]
        D_inv = diags(inv_col_sums, format='csr')
        H = H @ D_inv
    
    # Step 5: Power iteration with sparse operations
    pi = np.ones(n, dtype=np.float32) / n  # Initial uniform distribution
    
    for iteration in range(max_iter):
        # Standard PageRank update with teleportation
        # π_new = α * H * π + (α * dangling_sum + (1-α)) * article_vector
        
        # Calculate influence from dangling nodes
        dangling_sum = pi[dangling_mask].sum() if np.any(dangling_mask) else 0.0
        
        # Sparse matrix-vector multiplication
        pi_new = alpha * (H @ pi)
        
        # Add teleportation term
        teleport_term = alpha * dangling_sum + (1 - alpha)
        pi_new += teleport_term * article_vector
        
        # Check convergence using L1 norm
        if np.linalg.norm(pi_new - pi, ord=1) < epsilon:
            pi = pi_new
            break
            
        pi = pi_new
    
    # Step 6: Calculate final Eigenfactor scores
    # Normalize so pi sums to 1
    pi = pi / pi.sum()
    
    # Eigenfactor = influence received (H @ pi)
    eigenfactor_scores = H @ pi
    
    # Scale to sum to 100 (Eigenfactor convention)
    eigenfactor_scores = 100 * eigenfactor_scores / eigenfactor_scores.sum()
    
    return pd.DataFrame({
        'issn': journals,
        'eigenfactor_score': eigenfactor_scores
    })

def process_subject_parallel(args):
    """Helper function for parallel processing"""
    subject, group, journal_article_counts = args
    try:
        result = calculate_eigenfactor_sparse(group, journal_article_counts)
        result['subject'] = subject
        return result
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        return pd.DataFrame()

def calculate_eigenfactor_parallel(df, journal_article_counts, n_processes=None):
    """
    Parallel processing of Eigenfactor calculations by subject.
    
    Args:
        df: Citation network DataFrame
        journal_article_counts: Dictionary of journal article counts
        n_processes: Number of processes (default: CPU count)
    """
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    # Prepare arguments for parallel processing
    citation_groups = [(subject, group, journal_article_counts) 
                      for subject, group in df.groupby('subject')]
    
    print(f"Processing {len(citation_groups)} subjects using {n_processes} processes...")
    
    # Use multiprocessing Pool
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_subject_parallel, citation_groups),
            total=len(citation_groups),
            desc="Computing Eigenfactor scores"
        ))
    
    # Filter out empty results and combine
    valid_results = [r for r in results if not r.empty]
    
    if valid_results:
        return pd.concat(valid_results, ignore_index=True)
    else:
        return pd.DataFrame()

def optimize_data_loading(citation_file, articles_file):
    """
    Optimized data loading with memory-efficient dtypes and chunking.
    """
    # Use categorical for repeated string values to save memory
    citation_dtypes = {
        'citing_issn': 'category',
        'cited_issn': 'category', 
        'subject': 'category',
        'citation_count': 'int32'  # int32 sufficient for most citation counts
    }
    
    articles_dtypes = {
        'issn': 'category',
        'article_count': 'int32'
    }
    
    # Load with optimized dtypes
    df = pd.read_csv(citation_file, header=None, 
                    names=['citing_issn', 'cited_issn', 'subject', 'citation_count'],
                    sep='|', dtype=citation_dtypes)
    
    articles_df = pd.read_csv(articles_file, header=None,
                             names=['issn', 'article_count'],
                             sep='|', dtype=articles_dtypes)
    
    # Convert to dictionary efficiently
    journal_article_counts = dict(zip(articles_df['issn'], articles_df['article_count']))
    
    return df, journal_article_counts

def batch_process_large_datasets(df, journal_article_counts, batch_size=50):
    """
    Process large datasets in batches to manage memory usage.
    """
    subjects = df['subject'].unique()
    n_batches = len(subjects) // batch_size + (1 if len(subjects) % batch_size else 0)
    
    all_results = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(subjects))
        batch_subjects = subjects[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{n_batches} ({len(batch_subjects)} subjects)")
        
        # Filter data for current batch
        batch_df = df[df['subject'].isin(batch_subjects)]
        
        # Process batch
        batch_results = calculate_eigenfactor_parallel(batch_df, journal_article_counts)
        all_results.append(batch_results)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# Main execution with optimizations
if __name__ == "__main__":
    print("Loading data with optimized dtypes...")
    df, journal_article_counts = optimize_data_loading(
        'get_citation_network.txt', 
        'journal_articles.txt'
    )
    
    print(f"Loaded {len(df):,} citation records across {df['subject'].nunique()} subjects")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check if dataset is large enough to warrant batch processing
    if df['subject'].nunique() > 100:
        print("Large dataset detected. Using batch processing...")
        eigenfactor_combined_df = batch_process_large_datasets(df, journal_article_counts)
    else:
        eigenfactor_combined_df = calculate_eigenfactor_parallel(df, journal_article_counts)
    
    # Save results
    print("Saving results...")
    eigenfactor_combined_df.to_csv('eigenfactor_scores_optimized.csv', index=False)
    
    # Database operations
    conn = sqlite3.connect('impact.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS eigenfactor_scores')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eigenfactor_scores (
            issn TEXT,
            subject TEXT,
            eigenfactor_score REAL,
            PRIMARY KEY (issn, subject)
        )
    ''')
    
    eigenfactor_combined_df.to_sql('eigenfactor_scores', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    
    print(f"Successfully processed {len(eigenfactor_combined_df):,} journal-subject combinations")