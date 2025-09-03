import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def analyze_eigenfactor_distribution(db_path='impact.db'):
    """
    Analyze the distribution of Eigenfactor scores across all subjects and by subject
    to determine appropriate absolute thresholds.
    
    Args:
        db_path: Path to SQLite database containing eigenfactor_scores table
        
    Returns:
        DataFrame with distribution statistics
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Load eigenfactor scores
    ef_scores = pd.read_sql_query("SELECT * FROM eigenfactor_scores", conn)
    
    # Basic statistics for the entire dataset
    global_stats = ef_scores['eigenfactor_score'].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print("Global Eigenfactor Statistics:")
    print(global_stats)
    
    # Check skewness and kurtosis
    skewness = ef_scores['eigenfactor_score'].skew()
    kurtosis = ef_scores['eigenfactor_score'].kurt()
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    
    # Calculate statistics by subject
    subject_stats = ef_scores.groupby('subject')['eigenfactor_score'].describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).reset_index()
    
    # Save a subset of top and bottom subjects by eigenfactor scores for visualization
    subject_medians = ef_scores.groupby('subject')['eigenfactor_score'].median().sort_values()
    bottom_subjects = subject_medians.index[:5].tolist()
    top_subjects = subject_medians.index[-5:].tolist()
    highlighted_subjects = bottom_subjects + top_subjects
    
    # Visualize the distribution
    create_distribution_plots(ef_scores, highlighted_subjects)
    
    # Analyze high variance between subjects
    subject_variance = ef_scores.groupby('subject')['eigenfactor_score'].var().sort_values(ascending=False)
    print("\nSubjects with highest Eigenfactor variance:")
    print(subject_variance.head(10))
    
    # Calculate potential threshold values
    thresholds = calculate_threshold_recommendations(ef_scores)
    
    conn.close()
    
    return {
        'global_stats': global_stats,
        'subject_stats': subject_stats,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'thresholds': thresholds
    }

def create_distribution_plots(ef_scores, highlighted_subjects):
    """
    Create visualizations of the Eigenfactor distribution.
    
    Args:
        ef_scores: DataFrame with Eigenfactor scores
        highlighted_subjects: List of subjects to highlight in the plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Global distribution histogram (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(ef_scores['eigenfactor_score'], kde=True, ax=ax1)
    ax1.set_title('Global Eigenfactor Distribution')
    ax1.set_xlabel('Eigenfactor Score')
    ax1.set_ylabel('Frequency')
    
    # Plot 2: Global distribution histogram (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log10(ef_scores['eigenfactor_score'].clip(lower=1e-10)), kde=True, ax=ax2)
    ax2.set_title('Global Eigenfactor Distribution (Log10 Scale)')
    ax2.set_xlabel('Log10(Eigenfactor Score)')
    ax2.set_ylabel('Frequency')
    
    # Plot 3: Box plot for highlighted subjects
    ax3 = fig.add_subplot(gs[1, :])
    highlighted_data = ef_scores[ef_scores['subject'].isin(highlighted_subjects)]
    sns.boxplot(x='subject', y='eigenfactor_score', data=highlighted_data, ax=ax3)
    ax3.set_title('Eigenfactor Distribution by Subject (Selected Subjects)')
    ax3.set_xlabel('Subject')
    ax3.set_ylabel('Eigenfactor Score')
    ax3.tick_params(axis='x', rotation=90)
    
    # Plot 4: Box plot for highlighted subjects (log scale)
    ax4 = fig.add_subplot(gs[2, :])
    sns.boxplot(x='subject', y='eigenfactor_score', data=highlighted_data, ax=ax4)
    ax4.set_yscale('log')
    ax4.set_title('Eigenfactor Distribution by Subject (Log Scale)')
    ax4.set_xlabel('Subject')
    ax4.set_ylabel('Eigenfactor Score (Log Scale)')
    ax4.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig('eigenfactor_distribution.png', dpi=300)
    plt.close()
    
    # Create CDF plot to help identify thresholds
    plt.figure(figsize=(12, 8))
    
    # Calculate CDF
    ef_sorted = np.sort(ef_scores['eigenfactor_score'])
    y = np.arange(1, len(ef_sorted)+1) / len(ef_sorted)
    
    plt.plot(ef_sorted, y)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.title('Cumulative Distribution Function of Eigenfactor Scores')
    plt.xlabel('Eigenfactor Score (Log Scale)')
    plt.ylabel('Cumulative Probability')
    
    # Add vertical lines at potential thresholds
    plt.axvline(x=np.percentile(ef_scores['eigenfactor_score'], 75), color='r', linestyle='--', 
                label=f'75th Percentile: {np.percentile(ef_scores["eigenfactor_score"], 75):.6f}')
    plt.axvline(x=np.percentile(ef_scores['eigenfactor_score'], 25), color='g', linestyle='--', 
                label=f'25th Percentile: {np.percentile(ef_scores["eigenfactor_score"], 25):.6f}')
    plt.axvline(x=ef_scores['eigenfactor_score'].median(), color='b', linestyle='--', 
                label=f'Median: {ef_scores["eigenfactor_score"].median():.6f}')
    
    plt.legend()
    plt.savefig('eigenfactor_cdf.png', dpi=300)
    plt.close()

def calculate_threshold_recommendations(ef_scores):
    """
    Calculate recommended threshold values based on the distribution.
    
    Args:
        ef_scores: DataFrame with Eigenfactor scores
        
    Returns:
        Dictionary with recommended threshold values
    """
    # Calculate global percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    global_percentiles = np.percentile(ef_scores['eigenfactor_score'], percentiles)
    
    # Calculate mean and standard deviation for potential z-score based thresholds
    mean = ef_scores['eigenfactor_score'].mean()
    std = ef_scores['eigenfactor_score'].std()
    
    # Calculate subject-specific thresholds
    subject_thresholds = {}
    for subject, group in ef_scores.groupby('subject'):
        subject_percentiles = np.percentile(group['eigenfactor_score'], [25, 75])
        subject_thresholds[subject] = {
            'p25': subject_percentiles[0],
            'p75': subject_percentiles[1],
            'median': group['eigenfactor_score'].median(),
            'count': len(group)
        }
    
    # Find natural breaks in the distribution
    # This uses Jenks natural breaks optimization algorithm from PySAL
    try:
        from pysal.viz.mapclassify import Natural_Breaks
        nb = Natural_Breaks(ef_scores['eigenfactor_score'], k=5)
        natural_breaks = nb.bins.tolist()
    except (ImportError, ModuleNotFoundError):
        # If PySAL is not available, use a simple approximation
        sorted_vals = sorted(ef_scores['eigenfactor_score'])
        n = len(sorted_vals)
        natural_breaks = [sorted_vals[int(i*n/5)] for i in range(1, 5)]
        natural_breaks.append(sorted_vals[-1])
    
    # Create recommendation
    recommendations = {
        'global_percentiles': {f'p{p}': val for p, val in zip(percentiles, global_percentiles)},
        'z_score_based': {
            'mean': mean,
            'std': std,
            'mean_plus_1std': mean + std,
            'mean_plus_2std': mean + 2*std,
            'mean_minus_1std': max(0, mean - std),
            'mean_minus_2std': max(0, mean - 2*std)
        },
        'natural_breaks': natural_breaks,
        'subject_specific': subject_thresholds
    }
    
    # Generate SQL queries for different threshold approaches
    recommendations['sql_examples'] = {
        'global_top_25_percent': f"eigenfactor_score >= {global_percentiles[5]:.10f}",
        'global_bottom_25_percent': f"eigenfactor_score <= {global_percentiles[3]:.10f}",
        'z_score_top': f"eigenfactor_score >= {(mean + std):.10f}",
        'z_score_bottom': f"eigenfactor_score <= {max(0, mean - std):.10f}",
        'natural_break_top': f"eigenfactor_score >= {natural_breaks[-2]:.10f}",
        'natural_break_bottom': f"eigenfactor_score <= {natural_breaks[1]:.10f}"
    }
    
    return recommendations

def generate_threshold_sql(analysis_results, db_path='impact.db'):
    """
    Generate SQL queries for different threshold approaches based on the analysis.
    
    Args:
        analysis_results: Results from analyze_eigenfactor_distribution
        db_path: Path to SQLite database
    
    Returns:
        Dictionary with SQL queries for different threshold approaches
    """
    thresholds = analysis_results['thresholds']
    
    # Connect to database to get subject count
    conn = sqlite3.connect(db_path)
    subject_count = pd.read_sql_query("SELECT COUNT(DISTINCT subject) FROM eigenfactor_scores", conn).iloc[0, 0]
    conn.close()
    
    # Generate SQL queries
    sql_queries = {
        'Global Absolute Top 25%': f"""
-- Top journals based on global top 25% threshold
CREATE TABLE rolap.top_issn_by_subject AS
SELECT issn, subject
FROM eigenfactor_scores
WHERE eigenfactor_score >= {thresholds['global_percentiles']['p75']:.10f};
""",
        'Global Absolute Bottom 25%': f"""
-- Bottom journals based on global bottom 25% threshold
CREATE TABLE rolap.bottom_issn_by_subject AS
SELECT issn, subject
FROM eigenfactor_scores
WHERE eigenfactor_score <= {thresholds['global_percentiles']['p25']:.10f};
""",
        'Subject-Specific Top 25%': f"""
-- Top journals based on subject-specific top 25% threshold
CREATE TABLE rolap.top_issn_by_subject AS
WITH subject_thresholds AS (
    SELECT 
        subject,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY eigenfactor_score) AS p75
    FROM eigenfactor_scores
    GROUP BY subject
)
SELECT e.issn, e.subject
FROM eigenfactor_scores e
JOIN subject_thresholds st ON e.subject = st.subject
WHERE e.eigenfactor_score >= st.p75;
""",
        'Natural Breaks Top': f"""
-- Top journals based on natural breaks
CREATE TABLE rolap.top_issn_by_subject AS
SELECT issn, subject
FROM eigenfactor_scores
WHERE eigenfactor_score >= {thresholds['natural_breaks'][-2]:.10f};
""",
        'Z-Score Top (Mean + 1 Std)': f"""
-- Top journals based on z-score (mean + 1 std)
CREATE TABLE rolap.top_issn_by_subject AS
SELECT issn, subject
FROM eigenfactor_scores
WHERE eigenfactor_score >= {thresholds['z_score_based']['mean_plus_1std']:.10f};
""",
        'Combined Approach': f"""
-- Top journals using a combined approach:
-- Either in global top 10% OR in subject-specific top 25%
CREATE TABLE rolap.top_issn_by_subject AS
WITH subject_thresholds AS (
    SELECT 
        subject,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY eigenfactor_score) AS p75
    FROM eigenfactor_scores
    GROUP BY subject
)
SELECT e.issn, e.subject
FROM eigenfactor_scores e
LEFT JOIN subject_thresholds st ON e.subject = st.subject
WHERE e.eigenfactor_score >= {thresholds['global_percentiles']['p90']:.10f}  -- Global top 10%
   OR (e.eigenfactor_score >= st.p75);  -- OR Subject-specific top 25%
"""
    }
    
    return sql_queries

if __name__ == "__main__":
    # Analyze the distribution
    analysis_results = analyze_eigenfactor_distribution()
    
    # Print recommended thresholds
    print("\nRecommended Eigenfactor Thresholds:")
    print("----------------------------------")
    print(f"Top 25% (Global): {analysis_results['thresholds']['global_percentiles']['p75']:.10f}")
    print(f"Bottom 25% (Global): {analysis_results['thresholds']['global_percentiles']['p25']:.10f}")
    print(f"Mean + 1 Std: {analysis_results['thresholds']['z_score_based']['mean_plus_1std']:.10f}")
    print(f"Natural Break (Top): {analysis_results['thresholds']['natural_breaks'][-2]:.10f}")
    
    # Generate SQL queries
    sql_queries = generate_threshold_sql(analysis_results)
    
    # Save the SQL queries to a file
    with open('eigenfactor_threshold_sql.sql', 'w') as f:
        for name, query in sql_queries.items():
            f.write(f"-- {name}\n")
            f.write(f"{query}\n\n")
    
    print("\nSQL queries for different threshold approaches have been saved to 'eigenfactor_threshold_sql.sql'")
    
    # Save analysis results to CSV for further analysis
    subject_stats_df = analysis_results['subject_stats']
    subject_stats_df.to_csv('eigenfactor_subject_stats.csv', index=False)
    
    # Create a summary of subject-specific thresholds
    subject_thresholds = pd.DataFrame.from_dict(
        analysis_results['thresholds']['subject_specific'], 
        orient='index'
    ).reset_index().rename(columns={'index': 'subject'})
    subject_thresholds.to_csv('eigenfactor_subject_thresholds.csv', index=False)
    
    print("Analysis complete! Results saved to CSV files and visualizations.")