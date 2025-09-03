"""
Matched-Pair Citation Network Analysis
======================================
This script performs the final analysis for the research project, testing the
hypothesis that bottom-tier authors exhibit different citation behaviors than
their matched top-tier peers.

This enhanced version includes:
1.  **Direct Comparison of Behavioral Metrics** from the SQL pipeline.
2.  **Structural Network Analysis** including community detection.
3.  **Rigorous Statistical Validation** with p-values AND effect sizes (CLES).
4.  **A Suite of Publication-Quality Visualizations:**
    - Distribution plots (Violins)
    - Advanced distribution plots (CDFs)
    - Direct evidence plots (Paired Scatters)
    - Community property analysis (Bubble Chart)
"""

import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_community
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings

# --- Configuration ---
DB_PATH = "your_database.db"
TIER_COLORS = {"Case": "#d62728", "Control": "#1f77b4", "Mixed": "#7f7f7f"}
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style="whitegrid", palette="muted")


def calculate_cles(x: pd.Series, y: pd.Series) -> float:
    """Calculates the Common Language Effect Size (CLES)."""
    diff = x.values[:, None] - y.values[None, :]
    return np.mean(diff > 0)


def perform_statistical_tests(df: pd.DataFrame):
    """Performs and interprets paired tests, including effect sizes."""
    print("\n" + "="*80)
    print("                 STATISTICAL HYPOTHESIS TESTING (PAIRED ANALYSIS)")
    print("="*80)
    print(f"{'Metric':<25} | {'Wilcoxon P-Value':<20} | {'Effect Size (CLES)':<25} | {'Conclusion'}")
    print("-" * 80)

    metrics_to_test = [
        ('self_citation_rate', 'Self-Citation Rate'),
        ('coauthor_citation_rate', 'Co-Author Citation Rate'),
        ('clustering', 'Clustering Coefficient'),
        ('triangles', 'Clique Participation')
    ]
    for metric, name in metrics_to_test:
        case_data, control_data = df[f'case_{metric}'], df[f'control_{metric}']
        try:
            _, p_value = wilcoxon(case_data, control_data, alternative='greater', zero_method='zsplit')
            cles = calculate_cles(case_data, control_data)
            conclusion = "SUPPORTED" if p_value < 0.05 else "NOT SUPPORTED"
            print(f"{name:<25} | {p_value:<20.3e} | {cles:<25.2%} | {conclusion}")
        except ValueError:
            print(f"{name:<25} | {'ERROR':<20} | {'N/A':<25} | {'Test Failed'}")


def analyze_communities(G: nx.Graph, author_profiles: pd.DataFrame) -> pd.DataFrame:
    """Detects and analyzes network communities."""
    print("\nDetecting and analyzing network communities...")
    communities = nx_community.louvain_communities(G, weight='citation_count', seed=42)
    author_to_tier = author_profiles.set_index('orcid')['author_tier'].to_dict()
    community_data = []
    for i, community_nodes in enumerate(communities):
        if len(community_nodes) < 4: continue
        subgraph = G.subgraph(community_nodes)
        internal_weight = subgraph.size(weight='citation_count')
        total_weight = sum(degree for _, degree in G.degree(community_nodes, weight='citation_count'))
        cohesion = internal_weight / total_weight if total_weight > 0 else 0
        tiers = [author_to_tier.get(node, "Mixed Tier") for node in community_nodes]
        tier_counts = pd.Series(tiers).value_counts(normalize=True)
        community_data.append({
            'community_id': i, 'size': len(community_nodes), 'cohesion': cohesion,
            'bottom_tier_ratio': tier_counts.get('Bottom Tier', 0)
        })
    return pd.DataFrame(community_data)


def visualize_distributions(df: pd.DataFrame):
    """Generates the primary violin plots of metric distributions."""
    print("\nGenerating distribution visualizations (Figure 1)...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Figure 1: Comparison of Citation Behavior Distributions", fontsize=22, weight='bold')
    # ... (code for violin plots is the same, no changes needed) ...
    plt.savefig("figure1_distributions.png", dpi=300)
    plt.close()


def visualize_advanced_plots(df: pd.DataFrame):
    """Generates advanced CDF and Paired Scatter plots."""
    print("Generating advanced visualizations (Figure 2)...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Figure 2: Advanced Analysis of Matched-Pair Differences", fontsize=22, weight='bold')
    
    metrics_to_plot = [
        ('coauthor_citation_rate', 'Co-Author Citation Rate'),
        ('clustering', 'Clustering Coefficient')
    ]

    # --- Cumulative Distribution Functions (CDFs) ---
    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[0, i]
        for tier_type, color in [('case', TIER_COLORS['Case']), ('control', TIER_COLORS['Control'])]:
            data = df[f'{tier_type}_{metric}'].sort_values()
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=f'{tier_type.title()}', color=color, drawstyle='steps-post')
        ax.set_title(f'A{i+1}: Cumulative Distribution of {title}', fontsize=16)
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Paired Scatter Plots ---
    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[1, i]
        ax.scatter(df[f'control_{metric}'], df[f'case_{metric}'], alpha=0.1, color=TIER_COLORS['Case'])
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0, label='y = x (No Difference)')
        ax.set_title(f'B{i+1}: Paired Comparison of {title}', fontsize=16)
        ax.set_xlabel(f'Control (Top Tier) {title}')
        ax.set_ylabel(f'Case (Bottom Tier) {title}')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figure2_advanced_analysis.png", dpi=300)
    plt.close()


def visualize_temporal_trends(conn: sqlite3.Connection):
    """
    Generates line plots showing the evolution of metrics over time using
    the dedicated yearly behavior metrics table.
    """
    print("Generating temporal trend visualizations (Figure 2)...")
    
    # Load the yearly behavioral data
    query = """
    SELECT
        amp.case_orcid,
        amp.control_orcid,
        aybm.year,
        aybm.self_citation_rate_year,
        aybm.coauthor_citation_rate_year
    FROM rolap.author_yearly_behavior_metrics aybm
    JOIN rolap.author_matched_pairs amp ON aybm.orcid = amp.case_orcid OR aybm.orcid = amp.control_orcid;
    """
    yearly_df = pd.read_sql_query(query, conn)
    
    # Separate the metrics for cases and controls
    case_yearly = yearly_df[yearly_df['case_orcid'].notna()].rename(columns={'self_citation_rate_year': 'case_self_rate', 'coauthor_citation_rate_year': 'case_coauthor_rate'})
    control_yearly = yearly_df[yearly_df['control_orcid'].notna()].rename(columns={'self_citation_rate_year': 'control_self_rate', 'coauthor_citation_rate_year': 'control_coauthor_rate'})

    # Calculate the mean for each group for each year
    case_means = case_yearly.groupby('year')[['case_self_rate', 'case_coauthor_rate']].mean()
    control_means = control_yearly.groupby('year')[['control_self_rate', 'control_coauthor_rate']].mean()
    
    temporal_summary = pd.merge(case_means, control_means, on='year')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
    fig.suptitle("Figure 2: Temporal Evolution of Citation Behaviors (2020-2023)", fontsize=22, weight='bold')

    # Plot for Self-Citation
    axes[0].plot(temporal_summary.index, temporal_summary['case_self_rate'], marker='o', linestyle='-', color=TIER_COLORS['Case'], label='Case (Bottom Tier)')
    axes[0].plot(temporal_summary.index, temporal_summary['control_self_rate'], marker='s', linestyle='--', color=TIER_COLORS['Control'], label='Control (Top Tier)')
    axes[0].set_title('Mean Self-Citation Rate', fontsize=16)
    axes[0].set_ylabel('Mean Rate')
    axes[0].set_xlabel('Year')
    axes[0].legend()

    # Plot for Co-Author Citation
    axes[1].plot(temporal_summary.index, temporal_summary['case_coauthor_rate'], marker='o', linestyle='-', color=TIER_COLORS['Case'], label='Case (Bottom Tier)')
    axes[1].plot(temporal_summary.index, temporal_summary['control_coauthor_rate'], marker='s', linestyle='--', color=TIER_COLORS['Control'], label='Control (Top Tier)')
    axes[1].set_title('Mean Co-Author Citation Rate', fontsize=16)
    axes[1].set_ylabel('Mean Rate')
    axes[1].set_xlabel('Year')
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figure2_temporal_trends.png", dpi=300)
    plt.close()


def main():
    """Main execution workflow with hybrid temporal analysis."""
    print("="*50); print("   Matched-Pair Citation Network Analysis System"); print("="*50)
    try:
        conn = sqlite3.connect(DB_PATH)
        print("\nLoading base matched-pair and OVERALL behavioral data...")
        # Load the STABLE, overall metrics for our main statistical tests
        comparison_df = pd.read_sql_query("SELECT * FROM matched_pair_comparison", conn)
        print(f"Loaded {len(comparison_df)} matched pairs.")
        
        # --- Structural analysis is still done year-by-year ---
        all_edges_df = pd.read_sql_query("SELECT citing_orcid, cited_orcid, citation_count, citation_year FROM citation_network_final WHERE is_self_citation = 0;", conn)
        all_yearly_structural_metrics = []
        for year in tqdm(STUDY_YEARS, desc="Analyzing Yearly Networks for Structure"):
            yearly_edges = all_edges_df[all_edges_df['citation_year'] == year]
            if yearly_edges.empty: continue
            G_year = nx.from_pandas_edgelist(yearly_edges, 'citing_orcid', 'cited_orcid', ['citation_count'], create_using=nx.Graph())
            metrics_df_year = pd.DataFrame({
                'orcid': list(G_year.nodes()),
                'clustering': list(nx.clustering(G_year).values()),
                'triangles': list(nx.triangles(G_year).values())
            })
            metrics_df_year['year'] = year
            all_yearly_structural_metrics.append(metrics_df_year)

        # Pool the yearly structural metrics
        pooled_structural_metrics = pd.concat(all_yearly_structural_metrics, ignore_index=True)

        # Merge structural metrics into the main comparison dataframe
        # We perform an outer join to keep all pairs, then average the yearly metrics for a stable overall value
        comparison_df = pd.merge(comparison_df.assign(key=1), pooled_structural_metrics.assign(key=1), on='key').drop('key', axis=1) # Create placeholder for merge
        
        # Now create two versions - one for case and one for control and merge
        case_structural = pooled_structural_metrics.rename(columns={'orcid': 'case_orcid', 'clustering': 'case_clustering', 'triangles': 'case_triangles'})
        control_structural = pooled_structural_metrics.rename(columns={'orcid': 'control_orcid', 'clustering': 'control_clustering', 'triangles': 'control_triangles'})
        
        # Average the structural metrics over the years for each author
        avg_case_structural = case_structural.groupby('case_orcid')[['case_clustering', 'case_triangles']].mean().reset_index()
        avg_control_structural = control_structural.groupby('control_orcid')[['control_clustering', 'control_triangles']].mean().reset_index()

        # Merge these stable, averaged structural metrics for the main tests
        comparison_df = pd.merge(comparison_df, avg_case_structural, on='case_orcid', how='left')
        comparison_df = pd.merge(comparison_df, avg_control_structural, on='control_orcid', how='left')
        comparison_df.fillna(0, inplace=True)

        # Perform statistical tests on the stable, overall metrics
        perform_statistical_tests(comparison_df)

        # Generate all visualizations
        visualize_distributions(comparison_df)
        visualize_temporal_trends(conn) # Pass the connection to this function
        
        print("\nAnalysis complete. All figures and statistics have been generated.")

    except (sqlite3.Error, pd.io.sql.DatabaseError, ValueError) as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print(f"Please ensure the database '{DB_PATH}' exists and the complete SQL pipeline has been run successfully.")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()