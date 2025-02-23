import networkx as nx
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import sys
import numpy as np
import traceback

GRAPH_DATABASE_PATH = "impact.db"
ROLAP_DATABASE_PATH = "rolap.db"


def doi_workid(cursor, doi):
    """Return the work-id for the specified DOI"""
    cursor.execute("SELECT id FROM works WHERE doi = ?", (doi,))
    res = cursor.fetchone()
    if res:
        (id,) = res
        return id
    return None


def find_citation_cliques(graph):
    """Find and return all cliques in the given graph"""
    cliques = list(nx.find_cliques(graph))
    return cliques


def workid_doi(cursor, id):
    """Return the DOI for the specified work-id"""
    cursor.execute("SELECT doi FROM works WHERE id = ?", (id,))
    res = cursor.fetchone()
    if res:
        (doi,) = res
        return doi
    return None


def add_citation_edges(connection, graph, start, depth):
    if depth == -1:
        return
    cursor = connection.cursor()

    # Outgoing references
    cursor.execute(
        """SELECT id FROM work_references
        INNER JOIN works ON work_references.doi = works.doi
        WHERE work_references.work_id = ?""",
        (start,),
    )
    outgoing_ids = cursor.fetchall()
    for (id,) in outgoing_ids:
        graph.add_edge(start, id)
        add_citation_edges(connection, graph, id, depth - 1)

    # Incoming references
    work_doi = workid_doi(cursor, start)
    if work_doi:
        cursor.execute("SELECT work_id FROM work_references WHERE doi = ?", (work_doi,))
        incoming_ids = cursor.fetchall()
        for (id,) in incoming_ids:
            graph.add_edge(start, id)
            add_citation_edges(connection, graph, id, depth - 1)

    cursor.close()


def citation_graph(connection, start):
    """Return a graph induced by incoming and outgoing citations of
    distance 2 for the specified work-id"""
    graph = nx.Graph()
    add_citation_edges(connection, graph, start, 1)
    return graph


#def get_all_subjects(rolap_cursor):
#    """Get all unique subjects from random_top_works table"""
#    rolap_cursor.execute("SELECT DISTINCT subject FROM random_top_works")
#    return [row[0] for row in rolap_cursor.fetchall()]
#
#def get_top_5_cliques_subgraph(graph, cliques):
#    """Get subgraph containing only the top 5 largest cliques"""
#    sorted_cliques = sorted(cliques, key=len, reverse=True)[:5]
#    nodes_in_top_cliques = set()
#    for clique in sorted_cliques:
#        nodes_in_top_cliques.update(clique)
#    return graph.subgraph(nodes_in_top_cliques), sorted_cliques
#
#def visualize_cliques_for_table(rolap_connection, graph_connection, table_name, subject, limit=20):
#    """Create visualization of top 5 largest citation cliques for a specific table and subject"""
#    rolap_cursor = rolap_connection.cursor()
#    
#    # Get work IDs for the subject with limit from ROLAP database
#    query = f"SELECT id FROM {table_name} WHERE subject = ? LIMIT ?"
#    rolap_cursor.execute(query, (subject, limit))
#    work_ids = [row[0] for row in rolap_cursor.fetchall()]
#    
#    # Create combined graph using Graph database connection
#    combined_graph = nx.Graph()
#    for work_id in work_ids:
#        work_graph = citation_graph(graph_connection, work_id)
#        combined_graph.add_edges_from(work_graph.edges())
#    
#    # Find all cliques and get subgraph of top 5
#    all_cliques = find_citation_cliques(combined_graph)
#    top_cliques_graph, top_5_cliques = get_top_5_cliques_subgraph(combined_graph, all_cliques)
#    
#    if len(top_5_cliques) == 0:
#        print(f"No cliques found for {table_name} - {subject}")
#        return None
#    
#    # Create figure with adjusted size and gridspec
#    fig = plt.figure(figsize=(15, 10))
#    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])  # Ratio between plot and legend
#    ax_plot = fig.add_subplot(gs[0])
#    ax_legend = fig.add_subplot(gs[1])
#    ax_legend.axis('off')  # Hide legend axes
#    
#    # Draw basic graph structure
#    pos = nx.spring_layout(top_cliques_graph, k=1.5)
#    nx.draw(top_cliques_graph, pos, 
#           node_color='lightgray',
#           node_size=300,
#           alpha=0.3,
#           width=0.5,
#           ax=ax_plot)
#    
#    # Draw cliques with different colors
#    colors = plt.cm.rainbow(np.linspace(0, 1, len(top_5_cliques)))
#    legend_elements = []
#    
#    # Draw nodes and edges for each clique
#    for idx, clique in enumerate(top_5_cliques):
#        # Draw nodes for this clique
#        nx.draw_networkx_nodes(top_cliques_graph, pos,
#                             nodelist=clique,
#                             node_color=[colors[idx]],
#                             node_size=500,
#                             alpha=0.6,
#                             ax=ax_plot)
#        
#        # Draw edges within the clique
#        edge_list = [(u, v) for u in clique for v in clique if u < v]
#        nx.draw_networkx_edges(top_cliques_graph, pos,
#                             edgelist=edge_list,
#                             edge_color=colors[idx],
#                             width=2,
#                             alpha=0.6,
#                             ax=ax_plot)
#        
#        # Add to legend
#        legend_elements.append(mpatches.Patch(color=colors[idx],
#                                            label=f'Clique {idx+1}\nSize: {len(clique)}'))
#    
#    # Add node labels
#    labels = {node: str(node) for node in top_cliques_graph.nodes()}
#    nx.draw_networkx_labels(top_cliques_graph, pos, labels, font_size=8, ax=ax_plot)
#    
#    table_type = "Top Works" if table_name == "random_top_works" else "Other Works"
#    ax_plot.set_title(f'Top 5 Largest Citation Cliques in {table_type}\n'
#                     f'Subject: {subject}\n'
#                     f'Total works analyzed: {len(work_ids)}\n'
#                     f'Number of cliques found: {len(all_cliques)}')
#    
#    # Add legend to the separate axis
#    if legend_elements:
#        ax_legend.legend(handles=legend_elements,
#                        title="Clique Sizes",
#                        loc='center')
#    
#    # Adjust layout
#    plt.subplots_adjust(wspace=0.2, right=0.98, left=0.02)
#    return fig
#
#def main():
#    try:
#        # Connect to both databases
#        rolap_connection = sqlite3.connect(ROLAP_DATABASE_PATH)
#        graph_connection = sqlite3.connect(GRAPH_DATABASE_PATH)
#        
#        rolap_cursor = rolap_connection.cursor()
#        
#        # Get all subjects
#        subjects = get_all_subjects(rolap_cursor)
#        print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
#        
#        for subject in subjects:
#            print(f"\nProcessing subject: {subject}")
#            
#            # Create visualization for random_top_works
#            fig = visualize_cliques_for_table(rolap_connection, graph_connection, 
#                                            "random_top_works", subject)
#            if fig:
#                fig.savefig(f'top5_cliques_top_works_{subject.lower().replace(" ", "_")}.png',
#                           bbox_inches='tight', dpi=300)
#            plt.close(fig)
#            
#            # Create visualization for random_other_works
#            fig = visualize_cliques_for_table(rolap_connection, graph_connection, 
#                                            "random_other_works", subject)
#            if fig:
#                fig.savefig(f'top5_cliques_other_works_{subject.lower().replace(" ", "_")}.png',
#                           bbox_inches='tight', dpi=300)
#            plt.close(fig)
#            
#            print(f"Created visualizations for {subject}")
#            
#    except Exception as e:
#        print(f"Error: {e}")
#        traceback.print_exc()
#    finally:
#        rolap_cursor.close()
#        rolap_connection.close()
#        graph_connection.close()
#
#if __name__ == "__main__":
#    main()

def get_all_subjects(rolap_cursor):
    """Get all unique subjects from random_top_works table"""
    rolap_cursor.execute("SELECT DISTINCT subject FROM random_top_works")
    return [row[0] for row in rolap_cursor.fetchall()]

def get_top_3_cliques_subgraph(graph, cliques):
    """Get subgraph containing only the top 3 largest cliques"""
    sorted_cliques = sorted(cliques, key=len, reverse=True)[:3]
    nodes_in_top_cliques = set()
    for clique in sorted_cliques:
        nodes_in_top_cliques.update(clique)
    return graph.subgraph(nodes_in_top_cliques), sorted_cliques

def create_comprehensive_visualization(rolap_connection, graph_connection, table_name, subjects, limit=20):
    """Create a single visualization showing top 3 cliques for each subject"""
    num_subjects = len(subjects)
    if num_subjects == 0:
        return None
        
    # Calculate grid dimensions
    num_cols = min(3, num_subjects)
    num_rows = (num_subjects + num_cols - 1) // num_cols
    
    # Create figure with adjusted size and gridspec
    fig = plt.figure(figsize=(20, 6 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0.4, wspace=0.3)
    
    # Color map for all cliques across all subjects
    color_map = plt.cm.rainbow(np.linspace(0, 1, 3))  # 3 colors for top 3 cliques
    
    for idx, subject in enumerate(subjects):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get work IDs for the subject
        rolap_cursor = rolap_connection.cursor()
        query = f"SELECT id FROM {table_name} WHERE subject = ? LIMIT ?"
        rolap_cursor.execute(query, (subject, limit))
        work_ids = [row[0] for row in rolap_cursor.fetchall()]
        
        # Create combined graph
        combined_graph = nx.Graph()
        for work_id in work_ids:
            work_graph = citation_graph(graph_connection, work_id)
            combined_graph.add_edges_from(work_graph.edges())
        
        # Find top 3 cliques
        all_cliques = find_citation_cliques(combined_graph)
        top_cliques_graph, top_3_cliques = get_top_3_cliques_subgraph(combined_graph, all_cliques)
        
        if len(top_3_cliques) > 0:
            # Draw the subgraph
            pos = nx.spring_layout(top_cliques_graph, k=1.5)
            
            # Draw basic graph structure
            nx.draw(top_cliques_graph, pos,
                   node_color='lightgray',
                   node_size=300,
                   alpha=0.3,
                   width=0.5,
                   ax=ax)
            
            # Draw cliques
            legend_elements = []
            for cidx, clique in enumerate(top_3_cliques):
                # Draw nodes
                nx.draw_networkx_nodes(top_cliques_graph, pos,
                                     nodelist=clique,
                                     node_color=[color_map[cidx]],
                                     node_size=500,
                                     alpha=0.6,
                                     ax=ax)
                
                # Draw edges
                edge_list = [(u, v) for u in clique for v in clique if u < v]
                nx.draw_networkx_edges(top_cliques_graph, pos,
                                     edgelist=edge_list,
                                     edge_color=color_map[cidx],
                                     width=2,
                                     alpha=0.6,
                                     ax=ax)
                
                legend_elements.append(mpatches.Patch(color=color_map[cidx],
                                                    label=f'Clique {cidx+1}\nSize: {len(clique)}'))
            
            # Add node labels
            labels = {node: str(node) for node in top_cliques_graph.nodes()}
            nx.draw_networkx_labels(top_cliques_graph, pos, labels, font_size=8, ax=ax)
            
            # Add legend
            ax.legend(handles=legend_elements,
                     title="Clique Sizes",
                     loc='center left',
                     bbox_to_anchor=(1, 0.5))
        
        ax.set_title(f'Subject: {subject}\nWorks analyzed: {len(work_ids)}\nCliques found: {len(all_cliques)}')
    
    table_type = "Top Works" if table_name == "random_top_works" else "Other Works"
    fig.suptitle(f'Top 3 Largest Citation Cliques per Subject - {table_type}', 
                 fontsize=16, y=0.95)
    
    return fig

def main():
    try:
        # Connect to both databases
        rolap_connection = sqlite3.connect(ROLAP_DATABASE_PATH)
        graph_connection = sqlite3.connect(GRAPH_DATABASE_PATH)
        
        # Get all subjects
        rolap_cursor = rolap_connection.cursor()
        subjects = get_all_subjects(rolap_cursor)
        print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
        
        # Create visualization for random_top_works
        print("Creating visualization for top works...")
        fig_top = create_comprehensive_visualization(
            rolap_connection, graph_connection, "random_top_works", subjects)
        if fig_top:
            fig_top.savefig('top3_cliques_all_subjects_top_works.png',
                           bbox_inches='tight', dpi=300)
        plt.close(fig_top)
        
        # Create visualization for random_other_works
        print("Creating visualization for other works...")
        fig_other = create_comprehensive_visualization(
            rolap_connection, graph_connection, "random_other_works", subjects)
        if fig_other:
            fig_other.savefig('top3_cliques_all_subjects_other_works.png',
                           bbox_inches='tight', dpi=300)
        plt.close(fig_other)
        
        print("Completed all visualizations")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        rolap_cursor.close()
        rolap_connection.close()
        graph_connection.close()

if __name__ == "__main__":
    main()
