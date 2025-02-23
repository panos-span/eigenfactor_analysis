import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
import statistics
import traceback
import random
import csv   # Don't forget to import csv if you're saving CSVs

# Paths to your databases
ROLAP_DATABASE = "rolap.db"
IMPACT_DATABASE = "impact.db"

###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def get_doi_for_work_id(cursor, work_id):
    """
    Return the DOI for a given work_id from the 'works' table.
    """
    cursor.execute("SELECT doi FROM works WHERE id = ?", (work_id,))
    row = cursor.fetchone()
    return row[0] if row else None

def get_reference_weight(cursor, citing_work_id, cited_work_id):
    """
    Calculate how many times 'citing_work_id' references 'cited_work_id'.
    For demonstration, we simply count records in 'work_references'. 
    Adjust if your schema stores citations differently (e.g., a 'count' column).
    """
    cited_doi = get_doi_for_work_id(cursor, cited_work_id)
    if not cited_doi:
        return 0
    # Count how many references from 'citing_work_id' -> 'cited_doi'
    cursor.execute(
        """SELECT COUNT(*) FROM work_references
           WHERE work_id = ? AND doi = ?""",
        (citing_work_id, cited_doi)
    )
    row = cursor.fetchone()
    return row[0] if row else 0

def add_weighted_citation_edges(cursor, graph, current_id, depth):
    """
    Recursively add edges with weight for outgoing and incoming references,
    up to 'depth' hops.
    """
    if depth < 0:
        return

    # -- Outgoing references --
    cursor.execute(
        """SELECT w.id
           FROM work_references wr
           JOIN works w ON wr.doi = w.doi
           WHERE wr.work_id = ?""",
        (current_id,)
    )
    out_rows = cursor.fetchall()
    for (out_id,) in out_rows:
        wt = get_reference_weight(cursor, current_id, out_id)
        if wt > 0:
            graph.add_edge(current_id, out_id, weight=wt)
        add_weighted_citation_edges(cursor, graph, out_id, depth - 1)

    # -- Incoming references --
    current_doi = get_doi_for_work_id(cursor, current_id)
    if current_doi:
        cursor.execute(
            "SELECT wr.work_id FROM work_references wr WHERE wr.doi = ?",
            (current_doi,)
        )
        in_rows = cursor.fetchall()
        for (in_id,) in in_rows:
            wt = get_reference_weight(cursor, in_id, current_id)
            if wt > 0:
                graph.add_edge(in_id, current_id, weight=wt)
            add_weighted_citation_edges(cursor, graph, in_id, depth - 1)

def build_local_citation_graph(db_connection, work_id, depth=1):
    """
    Build a weighted, undirected citation graph for a single 'work_id', 
    up to the specified 'depth'.
    """
    G = nx.Graph()
    cursor = db_connection.cursor()
    add_weighted_citation_edges(cursor, G, work_id, depth)
    cursor.close()
    return G

def compute_graph_metrics(graph):
    """
    Compute some basic metrics of interest.
    - density
    - average clustering
    - average shortest path length (largest connected component)
    - mean edge weight
    - number of cliques and size of largest clique
    """
    metrics = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": 0,
        "avg_clustering": 0,
        "avg_shortest_path_length": 0,
        "mean_edge_weight": 0,
        "num_cliques": 0,
        "max_clique_size": 0
    }
    n = metrics["n_nodes"]

    if n <= 1:
        return metrics

    # Graph density
    metrics["density"] = nx.density(graph)

    # Average clustering
    metrics["avg_clustering"] = nx.average_clustering(graph)

    # Mean edge weight
    if graph.number_of_edges() > 0:
        wts = [d["weight"] for _, _, d in graph.edges(data=True)]
        metrics["mean_edge_weight"] = statistics.mean(wts)

    # Average shortest path length (largest connected component)
    largest_cc = max(nx.connected_components(graph), key=len)
    subg = graph.subgraph(largest_cc).copy()
    if subg.number_of_nodes() > 1:
        try:
            metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(subg)
        except:
            metrics["avg_shortest_path_length"] = None

    # Find cliques
    cliques = list(nx.find_cliques(graph))
    metrics["num_cliques"] = len(cliques)
    if cliques:
        metrics["max_clique_size"] = max(len(c) for c in cliques)

    return metrics

###############################################################################
#                     OUTLIER DETECTION (EXAMPLE)
###############################################################################

def detect_outliers_on_density(metrics_list, threshold_std=2.0):
    """
    A simple outlier detection approach on 'density' metric. 
    We take a list of dicts with at least:
        {
          "work_id": ...,
          "density": ...
        }
    and mark those with density more than 'threshold_std' standard 
    deviations above the mean as outliers.
    Returns a list of outlier items.
    """
    densities = [m["density"] for m in metrics_list]
    if len(densities) < 2:
        return []

    mean_density = statistics.mean(densities)
    stdev_density = statistics.pstdev(densities)
    if stdev_density == 0:
        return []

    threshold = mean_density + threshold_std * stdev_density
    return [m for m in metrics_list if m["density"] > threshold]

###############################################################################
#                     REPLICATE RANDOM MATCHING LOGIC
###############################################################################

def replicate_random_analysis(
    rolap_conn, 
    impact_conn, 
    sample_size=50, 
    depth=1,
    output_filename=None
):
    """
    1) Retrieve random samples from 'random_top_works' and 'random_other_works'.
    2) Build local citation graphs with weighted edges (depth=1).
    3) Compute metrics (density, cliques, etc.).
    4) Print or store results.
    5) Demonstrate an outlier detection approach.

    If 'output_filename' is provided, the final metrics will be saved to a CSV file.
    """
    cursor = rolap_conn.cursor()

    # A) Get top works (random sample)
    cursor.execute("SELECT id, citations_number, subject FROM random_top_works")
    all_top_rows = cursor.fetchall()
    top_sample = random.sample(all_top_rows, min(sample_size, len(all_top_rows)))

    # B) Get other works (random sample)
    cursor.execute("SELECT id, citations_number, subject FROM random_other_works")
    all_other_rows = cursor.fetchall()
    other_sample = random.sample(all_other_rows, min(sample_size, len(all_other_rows)))

    cursor.close()

    # C) Build graphs, compute metrics
    results = []
    for (work_id, cites, subj) in top_sample:
        G = build_local_citation_graph(impact_conn, work_id, depth=depth)
        met = compute_graph_metrics(G)
        met["work_id"] = work_id
        met["citations"] = cites
        met["subject"] = subj
        met["category"] = "TOP"
        results.append(met)

    for (work_id, cites, subj) in other_sample:
        G = build_local_citation_graph(impact_conn, work_id, depth=depth)
        met = compute_graph_metrics(G)
        met["work_id"] = work_id
        met["citations"] = cites
        met["subject"] = subj
        met["category"] = "OTHER"
        results.append(met)

    # D) Print out basic comparison
    print("\n=== Random Matching (TOP vs OTHER) ===")
    for row in results:
        print(
            f"WorkID={row['work_id']}, Cat={row['category']}, "
            f"Subject={row['subject']}, Cites={row['citations']}, "
            f"Nodes={row['n_nodes']}, Edges={row['n_edges']}, "
            f"Density={row['density']:.3f}, Cluster={row['avg_clustering']:.3f}, "
            f"PathLen={row['avg_shortest_path_length']}, "
            f"MeanW={row['mean_edge_weight']:.2f}, Cliques={row['num_cliques']}, "
            f"MaxCliqueSize={row['max_clique_size']}"
        )

    # E) Example: detect outliers on density
    outliers = detect_outliers_on_density(results, threshold_std=2.0)
    print(f"\n[INFO] Found {len(outliers)} outlier(s) based on density (>= mean + 2*std):")
    for o in outliers:
        print(
            f" -> Outlier WorkID={o['work_id']} "
            f"(Density={o['density']:.3f}, Category={o['category']})"
        )

    # F) Optionally save to CSV if output_filename is provided
    if output_filename:
        import csv
        fieldnames = [
            "work_id",
            "category",
            "subject",
            "citations",
            "n_nodes",
            "n_edges",
            "density",
            "avg_clustering",
            "avg_shortest_path_length",
            "mean_edge_weight",
            "num_cliques",
            "max_clique_size"
        ]
        try:
            with open(output_filename, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    output_row = {f: row[f] for f in fieldnames}
                    writer.writerow(output_row)
            print(f"[INFO] Results saved to {output_filename}")
        except Exception as ex:
            print(f"[WARNING] Could not write to {output_filename}: {ex}")

    return results

def main():
    rolap_conn = None
    impact_conn = None
    try:
        # Use the correct variable names:
        rolap_conn = sqlite3.connect(ROLAP_DATABASE)
        impact_conn = sqlite3.connect(IMPACT_DATABASE)

        print("[INFO] Starting extended random matching analysis...")

        replicate_random_analysis(
            rolap_conn=rolap_conn,
            impact_conn=impact_conn,
            sample_size=50,
            depth=1,
            output_filename="citation_metrics_output.csv"
        )

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
    finally:
        # Safely close connections if they were successfully opened
        if rolap_conn is not None:
            rolap_conn.close()
        if impact_conn is not None:
            impact_conn.close()

if __name__ == "__main__":
    main()
