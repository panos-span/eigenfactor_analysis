import sqlite3
import networkx as nx
import csv
import traceback
import statistics
from collections import deque
import matplotlib.pyplot as plt

# Database paths (adjust to your environment)
ROLAP_DB = "rolap.db"
IMPACT_DB = "impact.db"

# BFS parameters
MAX_BFS_NODES = 2000  # Cap the BFS expansion to prevent explosion
DEFAULT_DEPTH = 2     # Use depth 2 BFS to capture more interconnections

##############################################################################
# 1) Load "bad" authors from orcid_h5_bottom
##############################################################################
def load_bad_authors(rolap_conn):
    """
    Return a set of ORCIDs that are considered 'bad' (from orcid_h5_bottom).
    """
    bad_set = set()
    cur = rolap_conn.cursor()
    cur.execute("SELECT orcid FROM orcid_h5_bottom")
    for (orc,) in cur.fetchall():
        bad_set.add(orc)
    cur.close()
    return bad_set

##############################################################################
# 2) Helper Functions for DB Queries
##############################################################################
def get_works_for_author(cursor_rolap, orcid):
    sql = "SELECT id FROM works_orcid WHERE orcid = ?"
    cursor_rolap.execute(sql, (orcid,))
    return [row[0] for row in cursor_rolap.fetchall()]

def get_cited_work_ids(cursor_impact, work_id):
    sql = """
        SELECT w.id
        FROM work_references wr
        JOIN works w ON wr.doi = w.doi
        WHERE wr.work_id = ?
    """
    cursor_impact.execute(sql, (work_id,))
    return [r[0] for r in cursor_impact.fetchall()]

def get_authors_for_work(cursor_rolap, work_id):
    sql = "SELECT orcid FROM works_orcid WHERE id = ?"
    cursor_rolap.execute(sql, (work_id,))
    return [r[0] for r in cursor_rolap.fetchall() if r[0] is not None]

##############################################################################
# 3) BFS with Node Cap and Depth=2
##############################################################################
def build_local_author_subgraph(rolap_conn, impact_conn, start_author, depth=DEFAULT_DEPTH, max_nodes_in_bfs=MAX_BFS_NODES):
    """
    Build a directed graph for 'start_author' up to 'depth' levels.
    Expansion stops if more than 'max_nodes_in_bfs' authors are visited.
    """
    G = nx.DiGraph()
    visited = set([start_author])
    queue = deque([(start_author, 0)])

    rolap_cursor = rolap_conn.cursor()
    impact_cursor = impact_conn.cursor()

    while queue:
        current_author, dist = queue.popleft()
        if dist >= depth:
            continue

        if len(visited) >= max_nodes_in_bfs:
            break

        works = get_works_for_author(rolap_cursor, current_author)
        for wid in works:
            cited_ids = get_cited_work_ids(impact_cursor, wid)
            for cwid in cited_ids:
                cited_authors = get_authors_for_work(rolap_cursor, cwid)
                for ca in cited_authors:
                    if G.has_edge(current_author, ca):
                        G[current_author][ca]["weight"] += 1
                    else:
                        G.add_edge(current_author, ca, weight=1)
                    if ca not in visited:
                        visited.add(ca)
                        queue.append((ca, dist + 1))
                        if len(visited) >= max_nodes_in_bfs:
                            break
                if len(visited) >= max_nodes_in_bfs:
                    break
            if len(visited) >= max_nodes_in_bfs:
                break

    rolap_cursor.close()
    impact_cursor.close()
    return G

##############################################################################
# 4) Additional Clique Metrics
##############################################################################
def compute_clique_metrics(digraph, bad_authors, min_size=3):
    """
    Convert the directed graph to an undirected graph and find maximal cliques.
    For each clique (of size >= min_size), compute:
      - clique density (2*|E|/(n*(n-1)))
      - average edge weight within the clique
      - fraction of nodes that are bad.
    Returns a list of dicts, one per clique.
    """
    UG = nx.Graph()
    for u, v, data in digraph.edges(data=True):
        w = data["weight"]
        existing = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing + w)

    cliques = list(nx.find_cliques(UG))
    clique_metrics = []
    for clique in cliques:
        if len(clique) < min_size:
            continue
        subg = UG.subgraph(clique)
        # Density: 2*edges/(n*(n-1))
        n = subg.number_of_nodes()
        e = subg.number_of_edges()
        density = (2*e)/(n*(n-1)) if n > 1 else 0
        # Average edge weight:
        weights = [data["weight"] for _,_,data in subg.edges(data=True)]
        avg_w = statistics.mean(weights) if weights else 0
        # Fraction of bad nodes:
        bad_count = sum(1 for node in clique if node in bad_authors)
        frac_bad = bad_count / n
        clique_metrics.append({
            "clique_size": n,
            "density": density,
            "avg_edge_weight": avg_w,
            "fraction_bad": frac_bad
        })
    return clique_metrics

##############################################################################
# 5) Compute Local Metrics (with Additional Metrics)
##############################################################################
def compute_local_metrics(digraph, bad_authors):
    """
    Compute basic metrics on the undirected version plus:
      - Reciprocal ratio: fraction of edges that are bidirectional.
      - bad_bad_fraction: fraction of edges where both endpoints are bad.
      - largest strongly connected component size.
      - Also compute aggregated clique metrics: e.g. maximum clique density among bad cliques.
    """
    metrics = {
        "n_nodes": digraph.number_of_nodes(),
        "n_edges": digraph.number_of_edges(),
        "density": 0.0,
        "avg_clustering": 0.0,
        "avg_shortest_path_length": None,
        "num_cliques": 0,
        "max_clique_size": 0,
        "reciprocal_ratio": 0.0,
        "bad_bad_fraction": 0.0,
        "largest_scc_size": 0,
        "max_bad_clique_density": 0.0,  # new metric
        "avg_bad_clique_fraction": 0.0  # new metric: average fraction of bad nodes in bad cliques
    }
    n = metrics["n_nodes"]
    e = metrics["n_edges"]
    if n <= 1:
        return metrics

    # Build undirected graph for standard metrics
    UG = nx.Graph()
    for u, v, data in digraph.edges(data=True):
        w = data["weight"]
        existing = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing+w)

    metrics["density"] = nx.density(UG)
    if n > 1:
        metrics["avg_clustering"] = nx.average_clustering(UG)
        try:
            largest_cc = max(nx.connected_components(UG), key=len)
            subg = UG.subgraph(largest_cc).copy()
            if subg.number_of_nodes() > 1:
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(subg)
        except:
            metrics["avg_shortest_path_length"] = None

    try:
        cliques = list(nx.find_cliques(UG))
        metrics["num_cliques"] = len(cliques)
        if cliques:
            metrics["max_clique_size"] = max(len(c) for c in cliques)
    except:
        pass

    # Reciprocal ratio (directed)
    if e > 0:
        rec_edges = sum(1 for u, v in digraph.edges() if digraph.has_edge(v, u))
        metrics["reciprocal_ratio"] = rec_edges / e

    # bad_bad_fraction
    if e > 0:
        bb = sum(1 for u, v in digraph.edges() if u in bad_authors and v in bad_authors)
        metrics["bad_bad_fraction"] = bb / e

    # Largest strongly connected component size (directed)
    try:
        sccs = nx.strongly_connected_components(digraph)
        metrics["largest_scc_size"] = max(len(scc) for scc in sccs)
    except:
        metrics["largest_scc_size"] = 0

    # Compute clique-level metrics for "bad cliques"
    clique_mets = compute_clique_metrics(digraph, bad_authors, min_size=3)
    bad_clique_densities = [cm["density"] for cm in clique_mets if cm["fraction_bad"] >= 0.8]
    bad_clique_fracs = [cm["fraction_bad"] for cm in clique_mets if cm["fraction_bad"] >= 0.8]
    if bad_clique_densities:
        metrics["max_bad_clique_density"] = max(bad_clique_densities)
        metrics["avg_bad_clique_fraction"] = statistics.mean(bad_clique_fracs)
    else:
        metrics["max_bad_clique_density"] = 0.0
        metrics["avg_bad_clique_fraction"] = 0.0

    return metrics

##############################################################################
# 6) Visualization (Limit to Top 100 Nodes, Label Top 10)
##############################################################################
def visualize_subgraph(digraph, output_filename, title="Local Subgraph", max_nodes=100, label_top=10):
    # Convert to undirected
    UG = nx.Graph()
    for u, v, data in digraph.edges(data=True):
        w = data["weight"]
        existing = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing + w)
    
    # Limit to top 'max_nodes' by degree
    if UG.number_of_nodes() > max_nodes:
        deg_sorted = sorted(UG.degree(), key=lambda x: x[1], reverse=True)
        keep_nodes = set(n for n, d in deg_sorted[:max_nodes])
        UG = UG.subgraph(keep_nodes).copy()
    
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(UG, k=0.3, seed=42)
    nx.draw_networkx_edges(UG, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_nodes(UG, pos, node_color="lightblue", node_size=200, alpha=0.9)
    
    # Label only top 'label_top' nodes by degree
    deg_sorted = sorted(UG.degree(), key=lambda x: x[1], reverse=True)
    top_labels = set(n for n, d in deg_sorted[:label_top])
    labels = {n: str(n) for n in top_labels}
    nx.draw_networkx_labels(UG, pos, labels=labels, font_size=7)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()

##############################################################################
# 7) MAIN
##############################################################################
def main():
    try:
        rolap_conn = sqlite3.connect(ROLAP_DB)
        impact_conn = sqlite3.connect(IMPACT_DB)

        # Load bad authors from ROLAP
        bad_authors = load_bad_authors(rolap_conn)
        print(f"[INFO] Loaded {len(bad_authors)} bad authors from orcid_h5_bottom.")

        # Select top 5 pairs from matched_authors_with_counts (using total works)
        cur = rolap_conn.cursor()
        sql_top5 = """
            SELECT bottom_orcid, random_orcid,
                   (COALESCE(bottom_n_works,0) + COALESCE(random_n_works,0)) AS total_works
            FROM matched_authors_with_counts
            ORDER BY total_works DESC
            LIMIT 5
        """
        cur.execute(sql_top5)
        pairs = cur.fetchall()
        cur.close()
        print(f"[INFO] Fetched top {len(pairs)} pairs from matched_authors_with_counts by total_works.")

        results = []
        pair_idx = 0

        # Process only the top 5 pairs (10 BFS calls)
        for (bot_orc, rnd_orc, sum_works) in pairs:
            pair_idx += 1
            print(f"\n[INFO] Pair#{pair_idx}: bottom={bot_orc}, random={rnd_orc}, total_works={sum_works}")

            for (author, cat) in [(bot_orc, "BOTTOM"), (rnd_orc, "RANDOM")]:
                digraph = build_local_author_subgraph(
                    rolap_conn, impact_conn, start_author=author, depth=DEFAULT_DEPTH, max_nodes_in_bfs=MAX_BFS_NODES
                )
                m = compute_local_metrics(digraph, bad_authors)
                results.append({
                    "pair_idx": pair_idx,
                    "author": author,
                    "category": cat,
                    "metrics": m
                })

        # Sort results by a metric of interest (e.g., bad_bad_fraction)
        results.sort(key=lambda x: x["metrics"]["bad_bad_fraction"], reverse=True)

        # Visualize top 2 subgraphs by "bad_bad_fraction"
        top_2 = results[:2]
        for i, row in enumerate(top_2, start=1):
            author = row["author"]
            cat = row["category"]
            subg = build_local_author_subgraph(
                rolap_conn, impact_conn, start_author=author, depth=DEFAULT_DEPTH, max_nodes_in_bfs=MAX_BFS_NODES
            )
            m = row["metrics"]
            out_fn = f"pair{row['pair_idx']}_{cat}_{author}_plot{i}.png"
            title_str = (f"Pair#{row['pair_idx']} {cat}={author}\n"
                         f"density={m['density']:.3f}, bad->bad={m['bad_bad_fraction']:.3f}, "
                         f"reciprocal={m['reciprocal_ratio']:.3f}, largest_scc={m['largest_scc_size']}")
            visualize_subgraph(subg, out_fn, title=title_str, max_nodes=100, label_top=10)
            print(f"[INFO] Created plot: {out_fn}")

        # Save all metrics to CSV
        with open("analysis_depth2_withcap.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "pair_idx", "author", "category",
                "n_nodes", "n_edges", "density", 
                "avg_clustering", "avg_shortest_path_length",
                "num_cliques", "max_clique_size",
                "reciprocal_ratio", "bad_bad_fraction", "largest_scc_size"
            ]
            writer.writerow(headers)
            for row in results:
                m = row["metrics"]
                writer.writerow([
                    row["pair_idx"],
                    row["author"],
                    row["category"],
                    m["n_nodes"],
                    m["n_edges"],
                    f"{m['density']:.3f}",
                    f"{m['avg_clustering']:.3f}",
                    m["avg_shortest_path_length"] if m["avg_shortest_path_length"] else "",
                    m["num_cliques"],
                    m["max_clique_size"],
                    f"{m['reciprocal_ratio']:.3f}",
                    f"{m['bad_bad_fraction']:.3f}",
                    m["largest_scc_size"]
                ])
        print("[INFO] Wrote analysis_depth2_withcap.csv with new metrics.")

        rolap_conn.close()
        impact_conn.close()

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
