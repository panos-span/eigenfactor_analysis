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
MAX_BFS_NODES = 2000  # Node cap during BFS
DEFAULT_DEPTH = 2     # BFS depth

##############################################################################
# 1) Load "bad authors" from orcid_h5_bottom
##############################################################################

def load_bad_authors(rolap_conn):
    """
    Return a set of ORCIDs that are considered 'bad' (e.g. from orcid_h5_bottom).
    """
    bad_set = set()
    cur = rolap_conn.cursor()
    cur.execute("SELECT orcid FROM orcid_h5_bottom")
    rows = cur.fetchall()
    for (orc,) in rows:
        bad_set.add(orc)
    cur.close()
    return bad_set

##############################################################################
# HELPER FUNCTIONS FOR DB QUERIES
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
# 2) BFS with Node Cap + Depth=2
##############################################################################

def build_local_author_subgraph(
    rolap_conn, 
    impact_conn, 
    start_author, 
    depth=DEFAULT_DEPTH, 
    max_nodes_in_bfs=MAX_BFS_NODES
):
    """
    Build a directed graph (NetworkX DiGraph) up to 'depth' BFS levels 
    from 'start_author'. Node expansion stops if visited_authors 
    exceeds 'max_nodes_in_bfs'.
    """
    G = nx.DiGraph()
    visited_authors = set([start_author])
    queue = deque([(start_author, 0)])

    rolap_cursor = rolap_conn.cursor()
    impact_cursor = impact_conn.cursor()

    while queue:
        current_author, dist = queue.popleft()
        if dist >= depth:
            continue

        # BFS node cap check
        if len(visited_authors) >= max_nodes_in_bfs:
            # We hit our cap, stop expanding further
            break

        # 1) fetch all works by current_author
        works = get_works_for_author(rolap_cursor, current_author)
        for w in works:
            # 2) get cited works from IMPACT
            cited_ids = get_cited_work_ids(impact_cursor, w)
            for cwid in cited_ids:
                # 3) get authors of these cited works
                cited_authors = get_authors_for_work(rolap_cursor, cwid)
                for ca in cited_authors:
                    # add edge current_author -> ca
                    if G.has_edge(current_author, ca):
                        G[current_author][ca]["weight"] += 1
                    else:
                        G.add_edge(current_author, ca, weight=1)

                    if ca not in visited_authors:
                        visited_authors.add(ca)
                        queue.append((ca, dist + 1))

                        # BFS node cap check again
                        if len(visited_authors) >= max_nodes_in_bfs:
                            break
                if len(visited_authors) >= max_nodes_in_bfs:
                    break
            if len(visited_authors) >= max_nodes_in_bfs:
                break

    rolap_cursor.close()
    impact_cursor.close()
    return G

##############################################################################
# 3) Additional Metrics
##############################################################################

def count_reciprocal_edges(digraph):
    """
    Return the number of edges that are bidirectional: 
    i.e., for (u->v), we also have (v->u).
    """
    count = 0
    for u, v in digraph.edges():
        if digraph.has_edge(v, u):
            count += 1
    return count

def count_bad_bad_edges(digraph, bad_authors):
    """
    Return how many edges connect 'bad' authors to 'bad' authors.
    """
    count = 0
    for u, v in digraph.edges():
        if u in bad_authors and v in bad_authors:
            count += 1
    return count

def largest_strongly_connected_component_size(digraph):
    """
    Return the size (number of nodes) of the largest strongly connected component.
    """
    if digraph.number_of_nodes() == 0:
        return 0
    sccs = nx.strongly_connected_components(digraph)
    return max(len(scc) for scc in sccs)

##############################################################################
# 4) Metrics Calculation
##############################################################################

def compute_local_metrics(digraph, bad_authors):
    """
    1) basic node/edge/density/clustering on the undirected version
    2) reciprocal ratio
    3) fraction of edges that are bad->bad
    4) largest strongly connected component (directed)
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
        "largest_scc_size": 0
    }

    n = metrics["n_nodes"]
    e = metrics["n_edges"]
    if n <= 1:
        return metrics

    # Undirected copy for basic metrics
    UG = nx.Graph()
    for u, v, data in digraph.edges(data=True):
        weight_here = data["weight"]
        existing_w = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing_w + weight_here)

    # 1) density
    metrics["density"] = nx.density(UG)

    # 2) average clustering
    if n > 1:
        metrics["avg_clustering"] = nx.average_clustering(UG)
        try:
            largest_cc = max(nx.connected_components(UG), key=len)
            subg = UG.subgraph(largest_cc).copy()
            if subg.number_of_nodes() > 1:
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(subg)
        except:
            metrics["avg_shortest_path_length"] = None

    # 3) cliques
    try:
        cliques = list(nx.find_cliques(UG))
        metrics["num_cliques"] = len(cliques)
        if cliques:
            metrics["max_clique_size"] = max(len(c) for c in cliques)
    except:
        pass

    # B) reciprocal ratio
    if e > 0:
        rec_edges = count_reciprocal_edges(digraph)
        metrics["reciprocal_ratio"] = rec_edges / e

    # C) fraction of edges that are bad->bad
    if e > 0:
        bb_edges = count_bad_bad_edges(digraph, bad_authors)
        metrics["bad_bad_fraction"] = bb_edges / e

    # D) largest strongly connected component
    metrics["largest_scc_size"] = largest_strongly_connected_component_size(digraph)

    return metrics

##############################################################################
# 5) Visualization (limit top 100 by degree, label top 10)
##############################################################################

def visualize_subgraph(digraph, output_filename, title="Local Subgraph", 
                       max_nodes=100, label_top=10):
    """
    1) Convert to UG for layout
    2) Keep only top 'max_nodes' by degree
    3) Label only top 'label_top' by degree
    """
    # Convert to undirected
    UG = nx.Graph()
    for u, v, data in digraph.edges(data=True):
        w_direct = data["weight"]
        existing_w = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing_w + w_direct)

    if UG.number_of_nodes() > max_nodes:
        # Sort by descending degree
        deg_sorted = sorted(UG.degree(), key=lambda x: x[1], reverse=True)
        keep = set(n for n, deg in deg_sorted[:max_nodes])
        UG = UG.subgraph(keep).copy()

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(UG, k=0.3, seed=42)

    nx.draw_networkx_edges(UG, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_nodes(UG, pos, node_color="lightblue", node_size=200, alpha=0.9)

    # label only top 'label_top' by degree
    deg_sorted = sorted(UG.degree(), key=lambda x: x[1], reverse=True)
    top_label_nodes = set(n for n, deg in deg_sorted[:label_top])
    labels_dict = {node: str(node) for node in top_label_nodes}

    nx.draw_networkx_labels(UG, pos, labels=labels_dict, font_size=7)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()

##############################################################################
# 6) MAIN
##############################################################################

def main():
    try:
        rolap_conn = sqlite3.connect(ROLAP_DB)
        impact_conn = sqlite3.connect(IMPACT_DB)

        # Load "bad" authors
        bad_authors = load_bad_authors(rolap_conn)
        print(f"[INFO] Loaded {len(bad_authors)} bad authors from orcid_h5_bottom.")

        # Example: pick top 5 pairs from matched_authors_with_counts
        cur = rolap_conn.cursor()
        sql = """
            SELECT bottom_orcid, random_orcid,
                   (COALESCE(bottom_n_works,0) + COALESCE(random_n_works,0)) AS total_works
            FROM matched_authors_with_counts
            ORDER BY total_works DESC
            LIMIT 5
        """
        cur.execute(sql)
        pairs = cur.fetchall()
        cur.close()
        print(f"[INFO] Found {len(pairs)} pairs from matched_authors_with_counts.")

        all_results = []
        pair_idx = 0

        for (bottom_orc, random_orc, sum_works) in pairs:
            pair_idx += 1
            print(f"\n[INFO] Pair#{pair_idx}: bottom={bottom_orc}, random={random_orc}, total_works={sum_works}")

            for (author, cat) in [(bottom_orc, "BOTTOM"), (random_orc, "RANDOM")]:
                # BFS with depth=2 + node cap
                digraph = build_local_author_subgraph(
                    rolap_conn, 
                    impact_conn, 
                    start_author=author, 
                    depth=2, 
                    max_nodes_in_bfs=2000  # cap BFS at 2000 authors
                )

                # Compute metrics
                m = compute_local_metrics(digraph, bad_authors)
                all_results.append({
                    "pair_idx": pair_idx,
                    "author": author,
                    "category": cat,
                    "metrics": m
                })

        # Sort results e.g. by 'bad_bad_fraction'
        all_results.sort(key=lambda x: x["metrics"]["bad_bad_fraction"], reverse=True)

        # Visualize top 2
        top_2 = all_results[:2]
        for i, row in enumerate(top_2, start=1):
            author = row["author"]
            cat = row["category"]
            # Re-build BFS subgraph to visualize
            subg = build_local_author_subgraph(
                rolap_conn, 
                impact_conn, 
                start_author=author, 
                depth=2, 
                max_nodes_in_bfs=2000
            )

            dens = row["metrics"]["density"]
            bbf = row["metrics"]["bad_bad_fraction"]
            recp = row["metrics"]["reciprocal_ratio"]

            out_fn = f"pair{row['pair_idx']}_{cat}_{author}_plot{i}.png"
            title_str = (f"Pair#{row['pair_idx']} {cat}={author}, density={dens:.3f}\n"
                         f"bad->bad={bbf:.3f}, reciprocal={recp:.3f}")
            visualize_subgraph(subg, out_fn, title=title_str, max_nodes=100, label_top=10)
            print(f"[INFO] Created plot: {out_fn}")

        # Write results to CSV
        with open("analysis_depth2_withcap.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "pair_idx", "author", "category",
                "n_nodes", "n_edges", "density", 
                "avg_clustering", "avg_shortest_path_length",
                "num_cliques", "max_clique_size",
                "reciprocal_ratio", "bad_bad_fraction",
                "largest_scc_size"
            ]
            writer.writerow(headers)
            for row in all_results:
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

        print("[INFO] Analysis done. Results in 'analysis_depth2_withcap.csv'.")

        rolap_conn.close()
        impact_conn.close()

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
