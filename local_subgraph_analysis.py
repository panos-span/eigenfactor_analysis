import sqlite3
import networkx as nx
import csv
import traceback
import statistics
from collections import deque

ROLAP_DB = "rolap.db"
IMPACT_DB = "impact.db"

##############################################################################
#                          HELPERS
##############################################################################

def get_works_for_author(cursor_rolap, orcid):
    """
    Return all work_ids for the given author (ORCID) from works_orcid.
    """
    sql = "SELECT id FROM works_orcid WHERE orcid = ?"
    cursor_rolap.execute(sql, (orcid,))
    return [row[0] for row in cursor_rolap.fetchall()]

def get_cited_work_ids(cursor_impact, work_id):
    """
    Return all work_ids that 'work_id' cites (IMPACT DB: works + work_references).
    """
    sql = """
        SELECT w.id
        FROM work_references wr
        JOIN works w ON wr.doi = w.doi
        WHERE wr.work_id = ?
    """
    cursor_impact.execute(sql, (work_id,))
    return [r[0] for r in cursor_impact.fetchall()]

def get_authors_for_work(cursor_rolap, work_id):
    """
    Return all authors (ORCIDs) for the given work_id from ROLAP DB.
    """
    sql = "SELECT orcid FROM works_orcid WHERE id = ?"
    cursor_rolap.execute(sql, (work_id,))
    return [r[0] for r in cursor_rolap.fetchall() if r[0] is not None]


def build_local_author_subgraph(rolap_conn, impact_conn, start_author, depth=1):
    """
    Perform a BFS from 'start_author' up to 'depth' to build a local directed graph 
    of authors, where edges are "cites" relationships. 
    If depth=1, we only get 'start_author' -> others. 
    If depth=2, we also follow newly discovered authors, and so forth.

    Return a NetworkX DiGraph.
    """
    G = nx.DiGraph()
    visited_authors = set([start_author])
    queue = deque()
    queue.append((start_author, 0))

    rolap_cursor = rolap_conn.cursor()
    impact_cursor = impact_conn.cursor()

    while queue:
        current_author, dist = queue.popleft()
        if dist >= depth:
            continue

        # 1) Get works for current_author
        works = get_works_for_author(rolap_cursor, current_author)
        # 2) For each work, get its outgoing references (in IMPACT)
        for w in works:
            cited_ids = get_cited_work_ids(impact_cursor, w)
            # 3) For each cited work, get the authors
            for cwid in cited_ids:
                cited_authors = get_authors_for_work(rolap_cursor, cwid)
                for ca in cited_authors:
                    # Add edge current_author -> ca with weight
                    if G.has_edge(current_author, ca):
                        G[current_author][ca]["weight"] += 1
                    else:
                        G.add_edge(current_author, ca, weight=1)

                    if ca not in visited_authors:
                        visited_authors.add(ca)
                        queue.append((ca, dist + 1))

    rolap_cursor.close()
    impact_cursor.close()
    return G

def compute_local_metrics(graph):
    """
    Compute some basic metrics on a local subgraph:
    - number_of_nodes
    - number_of_edges
    - density (for an undirected version or keep directed if you prefer)
    - average clustering (again typically for undirected; we can convert)
    - average shortest path length (largest CC in undirected)
    - # of cliques, max clique size (in undirected sense)
    """
    metrics = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": 0.0,
        "avg_clustering": 0.0,
        "avg_shortest_path_length": None,
        "num_cliques": 0,
        "max_clique_size": 0
    }
    n = metrics["n_nodes"]

    if n <= 1:
        return metrics

    # Convert to undirected for the standard definitions of clustering, path length, cliques
    UG = nx.Graph()
    # Combine weights in both directions
    for u, v, data in graph.edges(data=True):
        w_direct = data["weight"]
        existing_w = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing_w + w_direct)

    metrics["density"] = nx.density(UG)
    if n > 1:
        metrics["avg_clustering"] = nx.average_clustering(UG)
        largest_cc = max(nx.connected_components(UG), key=len)
        subg = UG.subgraph(largest_cc).copy()
        if subg.number_of_nodes() > 1:
            try:
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(subg)
            except:
                metrics["avg_shortest_path_length"] = None

    # Cliques (warning: can be expensive if subgraph is big)
    cliques = list(nx.find_cliques(UG))
    metrics["num_cliques"] = len(cliques)
    if cliques:
        metrics["max_clique_size"] = max(len(c) for c in cliques)
    return metrics


##############################################################################
#                          MAIN
##############################################################################
def main():
    try:
        # Connect to DBs
        rolap_conn = sqlite3.connect(ROLAP_DB)
        impact_conn = sqlite3.connect(IMPACT_DB)

        # Step 1: Load the relevant authors from matched_authors (which has 50 bottom & 50 random authors per subject)
        # We'll store them in a list of (bottom_orcid, random_orcid).
        matched_list = []
        rcur = rolap_conn.cursor()
        rcur.execute("SELECT bottom_orcid, random_orcid FROM matched_authors")
        rows = rcur.fetchall()
        for (bot, rnd) in rows:
            if bot and rnd:
                matched_list.append((bot, rnd))
        rcur.close()

        print(f"[INFO] Found {len(matched_list)} matched bottom-random pairs.")

        # Step 2: For each pair, BFS for bottom_orcid and BFS for random_orcid
        #         Then compute metrics, store them in CSV or memory
        results = []
        count_pairs = 0
        for (bottom_author, random_author) in matched_list:
            count_pairs += 1
            if count_pairs % 10 == 0:
                print(f"[INFO] Processed {count_pairs} pairs so far...")

            # BFS depth=1 for bottom_author
            G_bottom = build_local_author_subgraph(rolap_conn, impact_conn, bottom_author, depth=1)
            met_bottom = compute_local_metrics(G_bottom)
            met_bottom["orcid"] = bottom_author
            met_bottom["category"] = "BOTTOM"

            # BFS depth=1 for random_author
            G_random = build_local_author_subgraph(rolap_conn, impact_conn, random_author, depth=1)
            met_random = compute_local_metrics(G_random)
            met_random["orcid"] = random_author
            met_random["category"] = "RANDOM"

            results.append(met_bottom)
            results.append(met_random)

        # Step 3: Write results to a CSV
        with open("local_subgraph_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "orcid", 
                "category",
                "n_nodes", 
                "n_edges", 
                "density", 
                "avg_clustering", 
                "avg_shortest_path_length",
                "num_cliques",
                "max_clique_size"
            ]
            writer.writerow(header)
            for row in results:
                writer.writerow([
                    row["orcid"],
                    row["category"],
                    row["n_nodes"],
                    row["n_edges"],
                    f"{row['density']:.5f}",
                    f"{row['avg_clustering']:.5f}",
                    f"{row['avg_shortest_path_length']}" if row["avg_shortest_path_length"] else "",
                    row["num_cliques"],
                    row["max_clique_size"]
                ])

        print("[INFO] Local subgraph metrics saved to local_subgraph_metrics.csv")

        rolap_conn.close()
        impact_conn.close()

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
