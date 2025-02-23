import sqlite3
import networkx as nx
import traceback
import statistics
import csv

# Paths to your DBs
ROLAP_DATABASE = "rolap.db"   # Contains works_orcid, orcid_h5_bottom
IMPACT_DATABASE = "impact.db" # Contains works, work_references

###############################################################################
#                          AUTHOR LABELING
###############################################################################
def is_bad_author(rolap_cursor, orcid):
    """
    Check if an author (ORCID) is in 'orcid_h5_bottom' (in ROLAP DB).
    """
    sql = "SELECT 1 FROM orcid_h5_bottom WHERE orcid = ? LIMIT 1"
    rolap_cursor.execute(sql, (orcid,))
    return rolap_cursor.fetchone() is not None

def label_author(rolap_cursor, orcid):
    """
    Return "BAD" if author is in orcid_h5_bottom, else "GOOD" (in ROLAP DB).
    """
    return "BAD" if is_bad_author(rolap_cursor, orcid) else "GOOD"

###############################################################################
#              BUILD A GLOBAL AUTHOR CITATION GRAPH (Chunked Processing)
###############################################################################
def build_author_graph_chunked(
    rolap_conn, impact_conn, batch_size=10000, skip_self_citations=True
):
    """
    Process authors in batches to build the graph incrementally.
    """
    rolap_cursor = rolap_conn.cursor()
    rolap_cursor.execute("SELECT DISTINCT orcid FROM works_orcid WHERE orcid IS NOT NULL")
    DG = nx.DiGraph()
    batch_num = 0

    while True:
        batch = rolap_cursor.fetchmany(batch_size)
        if not batch:
            break
        batch_orcids = [row[0] for row in batch]

        subgraph = build_author_graph(
            rolap_conn=rolap_conn,
            impact_conn=impact_conn,
            max_authors=len(batch_orcids),
            skip_self_citations=skip_self_citations
        )
        DG.update(subgraph)
        batch_num += 1
        print(f"[INFO] Processed batch {batch_num} of {len(batch_orcids)} authors.")

    rolap_cursor.close()
    return DG

def build_author_graph(
    rolap_conn, impact_conn, max_authors=None, skip_self_citations=True
):
    """
    Build a directed author graph:
      - Nodes = authors (ORCID) from ROLAP 'works_orcid'.
      - Edges from A->B with weight = total # references from A's works to B's works.
        Uses 'work_references' + 'works' in IMPACT DB to find citations.
    Returns: networkx.DiGraph
    """
    rolap_cursor = rolap_conn.cursor()
    impact_cursor = impact_conn.cursor()

    sql_authors = "SELECT DISTINCT orcid FROM works_orcid WHERE orcid IS NOT NULL"
    rolap_cursor.execute(sql_authors)
    all_orcids = [row[0] for row in rolap_cursor.fetchall()]

    if max_authors and len(all_orcids) > max_authors:
        all_orcids = all_orcids[:max_authors]

    orcid_set = set(all_orcids)
    print(f"[INFO] Building author-level graph for {len(all_orcids)} authors...")

    sql_works = "SELECT orcid, id FROM works_orcid WHERE orcid IS NOT NULL"
    rolap_cursor.execute(sql_works)
    author_to_works = {}
    for (orc, w_id) in rolap_cursor.fetchall():
        if orc in orcid_set:
            author_to_works.setdefault(orc, []).append(w_id)

    def get_outgoing_references(work_id):
        sql_refs = """
            SELECT w.id
            FROM work_references wr
            JOIN works w ON wr.doi = w.doi
            WHERE wr.work_id = ?
        """
        impact_cursor.execute(sql_refs, (work_id,))
        return [r[0] for r in impact_cursor.fetchall()]

    def get_authors_for_work(work_id):
        sql_auth = "SELECT orcid FROM works_orcid WHERE id = ? AND orcid IS NOT NULL"
        rolap_cursor.execute(sql_auth, (work_id,))
        return [r[0] for r in rolap_cursor.fetchall()]

    edge_weights = {}
    countA = 0
    for orcidA in all_orcids:
        countA += 1
        if countA % 50000 == 0:
            print(f"  Processed {countA} authors so far...")
        a_works = author_to_works.get(orcidA, [])
        for w_id in a_works:
            cited_ids = get_outgoing_references(w_id)
            for c_id in cited_ids:
                cited_authors = get_authors_for_work(c_id)
                for orcidB in cited_authors:
                    if skip_self_citations and orcidB == orcidA:
                        continue
                    if orcidB in orcid_set:
                        edge_weights[(orcidA, orcidB)] = edge_weights.get((orcidA, orcidB), 0) + 1

    DG = nx.DiGraph()
    DG.add_nodes_from(all_orcids)
    for (a, b), w in edge_weights.items():
        DG.add_edge(a, b, weight=w)

    rolap_cursor.close()
    impact_cursor.close()
    return DG

def make_undirected(digraph):
    """
    Convert a directed graph to an undirected one by summing
    the weights in both directions: w(u<->v) = w(u->v) + w(v->u).
    """
    UG = nx.Graph()
    UG.add_nodes_from(digraph.nodes())
    for u, v, data in digraph.edges(data=True):
        w_direct = data["weight"]
        existing_w = UG[u][v]["weight"] if UG.has_edge(u, v) else 0
        UG.add_edge(u, v, weight=existing_w + w_direct)
    return UG

###############################################################################
#                   GLOBAL METRICS & CLIQUE DETECTION
###############################################################################
def compute_global_author_metrics(und_graph):
    """
    Compute global metrics:
      - number of nodes, edges
      - density
      - avg_clustering
      - average shortest path length (largest CC)
      - #cliques, max_clique_size
    """
    nm = {
        "num_nodes": und_graph.number_of_nodes(),
        "num_edges": und_graph.number_of_edges(),
        "density": 0.0,
        "avg_clustering": 0.0,
        "avg_shortest_path_length": None,
        "num_cliques": 0,
        "max_clique_size": 0
    }

    n = nm["num_nodes"]
    if n > 1:
        nm["density"] = nx.density(und_graph)
        nm["avg_clustering"] = nx.average_clustering(und_graph)
        largest_cc = max(nx.connected_components(und_graph), key=len)
        subg = und_graph.subgraph(largest_cc).copy()
        if subg.number_of_nodes() > 1:
            try:
                nm["avg_shortest_path_length"] = nx.average_shortest_path_length(subg)
            except:
                nm["avg_shortest_path_length"] = None

    cliques = list(nx.find_cliques(und_graph))
    nm["num_cliques"] = len(cliques)
    if cliques:
        nm["max_clique_size"] = max(len(c) for c in cliques)

    return nm

###############################################################################
#                       OUTLIER DETECTION
###############################################################################
def detect_author_outliers_by_degree(graph, threshold_std=2.0):
    """
    Mark authors whose weighted degree is more than 'threshold_std' 
    standard deviations above the mean as outliers.
    """
    degs = []
    for node in graph.nodes():
        wdeg = graph.degree(node, weight="weight")
        degs.append((node, wdeg))

    if len(degs) < 2:
        return []

    values = [d[1] for d in degs]
    mean_val = statistics.mean(values)
    stdev_val = statistics.pstdev(values)
    if stdev_val == 0:
        return []

    outliers = []
    for (node, deg_val) in degs:
        if deg_val > mean_val + threshold_std * stdev_val:
            z = (deg_val - mean_val) / stdev_val
            outliers.append((node, deg_val, z))
    return outliers

###############################################################################
#                              MAIN
###############################################################################
def main():
    rolap_conn = None
    impact_conn = None
    try:
        rolap_conn = sqlite3.connect(ROLAP_DATABASE)
        impact_conn = sqlite3.connect(IMPACT_DATABASE)

        di_graph = build_author_graph_chunked(
            rolap_conn=rolap_conn,
            impact_conn=impact_conn,
            batch_size=10000,
            skip_self_citations=True
        )
        print(f"[INFO] Directed graph: {di_graph.number_of_nodes()} authors, {di_graph.number_of_edges()} edges.")

        und_graph = make_undirected(di_graph)
        print(f"[INFO] Undirected graph: {und_graph.number_of_nodes()} authors, {und_graph.number_of_edges()} edges.")

        print("[INFO] Computing global metrics on the author graph...")
        g_metrics = compute_global_author_metrics(und_graph)
        for k, v in g_metrics.items():
            print(f"{k} = {v}")

        outliers = detect_author_outliers_by_degree(und_graph, threshold_std=2.0)
        print(f"\n[INFO] Found {len(outliers)} outlier author(s).")
        for (orc, deg, z) in outliers[:30]:
            print(f" -> ORCID={orc}, WeightedDeg={deg:.2f}, Z={z:.2f}")

        with open("author_outliers.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ORCID", "WeightedDegree", "ZScore"])
            for (orc, deg, z) in outliers:
                writer.writerow([orc, deg, z])
        print("[INFO] Outlier authors saved to author_outliers.csv")

    except Exception as ex:
        print(f"ERROR: {ex}")
        traceback.print_exc()
    finally:
        if rolap_conn:
            rolap_conn.close()
        if impact_conn:
            impact_conn.close()

if __name__ == "__main__":
    main()
