import networkx as nx
import sqlite3

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
    (doi,) = cursor.fetchone()
    return doi

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
    for (id,) in cursor:
        graph.add_edge(start, id)
        add_citation_edges(connection, graph, id, depth - 1)

    # Incoming references
    work_doi = workid_doi(cursor, start)
    cursor.execute("SELECT work_id FROM work_references WHERE doi = ?", (work_doi,))
    for (id,) in cursor:
        graph.add_edge(start, id)
        add_citation_edges(connection, graph, id, depth - 1)

    cursor.close()

def citation_graph(connection, start):
    """Return a graph induced by incoming and outgoing citations of
    distance 2 for the specified work-id"""
    graph = nx.Graph()
    add_citation_edges(connection, graph, start, 1)
    return graph

def find_and_save_cliques(rolap_connection, graph_connection, selection, file_name):
    """Find cliques for the work-ids obtained from the specified selection statement"""
    cursor = rolap_connection.cursor()
    cursor.execute(selection)
    with open(file_name, "w") as fh:
        for (id, subject) in cursor:
            graph = citation_graph(graph_connection, id)
            cliques = find_citation_cliques(graph)
            # Write subject, number of cliques, and size of largest clique
            fh.write(f"{subject}\t{len(cliques)}\t{max([len(clique) for clique in cliques], default=0)}\n")
            print(f"Work ID: {id}, Subject: {subject}, Number of Cliques: {len(cliques)}, Largest Clique Size: {max([len(clique) for clique in cliques], default=0)}")
    cursor.close()

# Connect to databases
graph_connection = sqlite3.connect(GRAPH_DATABASE_PATH)
rolap_connection = sqlite3.connect(ROLAP_DATABASE_PATH)

# Find cliques for random_top_works and random_bottom_works
find_and_save_cliques(
    rolap_connection,
    graph_connection,
    "SELECT id, subject FROM random_top_works_h5",
    "reports/cliques-top_h5.txt"
)

find_and_save_cliques(
    rolap_connection,
    graph_connection,
    "SELECT id, subject FROM random_top_other_works_h5",
    "reports/cliques-other_h5.txt"
)
