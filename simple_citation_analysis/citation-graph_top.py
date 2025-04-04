import networkx as nx
import sqlite3
import matplotlib.pyplot as plt

s = set()

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
    print("START", start, depth, flush=True)
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
    graph = nx.Graph()  # Use directed graph to get in_degree and out_degree
    add_citation_edges(connection, graph, start, 1)
    return graph


def graph_properties(rolap_connection, graph_connection, selection, file_name):
    """Write to the specified file name the graph properties of the work-ids
    obtained from rolap through the specified selection statement"""
    cursor = rolap_connection.cursor()
    cursor.execute(selection)
    with open(file_name, "w") as fh:
        for (id, subject ) in cursor:
            graph = citation_graph(graph_connection, id)
            print(id, graph)
            avg_clustering = nx.average_clustering(graph)
            #avg_path_length = nx.average_shortest_path_length(graph)
            fh.write(f"{subject}\t{avg_clustering}\n")
            if avg_clustering > 0.07:
                s.add(id)
            print(id, subject, avg_clustering)
    cursor.close()

def visualize_graph(graph, file_name):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=50, font_size=8)
    plt.savefig(file_name, format="pdf")
    plt.close()

graph_connection = sqlite3.connect(GRAPH_DATABASE_PATH)
rolap_connection = sqlite3.connect(ROLAP_DATABASE_PATH)

graph_properties(
    rolap_connection,
    graph_connection,
    "SELECT id, subject FROM random_top_works ",
    "reports/graph-top.txt",
)

graph_properties(
    rolap_connection,
    graph_connection,
    "SELECT id, subject FROM random_top_other_works ",
    "reports/graph-other-top.txt",
)

# Example of visualizing a graph for a specific work-id
example_graph = citation_graph(graph_connection, s.pop())
visualize_graph(example_graph, "reports/example_top_graph.pdf")