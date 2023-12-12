import networkx as nx

def dijkstra_path(graph, start, end, weight='length'):
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight=weight)
        return path
    except nx.NetworkXNoPath:
        print("Không tìm thấy đường đi giữa hai đỉnh.")
        return None
