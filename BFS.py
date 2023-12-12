from queue import Queue


def bfs(graph, start, end):
    visited = set()
    queue = Queue()
    parent = {}

    queue.put(start)
    visited.add(start)

    while not queue.empty():
        current_node = queue.get()

        if current_node == end:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = parent.get(current_node)

            return path

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current_node
                queue.put(neighbor)

    return None





