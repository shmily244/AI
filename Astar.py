import numpy as np
import heapq

def heuristic(node, goal, G):
    pos1 = (G.nodes[node]['y'], G.nodes[node]['x'])
    pos2 = (G.nodes[goal]['y'], G.nodes[goal]['x'])
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def astar_path(G, start, goal, weight='length'):
    open_set = []
    closed_set = set()
    came_from = {}

    # Priority queue for the open set
    heapq.heappush(open_set, (0, start))

    # Cost from start along the best known path
    g_score = {start: 0}

    # Estimated total cost from start to goal through y
    f_score = {start: heuristic(start, goal, G)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for neighbor in G.neighbors(current):
            if neighbor in closed_set:
                continue

            if 'length' in G[current][neighbor]:
                edge_length = G[current][neighbor]['length']
            else:
                # Xử lý khi 'length' không tồn tại, gán edge_length = 0.00001
                edge_length = 0.00001
            tentative_g_score = g_score[current] + edge_length

            if neighbor not in [i[1] for i in open_set]:
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal, G), neighbor))
            elif tentative_g_score >= g_score.get(neighbor, 0):
                continue
            
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, G)

    return None
