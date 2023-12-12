import osmnx as ox
import cv2
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import heapq

def dijkstra_path(graph, start_node, end_node, weight='length'):
    try:
        # Khởi tạo các biến
        distances = {node: float('infinity') for node in graph.nodes}
        distances[start_node] = 0
        previous_nodes = {node: None for node in graph.nodes}
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == end_node:
                # Đã đến đỉnh kết thúc, trả về đường đi
                path = []
                while current_node is not None:
                    path.insert(0, current_node)
                    current_node = previous_nodes[current_node]
                return path

            for neighbor, edge_data in graph[current_node].items():
                weight_value = edge_data.get(weight, 1)
                new_distance = distances[current_node] + weight_value

                if new_distance < distances[neighbor]:
                    # Cập nhật khoảng cách và đỉnh trước đó nếu tìm thấy đường đi ngắn hơn
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        # Không có đường đi giữa hai đỉnh
        print("Không có đường đi giữa hai đỉnh này.")
        return []
    except nx.NodeNotFound:
        print("Một hoặc cả hai đỉnh không tồn tại trong đồ thị.")
        return []
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return []
