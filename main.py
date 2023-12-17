import osmnx as ox
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from Dijkstra import dijkstra_path
from Astar import astar_path
from BFS import bfs
import tkinter as tk
from tkinter import simpledialog
import copy

def convert_coords(img_coords):
    # Tọa độ pixel của ảnh
    img_point1 = (0, 0)  # Góc trái trên
    img_point2 = (1358, 990)  # Góc phải dưới

    # Tọa độ tương ứng của 2 điểm trên bản đồ OSM
    osm_point1 = (105.84587, 21.04141)
    osm_point2 = (105.85315, 21.03647)

    # Kích thước thực của khu vực trên bản đồ OSM
    map_width_degree = osm_point2[0] - osm_point1[0]
    map_height_degree = osm_point1[1] - osm_point2[1]

    # Kích thước thực của hình ảnh
    img_width = img_point2[0] - img_point1[0]
    img_height = img_point2[1] - img_point1[1]

    # Tính tỷ lệ chuyển đổi
    pixel_to_degree_x = map_width_degree / img_width
    pixel_to_degree_y = map_height_degree / img_height

    # Chuyển đổi từ tọa độ pixel sang tọa độ địa lý
    osm_coords = (
        round(img_coords[0] * pixel_to_degree_x + osm_point1[0], 5),
        round(img_coords[1] * pixel_to_degree_y + osm_point2[1], 5)
    )

    return osm_coords
# Hàm in tọa độ
def convert_coords_reverse(osm_coords):
    # Tọa độ pixel của ảnh
    img_point1 = (0, 0)  # Góc trái trên
    img_point2 = (1358, 990)  # Góc phải dưới

    # Tọa độ tương ứng của 2 điểm trên bản đồ OSM
    osm_point1 = (105.84587, 21.04141)
    osm_point2 = (105.85315, 21.03647)

    # Kích thước thực của khu vực trên bản đồ OSM
    map_width_degree = osm_point2[0] - osm_point1[0]
    map_height_degree = osm_point1[1] - osm_point2[1]

    # Kích thước thực của hình ảnh
    img_width = img_point2[0] - img_point1[0]
    img_height = img_point2[1] - img_point1[1]

    # Tính tỷ lệ chuyển đổi ngược lại
    degree_to_pixel_x = img_width / map_width_degree
    degree_to_pixel_y = img_height / map_height_degree
    # Chuyển đổi từ tọa độ địa lý trên bản đồ OSM về tọa độ pixel
    img_coords = (
        int((osm_coords[0] - osm_point1[0]) * degree_to_pixel_x),
        int((osm_point2[1] - osm_coords[1]) * degree_to_pixel_y)
    )

    return img_coords

# Hàm in tọa độ node
def draw_coordinates(img, coords, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    color = (14,59,255) 
    cv2.putText(img, text, coords, font, font_scale, color, font_thickness, cv2.LINE_AA)
#Hàm chọn thuật toán
def choose_algorithm():
    selected_algorithm = [None]

    def set_algorithm(algo):
        selected_algorithm[0] = algo
        algorithm_window.destroy()

    algorithm_window = tk.Tk()
    algorithm_window.title("Choose Algorithm")

    # Calculate window position to center it on the screen
    window_width = 300
    window_height = 200
    screen_width = algorithm_window.winfo_screenwidth()
    screen_height = algorithm_window.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    algorithm_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    button_font = ("Arial", 12)  # Adjust the font size as needed

    dijkstra_button = tk.Button(algorithm_window, text="Dijkstra", width=15, height=2, font=button_font, command=lambda: set_algorithm('dijkstra'))
    dijkstra_button.pack(side=tk.TOP, pady=5)

    astar_button = tk.Button(algorithm_window, text="A* (Astar)", width=15, height=2, font=button_font, command=lambda: set_algorithm('astar'))
    astar_button.pack(side=tk.TOP, pady=5)

    bfs_button = tk.Button(algorithm_window, text="BFS", width=15, height=2, font=button_font, command=lambda: set_algorithm('bfs'))
    bfs_button.pack(side=tk.TOP, pady=5)

    algorithm_window.mainloop()

    return selected_algorithm[0]


        
def click_event(event, x, y, flags, param):
    global click_count, selected_algorithm

    if event == cv2.EVENT_LBUTTONDOWN:
        global image, start, end, node1_start, node2_start, node1_end, node2_end
        if click_count == 0:
            global start
            img_coords = (x, height - y)
            osm_coords = convert_coords(img_coords)
            osm_node = ox.distance.nearest_nodes(G, osm_coords[0], osm_coords[1])
            img_node = convert_coords_reverse((G.nodes[osm_node]['x'], G.nodes[osm_node]['y']))
            draw_coordinates(image, (x, y), f'Start{osm_coords}')
            cv2.imshow('Image', image)  
            # cv2.line(image, (x, y), (img_node[0], height + img_node[1]), (255, 125, 38), 7)
            start = osm_node
            node1_start = (x,y)
            node2_start = img_node
        elif click_count == 1:
            img_coords = (x, height - y)
            osm_coords = convert_coords(img_coords)
            osm_node = ox.distance.nearest_nodes(G, osm_coords[0], osm_coords[1]) #check
            img_node = convert_coords_reverse((G.nodes[osm_node]['x'], G.nodes[osm_node]['y']))
            node1_end = (x,y)
            node2_end = img_node
            draw_coordinates(image, (x, y), f'End{osm_coords}')
            cv2.imshow('Image', image)
            # cv2.line(image, (x, y), (img_node[0], height + img_node[1]), (255, 125, 38), 7)
            end = osm_node
            selected_algorithm = choose_algorithm()
            print(f"Selected Algorithm: {selected_algorithm}")
            ans = None
            if selected_algorithm == 'dijkstra':
                ans = dijkstra_path(G, start, end, weight='length')
            elif selected_algorithm == 'astar':
                ans = astar_path(G, start, end, weight='length')
            elif selected_algorithm == 'bfs':
                ans = bfs(G, start, end)
            image_2 = image.copy()
            cv2.line(image, (node1_start[0], node1_start[1]), (node2_start[0], height + node2_start[1]), (0, 255, 0), 7)
            if ans:
                for i in range(len(ans) - 1):
                    node1 = ans[i]
                    node2 = ans[i + 1]
                    node1_img = convert_coords_reverse((G.nodes[node1]['x'], G.nodes[node1]['y']))
                    node2_img = convert_coords_reverse((G.nodes[node2]['x'], G.nodes[node2]['y']))
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                    cv2.imshow('Image', image)
                    cv2.line(image, (node1_img[0], height + node1_img[1]), (node2_img[0], height + node2_img[1]), (0, 255, 0), 7)
                    cv2.imshow('Image', image)
                    print(f'Node {i + 1}: OSM Coordinates = ({node1_img[0]:.5f}, {node1_img[1]:.5f})')
                    cv2.waitKey(20)
                    
                    for edge in G.edges(node1):
                        u, v = edge
                        u_img = convert_coords_reverse((G.nodes[u]['x'], G.nodes[u]['y']))
                        v_img = convert_coords_reverse((G.nodes[v]['x'], G.nodes[v]['y']))
                        cv2.line(image, (u_img[0], height + u_img[1]), (v_img[0], height + v_img[1]), (0, 255, 0), 7)
                        cv2.imshow('Image', image)
                        cv2.waitKey(20)
                image = image_2.copy()
                cv2.imshow('Image', image)
                cv2.line(image, (node1_start[0], node1_start[1]), (node2_start[0], height + node2_start[1]), (255, 125, 38), 7)
                cv2.line(image, (node1_end[0], node1_end[1]), (node2_end[0], height + node2_end[1]), (255, 125, 38), 7)
                for i in range(len(ans) - 1):
                    node1 = ans[i]
                    node2 = ans[i + 1]
                    node1_img = convert_coords_reverse((G.nodes[node1]['x'], G.nodes[node1]['y']))
                    node2_img = convert_coords_reverse((G.nodes[node2]['x'], G.nodes[node2]['y']))
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                    cv2.imshow('Image', image)
                    cv2.line(image, (node1_img[0], height + node1_img[1]), (node2_img[0], height + node2_img[1]), (255, 125, 38), 7)
                    cv2.imshow('Image', image)
                    print(f'Node {i + 1}: OSM Coordinates = ({node1_img[0]:.5f}, {node1_img[1]:.5f})')
                cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.imshow('Image', image)

                # Use cv2.waitKey() with a short delay
                cv2.waitKey(1)

        else:
            image = image_copy.copy()
            cv2.imshow('Image', image)

        click_count += 1
        if click_count > 2:
            click_count = 0


            
click_count = 0
start = 0
end = 0
selected_algorithm = None  # Initialize selected_algorithm
tk_root = None  # Initialize Tkinter root window
img_file_path = 'PhuongDongXuan.png'
image = cv2.imread(img_file_path)
image_copy = cv2.imread(img_file_path)
G = ox.graph_from_bbox(21.0414, 21.03647, 105.8458, 105.85315, network_type='all')
height, width, _ = image.shape


cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)

while cv2.getWindowProperty('Image', 0) >= 0:
    cv2.waitKey(1)

# Close all OpenCV windows
cv2.destroyAllWindows()
