import osmnx as ox
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Dijkstra import dijkstra_path
from Astar import astar_path
from BFS import bfs

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
        img_coords[0] * pixel_to_degree_x + osm_point1[0],
        img_coords[1] * pixel_to_degree_y + osm_point2[1]
    )

    return osm_coords

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

def click_event(event, x, y, flags, param):
    global click_count
    if event == cv2.EVENT_LBUTTONDOWN:
        global image
        if click_count == 0:
            global start
            img_coords = (x, height - y)
            osm_coords = convert_coords(img_coords)
            osm_node = ox.distance.nearest_nodes(G, osm_coords[0], osm_coords[1])
            img_node = convert_coords_reverse((G.nodes[osm_node]['x'], G.nodes[osm_node]['y']))
            cv2.line(image, (x, y), (img_node[0], height + img_node[1]), (0, 255, 0), 2)
            cv2.imshow('Image', image)
            start = osm_node
        elif click_count == 1:
            img_coords = (x, height - y)
            osm_coords = convert_coords(img_coords)
            osm_node = ox.distance.nearest_nodes(G, osm_coords[0], osm_coords[1]) #check
            img_node = convert_coords_reverse((G.nodes[osm_node]['x'], G.nodes[osm_node]['y']))
            cv2.line(image, (x, y), (img_node[0], height + img_node[1]), (0, 255, 0), 2)
            cv2.imshow('Image', image)
            end = osm_node
            
            # Giao diện cho 3 thuật toán 
            ans = dijkstra_path(G, start, end, weight='length') 
            # ans = astar_path(G, start, end, weight='length')
            # ans = bfs(G, start, end)
            ###########################################
            
            for i in range(len(ans) - 1):
                node1 = ans[i]
                node2 = ans[i + 1]
                # Lấy tọa độ của node1 và node2 từ đồ thị G
                node1_img = convert_coords_reverse((G.nodes[node1]['x'], G.nodes[node1]['y']))
                node2_img = convert_coords_reverse((G.nodes[node2]['x'], G.nodes[node2]['y']))
                # Vẽ đường nối giữa node1 và node2 trên ảnh
                cv2.line(image, (node1_img[0],height+node1_img[1]), (node2_img[0],height+node2_img[1]), (0, 255, 0), 2)
                cv2.imshow('Image', image)
        else:
            image = image_copy.copy()
            cv2.imshow('Image', image)

        click_count += 1
        if click_count > 2:
            click_count = 0
            

    
click_count = 0
start = 0
end = 0
img_file_path = 'PhuongDongXuan.png'
image = cv2.imread(img_file_path)
image_copy = cv2.imread(img_file_path)
G = ox.graph_from_bbox(21.0414,21.03647,105.8458,105.85315,network_type='all' )
height, width, _ = image.shape

cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)




