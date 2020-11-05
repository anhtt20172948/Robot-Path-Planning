import cv2
from setting import *
from collections import defaultdict
from delaunay2D import Delaunay2D
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
from dijkstra_algorithm import Graph
import math


class Main(object):
    def __init__(self):
        self.image = BLANK_IMAGE
        self.image[:] = tuple(reversed(WHITE))
        self.refPt = []
        self.save_data = defaultdict(int)
        self.num_obstacles = 0
        self.mid_point = []
        self.num_nodes = 0
        self.nodes = dict()
        self.graph_edges = []


    def click_and_crop(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the coordinat
        if event == cv2.EVENT_LBUTTONDOWN:
            if Polygon(Polygon_ROI).contains(Point(x, y)):
                self.refPt.append([x, y])
                cv2.circle(self.image, (x, y), 2, BLACK, thickness=-1, lineType=8, shift=0)

    def save_coord(self):

        area = ''

        pts = np.reshape(self.refPt, (-1, 1, 2))
        cv2.fillPoly(self.image, [pts], RED)
        cv2.imshow("image", self.image)
        cv2.waitKey(1)
        self.num_obstacles += 1
        while (area == ''):
            area = 'obstacles_{}'.format(self.num_obstacles)

        if area != '':
            self.save_data[area] = self.refPt
            print(f'Saved coordinate for area: {area}!')
            self.last_saved_image = self.image.copy()

        self.refPt = []

    def draw_init(self):
        # Draw Back Ground
        cv2.rectangle(self.image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), BLUE)
        # Draw start point
        cv2.circle(self.image, START_POINT, 10, RED, thickness=-1, lineType=8, shift=0)
        cv2.circle(self.image, START_POINT, 5, BLACK, thickness=-1, lineType=8, shift=0)
        cv2.putText(self.image, 'START POINT', (START_POINT[0] + 10, START_POINT[1] + 10),
                    cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, BLACK, 1, cv2.LINE_AA)

        # Draw end point
        cv2.circle(self.image, END_POINT, 10, RED, thickness=-1, lineType=8, shift=0)
        cv2.circle(self.image, END_POINT, 5, BLACK, thickness=-1, lineType=8, shift=0)
        cv2.putText(self.image, 'END POINT', (END_POINT[0] - 10, END_POINT[1] + 20),
                    cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, BLACK, 1, cv2.LINE_AA)

    def is_valid_polygon(self, triangle):
        for key in self.save_data.keys():
            obstacles = Polygon(self.save_data[key])
            polygon_intersection = obstacles.intersection(triangle).area
            polygon_union = obstacles.union(triangle).area
            IOU = polygon_intersection / polygon_union
            if IOU > 0:
                return False
        return True

    def is_valid_point(self, point):
        for key in self.save_data.keys():
            obstacles = Polygon(self.save_data[key])
            if obstacles.contains(Point(point)):
                return False
        return True

    def distance(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def init_graph(self):
        for pre_key in self.nodes.keys():
            for cur_key in self.nodes.keys():
                if pre_key != cur_key:
                    line = LineString([self.nodes[pre_key], self.nodes[cur_key]])
                    is_cutted = False
                    for key in self.save_data.keys():
                        obstacles = Polygon(self.save_data[key])
                        if line.intersects(obstacles):
                            is_cutted = True
                    if not is_cutted:
                        self.graph_edges.append((pre_key, cur_key, self.distance(self.nodes[pre_key], self.nodes[cur_key])))

    def draw_triangle(self, triangles, seeds):
        for triangle in triangles:
            p1 = seeds[triangle[0]]
            p2 = seeds[triangle[1]]
            p3 = seeds[triangle[2]]
            vertices = np.array([p1, p2, p3], np.int32)
            pts = vertices.reshape((-1, 1, 2))
            mid_12 = self.midpoint(p1, p2)
            mid_23 = self.midpoint(p2, p3)
            mid_13 = self.midpoint(p1, p3)

            if self.is_valid_point(mid_12):
                self.mid_point.append(mid_12)

            if self.is_valid_point(mid_23):
                self.mid_point.append(mid_23)

            if self.is_valid_point(mid_13):
                self.mid_point.append(mid_13)

            if self.is_valid_polygon(Polygon([p1, p2, p3])):
                cv2.polylines(self.image, [pts], isClosed=True, color=GRAY, thickness=1)

    def run(self):
        self.draw_init()
        self.clone = self.image.copy()
        self.last_saved_image = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
        # keep looping until the 'q' key is pressed

        while True:
            # display the image and wait for a key/mouse actions
            # _, image = cap.read()
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF
            # if the key 's' is pressed, save shape coordinates
            if key == ord("s"):
                self.save_coord()

            # if the key 'c' is pressed, clear current points
            if key == ord("c"):
                self.refPt = []
                self.image = self.last_saved_image.copy()
                # if the 'r' key is pressed, reset all drawn shape
            if key == ord("r"):
                self.image = self.clone.copy()
                self.last_saved_image = self.clone.copy()
                self.save_data = defaultdict(int)
                print('Reset all saved data!')

            if key == ord(" "):
                # Create a random set of 2D points
                seeds = np.empty([0, 2])
                seeds = np.concatenate((seeds, np.asarray([START_POINT])), axis=0)

                for key in self.save_data.keys():
                    obstacles = self.save_data[key]
                    seeds = np.concatenate((seeds, np.asarray(obstacles)), axis=0)

                for cor in Polygon_ROI:
                    print(cor)
                    seeds = np.concatenate((seeds, np.asarray([cor])), axis=0)

                seeds = np.concatenate((seeds, np.asarray([END_POINT])), axis=0)
                # Create Delaunay Triangulation and insert points one by one
                dt = Delaunay2D()
                for s in seeds:
                    dt.addPoint(s)
                triangles = dt.exportTriangles()
                self.draw_triangle(triangles, seeds)
                self.mid_point = list(set(self.mid_point))
                self.nodes[self.num_nodes] = START_POINT
                for midpoint in self.mid_point:
                    self.num_nodes += 1
                    self.nodes[self.num_nodes] = midpoint
                    cv2.circle(self.image, (int(midpoint[0]), int(midpoint[1])), 2, BLACK, thickness=-1, lineType=8,
                               shift=0)
                    cv2.putText(self.image, '{}'.format(self.num_nodes), (int(midpoint[0] + 2), int(midpoint[1])), cv2.FONT_HERSHEY_PLAIN ,
                                FONT_SCALE, BLACK, 1, cv2.LINE_AA)
                self.nodes[self.num_nodes + 1] = END_POINT
                print(self.nodes)
                self.init_graph()
                self.dijkstra_algorithm = Graph(self.graph_edges)
                returned_path, returned_distance = self.dijkstra_algorithm.shortest_path(0, self.num_nodes + 1)
                print(returned_path, returned_distance)
                for i in range(len(returned_path) - 1):
                    cv2.line(self.image, (int(self.nodes[returned_path[i]][0]), int(self.nodes[returned_path[i]][1])),
                             (int(self.nodes[returned_path[i+1]][0]), int(self.nodes[returned_path[i+1]][1])), BLUE, thickness=2)

                # if the 'q' key is pressed, quit
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main()
    main.run()
