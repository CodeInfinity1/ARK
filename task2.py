import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import math
from typing import List, Tuple, Set
import pygame
from pygame import gfxdraw

class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class PRMPlanner:
    def __init__(self, map_img: np.ndarray, num_nodes: int = 1000, connection_radius: float = 50):
        self.map_img = map_img
        self.height, self.width = map_img.shape[:2]
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
        self.nodes: List[Node] = []
        self.start = None
        self.goal = None
        
    def is_valid_point(self, x: float, y: float) -> bool:
        """Check if a point is within the map bounds and not in an obstacle."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.map_img[int(y), int(x)] > 0  # Assuming white (255) is free space
        
    def is_valid_edge(self, node1: Node, node2: Node) -> bool:
        """Check if the edge between two nodes is collision-free."""
        # Bresenham's line algorithm for checking collision
        x1, y1 = int(node1.x), int(node1.y)
        x2, y2 = int(node2.x), int(node2.y)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            if not self.is_valid_point(x, y):
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True
        
    def generate_random_node(self) -> Node:
        """Generate a random valid node in the configuration space."""
        while True:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            if self.is_valid_point(x, y):
                return Node(x, y)
                
    def build_roadmap(self):
        """Build the PRM roadmap."""
        # Add start and goal nodes
        self.nodes = [self.start, self.goal]
        
        # Generate random nodes
        for _ in range(self.num_nodes):
            self.nodes.append(self.generate_random_node())
            
        # Build KD-tree for efficient nearest neighbor search
        points = np.array([[node.x, node.y] for node in self.nodes])
        tree = KDTree(points)
        
        # Connect nodes
        for i, node in enumerate(self.nodes):
            # Find nearest neighbors within connection radius
            indices = tree.query_ball_point([node.x, node.y], self.connection_radius)
            
            for j in indices:
                if i != j:  # Don't connect to self
                    neighbor = self.nodes[j]
                    if self.is_valid_edge(node, neighbor):
                        # Add edge to the graph (implicitly through the KDTree)
                        pass
                        
    def find_path(self) -> List[Node]:
        """Find a path from start to goal using A*."""
        if not self.start or not self.goal:
            return []
            
        # Initialize open and closed sets
        open_set = {self.start}
        closed_set = set()
        self.start.cost = 0
        self.start.parent = None
        
        while open_set:
            # Find node with lowest cost in open set
            current = min(open_set, key=lambda n: n.cost + self.heuristic(n, self.goal))
            
            if current == self.goal:
                return self.reconstruct_path(current)
                
            open_set.remove(current)
            closed_set.add(current)
            
            # Find neighbors using KD-tree
            points = np.array([[node.x, node.y] for node in self.nodes])
            tree = KDTree(points)
            current_point = np.array([current.x, current.y])
            indices = tree.query_ball_point(current_point, self.connection_radius)
            
            for i in indices:
                neighbor = self.nodes[i]
                if neighbor in closed_set:
                    continue
                    
                if self.is_valid_edge(current, neighbor):
                    tentative_cost = current.cost + self.distance(current, neighbor)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        neighbor.cost = tentative_cost
                        neighbor.parent = current
                    elif tentative_cost < neighbor.cost:
                        neighbor.cost = tentative_cost
                        neighbor.parent = current
                        
        return []  # No path found
        
    def heuristic(self, node1: Node, node2: Node) -> float:
        """Calculate heuristic (Euclidean distance) between two nodes."""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
        
    def distance(self, node1: Node, node2: Node) -> float:
        """Calculate actual distance between two nodes."""
        return self.heuristic(node1, node2)
        
    def reconstruct_path(self, goal: Node) -> List[Node]:
        """Reconstruct the path from goal to start."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]
        
    def visualize(self, path: List[Node] = None):
        """Visualize the PRM roadmap and path."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map_img, cmap='gray')
        
        # Plot all nodes
        for node in self.nodes:
            plt.plot(node.x, node.y, 'b.', markersize=2)
            
        # Plot start and goal
        if self.start:
            plt.plot(self.start.x, self.start.y, 'g*', markersize=10, label='Start')
        if self.goal:
            plt.plot(self.goal.x, self.goal.y, 'r*', markersize=10, label='Goal')
            
        # Plot path if provided
        if path:
            path_x = [node.x for node in path]
            path_y = [node.y for node in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
            
        plt.legend()
        plt.axis('off')
        plt.savefig('prm_result.png')
        plt.close()

def main():
    # Load the map
    map_img = cv2.imread('maze.png', cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise ValueError("Could not load maze.png")
    
    # Debug: Print map information
    print(f"Map dimensions: {map_img.shape}")
    print(f"Map value range: min={np.min(map_img)}, max={np.max(map_img)}")
    
    # Define start and goal points for both scenarios
    start_easy = Node(50, 300)  # Left side of maze
    end_easy = Node(100, 300)    # Left side of maze, below start_easy
    start_hard = Node(170, 50)  # Top middle, where "START HARD" is labeled
    end_hard = Node(400, 300)   # Bottom right, where "END HARD" is labeled
    
    # Let user choose the scenario
    print("\nChoose the scenario:")
    print("1. Start Easy - End Easy")
    print("2. Start Hard - End Hard")
    choice = input("Enter your choice (1 or 2): ")
    
    # Create PRM planner
    planner = PRMPlanner(map_img)
    
    # Set start and goal points based on user choice
    if choice == "1":
        planner.start = start_easy
        planner.goal = end_easy
        print("\nSelected: Start Easy - End Easy")
    elif choice == "2":
        planner.start = start_hard
        planner.goal = end_hard
        print("\nSelected: Start Hard - End Hard")
    else:
        print("Invalid choice. Using Start Easy - End Easy")
        planner.start = start_easy
        planner.goal = end_easy
    
    # Debug: Check if start and goal points are valid
    print(f"Start point valid: {planner.is_valid_point(planner.start.x, planner.start.y)}")
    print(f"Goal point valid: {planner.is_valid_point(planner.goal.x, planner.goal.y)}")
    
    # Build roadmap
    print("Building roadmap...")
    planner.build_roadmap()
    print(f"Number of nodes in roadmap: {len(planner.nodes)}")
    
    # Find path
    print("Finding path...")
    path = planner.find_path()
    
    if path:
        print("Path found!")
        planner.visualize(path)
    else:
        print("No path found!")
        planner.visualize()

if __name__ == "__main__":
    main() 