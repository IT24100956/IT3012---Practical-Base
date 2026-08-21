import math
import random
import heapq

class SearchAgent:
    def __init__(self, active_algo='AStar'):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
        self.active_algo = active_algo
        self.plan = []
        
        self.directions = {
            'Up': (0, -1),
            'Down': (0, 1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def get_heuristic(self, pos, goal, heuristic_type):
        if heuristic_type == 'euclidean':
            return self.euclidean_distance(pos, goal)
        return self.manhattan_distance(pos, goal)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        frontier = []
        reached_states = set()
        
        g_cost = 0
        h_cost = self.get_heuristic(start_pos, goal_pos, heuristic_type)
        f_cost = g_cost + h_cost
        
        heapq.heappush(frontier, (f_cost, g_cost, start_pos, []))
        
        while frontier:
            current_f, current_g, current_pos, path_taken = heapq.heappop(frontier)
            
            if current_pos == goal_pos:
                return path_taken
                
            if current_pos in reached_states:
                continue
            
            reached_states.add(current_pos)
            
            for action, (dx, dy) in self.directions.items():
                next_x, next_y = current_pos[0] + dx, current_pos[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < grid_size[0] and 
                    0 <= next_y < grid_size[1] and 
                    next_pos not in walls and 
                    next_pos not in reached_states):
                    
                    new_g = current_g + 1
                    new_h = self.get_heuristic(next_pos, goal_pos, heuristic_type)
                    new_f = new_g + new_h
                    new_path = path_taken + [action]
                    
                    heapq.heappush(frontier, (new_f, new_g, next_pos, new_path))
                    
        return []

    def sense_and_act(self, percept: dict) -> str:
        pos = percept['agent_pos']
        remaining_food = percept.get('remaining_food', [])
        walls = percept.get('walls', set())
        grid_size = percept.get('grid_size', (10, 10)) 
        
        if not self.plan and remaining_food:
            closest_food = min(remaining_food, key=lambda f: self.manhattan_distance(pos, f))
            
            if self.active_algo == 'AStar':
                self.plan = self.astar_search(pos, closest_food, walls, grid_size, heuristic_type='manhattan')
                
        if self.plan:
            return self.plan.pop(0)
            
        return random.choice(self.actions_pool)
