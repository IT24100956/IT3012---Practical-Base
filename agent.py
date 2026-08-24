# agent.py
from collections import deque
import heapq

class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)

class SearchAgent:
    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def sense_and_act(self, percept):
        if not self.plan:
            agent_pos = percept.get('agent_pos')
            walls = set(percept['walls'])
            all_food = percept['all_food']
            grid_size = percept['grid_size']

            if not all_food:
                return None

            current_pos = tuple(percept.get('agent_pos', (0, 0)))

            target_food = min(all_food, key=lambda f: abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1]))

            initial_state = current_pos

            def goal_test_fn(state):
                return state == tuple(target_food)

            def actions_fn(state):
                possible_moves = [
                    ('UP', (0, -1)),
                    ('DOWN', (0, 1)),
                    ('LEFT', (-1, 0)),
                    ('RIGHT', (1, 0))
                ]
                valid_actions = []
                width, height = grid_size
                for action_name, (dx, dy) in possible_moves:
                    next_x, next_y = state[0] + dx, state[1] + dy
                    if 0 <= next_x < width and 0 <= next_y < height and (next_x, next_y) not in walls:
                        valid_actions.append(action_name)
                return valid_actions

            def result_fn(state, action):
                move_map = {
                    'UP': (0, -1),
                    'DOWN': (0, 1),
                    'LEFT': (-1, 0),
                    'RIGHT': (1, 0)
                }
                dx, dy = move_map[action]
                return (state[0] + dx, state[1] + dy)

            def cost_fn(state, action, child_state):
                return 1

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(initial_state, goal_test_fn, actions_fn, result_fn)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(initial_state, goal_test_fn, actions_fn, result_fn)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(initial_state, goal_test_fn, actions_fn, result_fn, cost_fn)
            else:
                self.plan = []

            if not self.plan:
                return None

        return self.plan.pop(0)

    def bfs_search(self, initial_state, goal_test_fn, actions_fn, result_fn):
        node = {'state': initial_state, 'path': [], 'cost': 0}
        if goal_test_fn(node['state']):
            return node['path']
        
        frontier = deque([node])
        reached = {initial_state}
        
        while frontier:
            current_node = frontier.popleft()
            
            for action in actions_fn(current_node['state']):
                child_state = result_fn(current_node['state'], action)
                
                if child_state not in reached:
                    if goal_test_fn(child_state):
                        return current_node['path'] + [action]
                    
                    reached.add(child_state)
                    child_node = {
                        'state': child_state,
                        'path': current_node['path'] + [action],
                        'cost': current_node['cost'] + 1
                    }
                    frontier.append(child_node)
        return []

    def dfs_search(self, initial_state, goal_test_fn, actions_fn, result_fn):
        node = {'state': initial_state, 'path': [], 'cost': 0}
        frontier = [node]
        reached = {initial_state}
        
        while frontier:
            current_node = frontier.pop()
            
            if goal_test_fn(current_node['state']):
                return current_node['path']
            
            for action in actions_fn(current_node['state']):
                child_state = result_fn(current_node['state'], action)
                
                if child_state not in reached:
                    reached.add(child_state)
                    child_node = {
                        'state': child_state,
                        'path': current_node['path'] + [action],
                        'cost': current_node['cost'] + 1
                    }
                    frontier.append(child_node)
        return []

    def ucs_search(self, initial_state, goal_test_fn, actions_fn, result_fn, cost_fn):
        counter = 0
        initial_cost = 0
        frontier = []
        heapq.heappush(frontier, (initial_cost, counter, initial_state, []))
        reached = {initial_state: initial_cost}
        
        while frontier:
            current_cost, _, current_state, path = heapq.heappop(frontier)
            
            if goal_test_fn(current_state):
                return path
            
            if current_cost > reached.get(current_state, float('inf')):
                continue
                
            for action in actions_fn(current_state):
                child_state = result_fn(current_state, action)
                step_cost = cost_fn(current_state, action, child_state)
                new_cost = current_cost + step_cost
                
                if child_state not in reached or new_cost < reached[child_state]:
                    reached[child_state] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, child_state, path + [action]))
        return []
