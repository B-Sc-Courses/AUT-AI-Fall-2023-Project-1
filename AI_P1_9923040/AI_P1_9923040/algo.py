from collections import OrderedDict as OD
import numpy as np
from state import next_state, solved_state
from location import next_location
import heapq
import random
# From brackley prj
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (pri, _, item) = heapq.heappop(self.heap)
        return pri, item

    def isEmpty(self):
        return len(self.heap) != 0

    # def update(self, item, priority):
    #     # If item already in priority queue with higher priority, update its priority and rebuild the heap.
    #     # If item already in priority queue with equal or lower priority, do nothing.
    #     # If item not in priority queue, do the same thing as self.push.

    #     for index, (p, c, i) in enumerate(self.heap):
    #         if item == i and p <= priority:
    #             print("if1")
    #             break
    #         elif item == i and p > priority:
    #             print("if2")
    #             del self.heap[index]
    #             self.heap.append((priority, c, item))
    #             heapq.heapify(self.heap)
    #         else:
    #             print("if3")
    #             self.push(item, priority)
def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.
 
    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == 'Random':
        return list(np.random.randint(1, 12+1, 10))
    
    elif method == 'IDS-DFS':
        actions, exp_node = IDS(init_state)
        message = f'''actions:\n{actions},\
                \nexpanded nodes:\n{exp_node[1]},\
                \nexplored nodes:\n{exp_node[0]}'''
        print(message)
        return actions
    
    elif method == 'A*':
        actions, exp_node = A_STAR(init_state, init_location)
        message = f'''actions:\n{actions},\
                \nexpanded nodes:\n{exp_node[1]},\
                \nexplored nodes:\n{exp_node[0]}'''
        print(message)
        return actions

    elif method == 'BiBFS':
        actions, exp_node = BiBFS(init_state) 
        message = f'''actions:\n{actions},\
                \nexpanded nodes:\n{exp_node[1]},\
                \nexplored nodes:\n{exp_node[0]}'''
        print(message)
        return actions 
    
    else:
        return []


def IDS(init_state):
    depthlimit = 14 # D.L. of 2x2x2 rubik is 14 (God's No.)
    # in base alog. of IDS the depthlimit is infinity
    for depth in range(depthlimit):
        actions, exp_node = DLS(init_state, depth)
        # if actions is not None:
        if actions:
            return actions, exp_node

def DLS(init_state, depth):
    exp_node = [0, 0]
    goal_state = solved_state()
    actions = []
    now_state = init_state
    num_stack = 0
    # act = list(range(1, 13))
    # act = [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12]
    act = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    fringe  = OD({num_stack: [now_state, actions]})
    reached = set() # visited node (base in psudo code)
    while fringe:
        num_stack, [now_state, actions] = fringe.popitem()
        exp_node[0] += 1
        # if np.array_equal(now_state, goal_state):
        if (now_state == goal_state).all():
            return actions, exp_node
        ##################################################
        # Expandation
        # now_state.flags.writeable = False
        now_hash = hash(now_state.data.tobytes())
        # if not (now_hash in reached):
        #     reached.add(now_hash)
        if len(actions) < depth:
            # Uncomment here to reduce the action and children
            act = list(range(1, 13))
            exp_node[1] += 12
            if actions:
                x = actions[-1]
                no = x+6 if x<7 else x-6
                act.remove(no)
                exp_node[1] -= 1
            for action in act:
                new_state = next_state(now_state, action)
                num_stack = num_stack+1
                fringe[num_stack] = [new_state, actions+[action]]
            exp_node[1] += 12
    
    return None, None
#region
# def getChild(now_state, action):
#     act= list(range(1, 13))
#     react = {1:7,
#              2:8,
#              3:9,
#              4:10,
#              5:11,
#              6:12,
#              7:1,
#              8:2,
#              9:3,
#              10:4,
#              11:5,
#              12:6}
#     if action:
#         # print(action)
#         act.remove(react[action[-1]]) # all actions those possible
#     else :
#         pass
#     children = list()
#     for action in act:
#         children.append([next_state(now_state, act), action])
#     return children
#endregion
def hurestic(loc):
    solved_loc = {1: np.array([[0,0,0]]),
                  2: np.array([[0,0,1]]),
                  3: np.array([[0,1,0]]),
                  4: np.array([[0,1,1]]),
                  5: np.array([[1,0,0]]),
                  6: np.array([[1,0,1]]),
                  7: np.array([[1,1,0]]),
                  8: np.array([[1,1,1]])}
    h = []
    # for i in range(1, 9):
    #     h += np.sum(np.abs(solved_loc[i] - np.argwhere(loc == i)))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = loc[i, j, k]
                # h += np.sum(np.abs(solved_loc[x] - np.array([[i, j, k]])))
                # h += np.abs(solved_loc[x] - np.array([[i, j, k]]))
                # h += np.abs(np.subtract(solved_loc[x], np.array([[i, j, k]])))
                h.append(np.abs(np.subtract(solved_loc[x], np.array([[i, j, k]]))))
    return np.sum(h)/4

def A_STAR(init_state, init_location):
    goal_state = solved_state()
    exp_node = [0, 0]
    act = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    reached  = set()
    fringe = PriorityQueue()
    cost = 0
    root = []
    expanded = dict()
    root_cost = len(root)
    now_state = init_state
    now_hash = hash(now_state.data.tobytes())
    node = (now_state,init_location, now_hash, root, root_cost)
    fringe.push(node, cost)
    # expanded = {state_hash: [real_cost, len_root, root, now_state]}
    # frontier = [(real_cost, len_root, root, state_hash, now_state)]
    # hq.heapify(frontier)
    while fringe.isEmpty():
        _, (past_state,past_loc, past_hash, past_root, root_cost) = fringe.pop()
        if (past_state == goal_state).all():
            return past_root, exp_node
        if not (past_hash in reached):
            reached.add(past_hash)
            exp_node[0] += 1
            for action in act:
                exp_node[1] += 1
                new_loc = next_location(past_loc, action)
                new_state  = next_state(past_state, action)
                new_hash = new_state.data.tobytes()
                new_root = past_root + [action]
                new_root_cost = root_cost + 1
                new_cost = new_root_cost + hurestic(new_loc)
                if expanded.get(new_hash) is None:
                    expanded[new_hash] = new_cost 
                    items = (new_state,new_loc,new_hash, new_root, new_root_cost)
                    fringe.push(items, new_cost)
                elif expanded.get(new_hash) > new_cost:
                    expanded[new_hash] = new_cost
                    items = (new_state,new_loc, new_hash, new_root, new_root_cost)
                    fringe.push(items, new_cost)
                else :
                    exp_node[1] -= 1

        # print("test")
                # if child not in reached:
                # pass
        # # past_cost, len_root, past_root, past_hash, past_state = hq.heappop(frontier)
        # _, _, _, past_hash, _ = hq.heappop(frontier)
        # if past_hash in expanded.keys():
        #     past_cost, len_root, past_root, past_hash, past_state = hq.heappushpop(frontier)
        # if (now_state == goal_state).all():
        #     return past_root, exp_node
        # # state_hash = hash(now_state.data.tobytes())
        # if not (past_hash in reached):
        #     exp_node[0] += 1
        #     reached.add(state_hash)
        #     real_cost = len_root
        #     for action in act:
        #         new_loc = next_location(now_state, action)
        #         new_cost = real_cost+hurestic(new_loc)+1
        #         new_state  = next_state(now_state, action)
        #         state_hash = hash(new_state.data.tobytes())
        #         if state_hash in expanded.keys():
        #             frontier
        #         hq.heappush(frontier, 
        #                     (new_cost, 
        #                      past_root+[action], state_hash, new_state))

def BiBFS(init_state):
    exp_node = [0, 0]
    node_now = init_state
    now_hash = hash(node_now.data.tobytes()) 
    node_goal = solved_state()
    goal_hash = hash(node_goal.data.tobytes()) 
    # act = list(range(1, 13))
    # act = [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12]
    act = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    act_reversed = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    actions_dir = {now_hash: (node_now, [])}
    actions_indir = {goal_hash: (node_goal, [])}
    q_direct = [[0, now_hash]]
    heapq.heapify(q_direct)
    q_indirect = [[0, goal_hash]]
    heapq.heapify(q_indirect)
    reached_dir = set()
    reached_indir = set()
    direction = 1
    test = 0
    while len(q_direct)*len(q_indirect)>0:
        if direction==1:
            _, node = heapq.heappop(q_direct)
            if not (node in reached_dir):
                exp_node[0] += 1
                reached_dir.add(node)
                state, actions = actions_dir[node]
                for action in act:
                    exp_node[1] += 1
                    new_state = next_state(state, action)
                    new_hash = hash(new_state.data.tobytes())
                    root = actions + [action]
                    actions_dir[new_hash] = (new_state, root)
                    if new_hash in reached_indir:
                        _, rev_actions = actions_indir[new_hash]
                        actions_remaining = [act_reversed[i-1] for i in rev_actions]
                        return root + actions_remaining[::-1], exp_node
                    heapq.heappush(q_direct, [len(root), new_hash])
        else:
            _, node = heapq.heappop(q_indirect)
            if not (node in reached_indir):
                exp_node[0] += 1
                reached_indir.add(node)
                state, actions = actions_indir[node]
                for action in act:
                    exp_node[1] += 1
                    new_state = next_state(state, action)
                    new_hash = hash(new_state.data.tobytes())
                    root = actions + [action]
                    actions_indir[new_hash] = (new_state, root)
                    if new_hash in reached_dir:
                        print(test)
                        _, rev_actions = actions_dir[new_hash]
                        actions_remaining = [act_reversed[i-1] for i in root]
                        return rev_actions + actions_remaining[::-1], exp_node
                    heapq.heappush(q_indirect, [len(root), new_hash])
        direction *= -1

