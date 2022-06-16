## AI LAB
# 1. DFS (usinformed search)
## https://www.educative.io/edpresso/how-to-implement-a-breadth-first-search-in-python
```python
graph = { 'a' : ['b', 'c'],
            'b' : ['d', 'e'], 
            'c' : ['f'], 
            'd' : [], 
            'e' : ['f'], 
            'f' : []
} 
graph
visited = set()

def dfs (graph , visited , node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph[node]: 
            dfs(graph, visited , neighbour)

dfs(graph , visited, 'a')
```

# 2. Best First Search. : informed search
## https://www.educative.io/edpresso/what-are-informed-search-algorithms
```python
from queue import PriorityQueue
# store a graph in a dictionary
graph = dict()

def best_first_search(start, goal ):
    pq = PriorityQueue()
    pq.put((0, start))
    totalcost = 0
    visited = []
    while not pq.empty():
        cost, node = pq.get()
        totalcost+=cost
        visited.append(node)
        if node == goal:
            print("\nPath to reach goal : ",visited)
            return totalcost
        else:
            for n, c in graph[node].items():
                if n not in visited:
                    pq.put((c, n))
    return -1

def add_edge(x, y, cost):
    if x not in graph:
        graph[x] = dict()
    graph[x][y] = cost

add_edge('s', 'a', 3)
add_edge('s', 'b', 6)
add_edge('s', 'c', 5)

add_edge('a', 'd', 9)
add_edge('a', 'e', 8)

add_edge('b', 'f', 12)
add_edge('b', 'g', 14)

add_edge('c', 'h', 7)

add_edge('h', 'i', 5)
add_edge('h', 'j', 6)

add_edge('i', 'k', 1)
add_edge('i', 'l', 10)
add_edge('i', 'm', 2)

print(graph)

start = 's'
goal = 'i'
tc = best_first_search(start, goal)

if tc != -1:
    print(f'Found : {goal} and cost of the path is {tc}\n')
else:
    print('no path found')
```
# 3. A Star Search : infromed Search 
## https://www.educative.io/edpresso/what-are-informed-search-algorithms
## https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/a-star-algorithm
```python
tree = {'S': [['A', 1], ['B', 5], ['C', 8]],
        'A': [['S', 1], ['D', 3], ['E', 7], ['G', 9]],
        'B': [['S', 5], ['G', 4]],
        'C': [['S', 8], ['G', 5]],
        'D': [['A', 3]],
        'E': [['A', 7]]}

tree2 = {'S': [['A', 1], ['B', 2]],
         'A': [['S', 1]],
         'B': [['S', 2], ['C', 3], ['D', 4]],
         'C': [['B', 2], ['E', 5], ['F', 6]],
         'D': [['B', 4], ['G', 7]],
         'E': [['C', 5]],
         'F': [['C', 6]]
         }

heuristic = {'S': 8, 'A': 8, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}
heuristic2 = {'S': 0, 'A': 5000, 'B': 2, 'C': 3, 'D': 4, 'E': 5000, 'F': 5000, 'G': 0}

cost = {'S': 0}             # total cost for nodes visited 


def AStarSearch():
    global tree, heuristic
    closed = []             # closed nodes
    opened = [['S', 8]]     # opened nodes

    '''find the visited nodes'''
    while True:
        fn = [i[1] for i in opened]     # fn = f(n) = g(n) + h(n)
        chosen_index = fn.index(min(fn))
        node = opened[chosen_index][0]  # current node
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == 'G':        # break the loop if node G has been found
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:
                continue
            cost.update({item[0]: cost[node] + item[1]})            # add nodes to cost dictionary
            fn_node = cost[node] + heuristic[item[0]] + item[1]     # calculate f(n) of current node
            temp = [item[0], fn_node]
            opened.append(temp)                                     # store f(n) of current node in array opened

    '''find optimal sequence'''
    trace_node = 'G'                        # correct optimal tracing node, initialize as node G
    optimal_sequence = ['G']                # optimal node sequence
    for i in range(len(closed)-2, -1, -1):
        check_node = closed[i][0]           # current node
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]

            '''check whether h(s) + g(s) = f(s). If so, append current node to optimal sequence
            change the correct optimal tracing node to current node'''
            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                optimal_sequence.append(check_node)
                trace_node = check_node
    optimal_sequence.reverse()              # reverse the optimal sequence

    return closed, optimal_sequence


if __name__ == '__main__':
    visited_nodes, optimal_nodes = AStarSearch()
    print('visited nodes: ' + str(visited_nodes))
    print('optimal nodes sequence: ' + str(optimal_nodes))
    
   ```
   
# 4. Bayesian network from given data 
### feed some values :) 
##### Implement Bayesian Network
```python
def stringMod1(n, fill):
    string = bin(n).replace("0b", "").zfill(fill).replace("0","t").replace("1","0").replace("t","1")
    # print(string)
    ls = list(string)
    return ls

def stringMod2(ls):
    string = "".join(ls).replace("0","t").replace("1","0").replace("t","1")
    # print(string)
    num = int(string, 2)
    return num
bbn = {
    "Bulgary" : ["Alarm"],
    "EarthQuake" : ["Alarm"],
    "Alarm" : ["JohnCalls", "MarryCalls"],
    "JohnCalls" : [],
    "MarryCalls" : []
}

dependencyGraph = {}
for i in bbn.keys():
    dependencyGraph[i] = list()

for (i,j) in bbn.items():
    for k in j:
        if k in dependencyGraph:
            # print(dependencyGraph)
            ls = dependencyGraph[k]
            ls.append(i)
            # print(ls)
            dependencyGraph[k] = ls
        else:
            dependencyGraph[k] = [i]

probabilityGraph = {}
for i in bbn.keys():
    probabilityGraph[i] = list()

# print(dependencyGraph)

for i,j in dependencyGraph.items():
    predValues = []
    #probabilities when happen
    for k in range(2**(len(j))):
        ls = stringMod1(k, len(j)+1)
        val = " ".join(j)
        prob = round(float(input(f"Probability for {ls} {i} {val} : ")),8)
        predValues.append(prob)
    #probabilities when not happen
    for l in range(len(predValues)):
        predValues.append(round(1-predValues[l],8))

    probabilityGraph[i] = predValues

# print(probabilityGraph)
def getJointProbability(ls):
    lsLen = len(ls)
    prob = 1
    for i in range(lsLen):
        string = ""
        lsKey = ls[i][0]
        lsValue = ls[i][1]
        string += str(lsValue)
        for k in dependencyGraph[lsKey]:
            for l,m in ls:
                if l == k:
                    string += str(m)
        
        # print(string)
        index = stringMod2(list(string))
        tempprob = probabilityGraph[lsKey][index]
        prob *= tempprob
        # print(tempprob)
    return round(prob,8)

print("\nBayesian Belief Network : ")
for i,v in bbn.items():
    print(f"{i} : {v}")

print("\nDependency Graph : ")
for i,v in dependencyGraph.items():
    print(f"{i} : {v}")

print("\nProbability Graph : ")
for i,v in probabilityGraph.items():
    print(f"{i} : {v}")

joint_calc = getJointProbability([("Bulgary", 0), ("EarthQuake", 0), ("Alarm", 1), ("JohnCalls", 1), ("MarryCalls", 1)])
print(f"Joint Probability Distribution : {joint_calc}")

```


# 5. WAP to construct value and policy iteration in a grid world
## https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff#:~:text=Value%20iteration,%20just%20as%20its,initialises%20all%20reward%20as%200.
```python
import numpy as np

# global variables Rules for board.
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True

# Class to justify each state(position) of our agent,
#  giving reward according to its state.
class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (1, 1):
                        return nxtState
            return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


# Agent of player

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    ag.play(50)
    print(ag.showValues())  
    
 ```


