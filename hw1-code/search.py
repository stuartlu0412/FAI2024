# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import queue
from dataclasses import dataclass, field
from typing import Any

class State():

    def __init__(self, pos, dots, path):
        self.pos = pos
        self.dots = set(dots)
        self.path = path

    def __eq__(self, value: object) -> bool:
        
        if not isinstance(value, State):
            return False
        
        return self.pos == value.pos and self.dots == value.dots

    def eatDot(self, dot):
        if dot in self.dots:
            self.dots.remove(dot)

    def isTrue(self, dot):
        return dot in self.dots

    def getPos(self):
        return self.pos

    def getDots(self):
        return self.dots

    def getPath(self):
        return self.path

    def moveNorth(self):
        self.pos = (self.pos[0] - 1, self.pos[1])
    
    def moveSouth(self):
        self.pos = (self.pos[0] + 1, self.pos[1])

    def moveEast(self):
        self.pos = (self.pos[0], self.pos[1] + 1)

    def moveWest(self):
        self.pos = (self.pos[0], self.pos[1] - 1)

    def isGoal(self):
        return len(self.dots) == 0

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

def manhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculateMean(dots: list):
    mean = (0, 0)
    for dot in dots:
        mean = (mean[0] + dot[0], mean[1] + dot[1])
    mean = (mean[0] / len(dots), mean[1] / len(dots))
    return mean

def minManhattanDistance(pos, dots):
    min = 1000000
    for dot in dots:
        distance = manhattanDistance(pos, dot)
        if distance < min:
            min = distance
    return min

def maxManhattanDistance(pos, dots):
    max = 0
    for dot in dots:
        distance = manhattanDistance(pos, dot)
        if distance > max:
            max = distance
    return max

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    fringe = queue.Queue()
    start = maze.getStart()
    #path = [start]
    parent = None
    checked = []

    start_state = State(start, maze.getObjectives(), parent)
    print(maze.getObjectives())
    fringe.put(start_state)

    while (not fringe.empty()):

        state = fringe.get()    

        if state.isGoal():
            
            print("Goal found!!!") 
            print(state.getPath())
            path = []
            while state.getPath() is not None:
                path.append(state.getPos())
            assert maze.isValidPath(state.getPath())
            return reversed(path)
            #return state.getPath()
        
        if state in checked:
            continue

        for neighbor in maze.getNeighbors(state.getPos()[0], state.getPos()[1]):

            #if neighbor not in checked:
            #checked.append(neighbor)
            newState = State(neighbor, state.getDots().copy(), state.getPos().copy())#state.getPath().copy())
            newState.eatDot(neighbor)
            #newState.getPath().append(neighbor)
            fringe.put(newState)
            print("Neighbor: ", neighbor)

        checked.append(state)

    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    fringe = queue.PriorityQueue()
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    path = [start]
    checked = []

    start_state = State(start, maze.getObjectives(), path)
    fringe.put(PrioritizedItem(manhattanDistance(start, goal) + len(path) - 1, start_state))

    while (not fringe.empty()):

        state = fringe.get().item
        #print(state.getPos())

        if state.isGoal():
            assert maze.isValidPath(state.getPath())
            print("Goal found!!!") 
            print(state.getPath())
            return state.getPath()

        for neighbor in maze.getNeighbors(state.getPos()[0], state.getPos()[1]):

            if neighbor not in checked:
                checked.append(neighbor)
                newState = State(neighbor, state.getDots().copy(), state.getPath().copy())
                newState.eatDot(neighbor)
                newState.getPath().append(neighbor)
                fringe.put(PrioritizedItem(manhattanDistance(neighbor, goal) + len(state.getPath()), newState))
                print("Neighbor: ", neighbor)

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    fringe = queue.PriorityQueue()
    start = maze.getStart()
    dots = maze.getObjectives()
    
    path = [start]
    checked = []

    start_state = State(start, maze.getObjectives(), path)
    fringe.put(PrioritizedItem(maxManhattanDistance(start_state.getPos(), start_state.getDots()) + len(path) - 1, start_state))

    while (not fringe.empty()):

        state = fringe.get().item

        state.eatDot(state.getPos())
        #print(state.getDots())

        if state.isGoal():
            assert maze.isValidPath(state.getPath())
            print("Goal found!!!") 
            print(state.getPath())
            return state.getPath()
        
        if state in checked:
            continue

        for neighbor in maze.getNeighbors(state.getPos()[0], state.getPos()[1]):

            newState = State(neighbor, state.getDots().copy(), state.getPath().copy())
            newState.eatDot(neighbor)
            newState.getPath().append(neighbor)
            fringe.put(PrioritizedItem(maxManhattanDistance(newState.getPos(), newState.getDots()) + len(state.getPath()), newState))
            #print("Neighbor: ", neighbor)

        checked.append(state)
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
