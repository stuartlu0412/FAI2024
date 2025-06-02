# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print("newPos: ", newPos)
        #print("newFood: ", newFood.asList())
        #print("newGhostStates: ", newGhostStates)
        #print("newScaredTimes: ", newScaredTimes)

        "*** YOUR CODE HERE ***"
        
        if newFood.asList():
            closestFoodDistance = min(manhattanDistance(newPos, food) for food in newFood.asList())
            return successorGameState.getScore() + 1 / closestFoodDistance  # Reciprocal of the closest food distance
        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # get all the agents in a given states
        numMinLayer = gameState.getNumAgents() - 1
        currentDepth = 0

        def max_value(gameState: GameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            selected_action = None
            for action in gameState.getLegalActions(0):
                if (v < min_value(gameState.generateSuccessor(0, action), 1, depth)):
                    v = min_value(gameState.generateSuccessor(0, action), 1, depth)
                    selected_action = action

            if depth == 0:
                return selected_action
            else:
                return v
        
        def min_value(gameState: GameState, agentIndex, depth):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == numMinLayer:
                    v = min(v, max_value(gameState.generateSuccessor(agentIndex, action), depth + 1))
                else:
                    v = min(v, min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth))
            return v
        
        return max_value(gameState, currentDepth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numMinLayer = gameState.getNumAgents() - 1
        currentDepth = 0

        alpha = float('-inf')
        beta = float('inf')

        def max_value(gameState: GameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            selected_action = None
            for action in gameState.getLegalActions(0):
                if (v < min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)):
                    v = min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)
                    selected_action = action
                if v > beta:
                    break
                alpha = max(alpha, v)

            if depth == 0:
                return selected_action
            else:
                return v
        
        def min_value(gameState: GameState, agentIndex, depth, alpha, beta):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == numMinLayer:
                    v = min(v, max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))

                if v < alpha:
                    break
                
                beta = min(beta, v)
            return v
        
        return max_value(gameState, currentDepth, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # get all the agents in a given states
        numMinLayer = gameState.getNumAgents() - 1
        currentDepth = 0

        def max_value(gameState: GameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            selected_action = None
            for action in gameState.getLegalActions(0):
                if (v < expected_value(gameState.generateSuccessor(0, action), 1, depth)):
                    v = expected_value(gameState.generateSuccessor(0, action), 1, depth)
                    selected_action = action

            if depth == 0:
                return selected_action
            else:
                return v
        
        def expected_value(gameState: GameState, agentIndex, depth):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == numMinLayer:
                    v += max_value(gameState.generateSuccessor(agentIndex, action), depth + 1)
                else:
                    v +=expected_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
            return v
        
        return max_value(gameState, currentDepth)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # 取得必要的遊戲狀態資訊
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # 基礎得分
    score = currentGameState.getScore()

    # 食物因素
    foodList = food.asList()
    if foodList:
        # 計算所有食物與 Pacman 之間的曼哈頓距離
        foodDistances = [manhattanDistance(pos, f) for f in foodList]
        minFoodDist = min(foodDistances)
        # 靠近食物獎勵，並以倒數方式呈現（+1 避免除零）
        score += 1.0 / (minFoodDist + 1)
        # 以剩餘食物數量給予輕微懲罰，鼓勵清空所有食物
        score -= len(foodList) * 0.1

    # 鬼魂因素
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        if ghost.scaredTimer > 0:
            # 當鬼魂處於驚嚇狀態時，靠近反而是機會
            score += 1.0 / (dist + 1)
        else:
            # 正常狀態下，若距離非常接近則給予嚴重懲罰，否則以倒數方式懲罰
            if dist < 2:
                score -= 500
            else:
                score -= 1.0 / (dist + 1)

    '''
    # 膠囊因素
    if capsules:
        capsuleDistances = [manhattanDistance(pos, cap) for cap in capsules]
        minCapDist = min(capsuleDistances)
        # 鼓勵靠近膠囊（倒數獎勵）
        score += 1.0 / (minCapDist + 1)
        # 每個膠囊也給予額外獎勵，因為它們可以使鬼魂驚嚇
        score += len(capsules) * 10
    '''
    return score

# Abbreviation
better = betterEvaluationFunction
