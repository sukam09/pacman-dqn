# import pacman game 
from pacman import Directions
from game import Agent
import game
import numpy as np

class PacmanUtils(game.Agent):  
    def registerInitialState(self, state):
        # state를 reset.
        self.state = None
        self.next_state = None
        self.action = None
        
        self.reset()

    def getAction(self, state):
        act = self.predict(state)
        move = self.get_direction(act)
        legal = state.getLegalActions(0) # check for illegal moves
        if move not in legal:
            move = Directions.STOP
        return move
    
    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST
	
    def getScore(self, state):
        self.current_score = state.getScore()
        score = self.current_score - self.last_score
        self.last_score = self.current_score

        return score

    def observationFunction(self, state):
        # do observation
        done = False
        reward = self.getScore(state) # get reward
        self.step(state, reward, done)
        return state


