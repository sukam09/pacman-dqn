# import pacman game 
from pacman import Directions
from pacmanUtils import *
from game import Agent
import game

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import other libraries
import random
import numpy as np
import time
from collections import deque

# model parameters
DISCOUNT_RATE = 0.95        # discount factor
LEARNING_RATE = 0.0005      # learning rate parameter
REPLAY_MEMORY = 50000       # Replay buffer 의 최대 크기
LEARNING_STARTS = 300 	    # 300 스텝 이후 training 시작
TARGET_UPDATE_ITER = 400   # update target network

EPSILON_START = 0.8

    
class PacmanDQN(PacmanUtils):
    def __init__(self, args):        
        print("Started Pacman DQN algorithm")
        #print(args)
        self.double = args['double']
        self.multistep = args['multistep']
        self.n_steps = args['n_steps']

        self.trained_model = args['trained_model']
        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"
        
        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode))
        print("=" * 100)

        # Target 네트워크와 Local 네트워크, epsilon 값을 설정
        if self.trained_model:  # Test
            self.epsilon = 0
        else:                   # Train
            self.epsilon = EPSILON_START  # epsilon init value

        # statistics
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes   
        self.episode_number = 0
        self.episode_rewards =[]  
        
        self.epsilon = EPSILON_START  # epsilon init value

    
    def predict(self, state): 
        # state를 넣어 policy에 따라 action을 반환 (epsilon greedy)
        # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다.
        #print(state)
        act = np.random.randint(0, 4)  # random value between 0 and 3
        self.action = act # save action
        return act
    
    def update_epsilon(self):
        # Exploration 시 사용할 epsilon 값을 업데이트
        pass
                 
    def step(self, next_state, reward, done):
        # next_state = self.state에 self.action 을 적용하여 나온 state
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.
        if self.action is None:
            self.state = self.preprocess(next_state)
        else:
            self.next_state = self.preprocess(next_state)

            self.state = self.next_state
        
        # next
        self.episode_reward += reward
        self.steps_taken += 1
        self.steps_per_epi += 1
        self.update_epsilon()

        if(self.trained_model == False):
            self.train()
            self.update_epsilon()
            if(self.steps_taken % TARGET_UPDATE_ITER == 0):
                # UPDATING target network 
                pass
		
    def train(self):
        # replay_memory로부터 mini batch를 받아 policy를 업데이트
        if (self.steps_taken > LEARNING_STARTS):
            pass

    def reset(self):
        # 새로운 episode 시작시 불러 오는 함수.
        self.last_score = 0
        self.current_score = 0
        self.episode_reward = 0

        self.episode_number += 1
        self.steps_per_epi = 0
    
    def final(self, state):
        # epsidoe 종료시 불려오는 함수. 수정할 필요 없음.
        done = True
        reward = self.getScore(state)
        if reward >= 0: # not eaten by ghost when the game ends
            self.win_counter +=1

        self.step(state, reward, done)
        self.episode_rewards.append(self.episode_reward)
        win_rate = float(self.win_counter) / 500.0
        avg_reward = np.mean(np.array(self.episode_rewards))
		# print episode information
        if(self.episode_number%500 == 0):
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))
            self.win_counter = 0
            self.episode_rewards= []

    def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
        result = state.getPacmanPosition()
        return result
        