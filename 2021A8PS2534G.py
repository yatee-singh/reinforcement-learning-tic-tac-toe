
import datetime
import sys
import random
from TicTacToe1 import *
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.metrics import MeanSquaredError as MSE


policy_model = Sequential()
policy_model.add(Dense(64, input_dim=9, activation='relu'))  
policy_model.add(Dense(64, activation='relu'))             
policy_model.add(Dense(9, activation='linear'))           
policy_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))


class PlayerSQN_train:
    def __init__(self):
        
        pass

    def move(self, state,epsilon,policy_model):
       
        action=0
        q_values =policy_model.predict(np.array(state).reshape(1, 9), verbose=0)
       
        legal_actions=[]
        for i in range(9):
            if(state[i]==0):
                legal_actions.append(i)
            else:
                q_values[0][i]=-10000
                pass
                
        if np.random.rand() < epsilon:
            action = random.choice(legal_actions) 
        else:
            action= np.argmax(q_values)
        return action

def Train(episodes,smartness,epsilon,cnt):

    replay_buffer = deque(maxlen=10000)    

    for ep in range(episodes):
        playerSQN = PlayerSQN_train()
        state_action=[]
        game = TicTacToe(smartness,playerSQN)
        player_turn = -1  
        action=0
        state=[]
        while not game.is_full() and game.current_winner is None:
            if player_turn == -1:
                game.player1_move()
                player_turn =1
                
            else:
                action=-1
                reward=0
                valid_move = False
                while not valid_move:
                    try:
                        position = game.playerSQN.move(game.board.copy(),epsilon,policy_model)
                        if position in game.empty_positions():
                            valid_move = True
                            state = np.array(game.board.copy())
                            state_action.append([state,position])
                            
                            game.make_move(position, 1)
                            action=position
                            
                        else:
                            print("Invalid move, position already taken. Try again.")
                    except ValueError:
                        print("Invalid input, please enter a number between 1 and 9.")
                
                player_turn = -1
        last_state=np.array(game.board.copy())
            
        
        reward = game.get_reward()
        if  not (np.all(last_state ==state)):
            state_action.append([last_state,0,0])
            for i in range(len(state_action)-1):
                if(i<len(state_action)-2):
                    replay_buffer.append([state_action[i][0],state_action[i][1],0,state_action[i+1][0],False]) 
                else:
                    replay_buffer.append([state_action[i][0],state_action[i][1],reward,last_state,True]) 
        
        else:
            for i in range(len(state_action)-1):
                if(i<len(state_action)-2):
                    replay_buffer.append([state_action[i][0],state_action[i][1],0,state_action[i+1][0],False]) 
                else:
                    replay_buffer.append([state_action[i][0],state_action[i][1],reward,last_state,True])

    
    print("size of replay buffer",len(replay_buffer))

    BATCH_SIZE =65
    GAMMA = 0.99     
    EPOCHS = 5   
    LEARNING_RATE=0.001 

    print(LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}",datetime.datetime.now())
        for step in range(len(replay_buffer) // BATCH_SIZE):
            mini_batch = random.sample(replay_buffer, BATCH_SIZE)
            states = np.zeros((BATCH_SIZE, 9))
            q_values_batch = np.zeros((BATCH_SIZE, 9))

            for j, (state, action, reward, next_state, done) in enumerate(mini_batch):
                q_values = policy_model.predict(state[np.newaxis, :], verbose=0) 
                ns=np.array(next_state)
                q_next = policy_model.predict(ns[np.newaxis, :], verbose=0)  
                if done:
                    q_target = reward  
                else:
                    q_target = reward + GAMMA * np.max(q_next[0])  

                q_values[0, action] = q_target
                states[j] = state
                q_values_batch[j] = q_values

            policy_model.fit(states, q_values_batch, epochs=1, verbose=0)

    print("Training completed.")
    filename="final_EPCOH5_ep"+str(cnt) +".h5"
    policy_model.save(filename)
    
model_use = load_model('2021A8PS2534G.h5', custom_objects={'mse': MSE()})

class PlayerSQN:
    def __init__(self):
        pass

    def move(self, state):
        epsilon=0
        state = np.array(state)  
        state[state == 1] = -1  
        state[state == 2] = 1   
        action=0
        q_values =model_use.predict(np.array(state).reshape(1, 9), verbose=0)
        print(q_values)
        legal_actions=[]
        for i in range(9):
            if(state[i]==0):
                legal_actions.append(i)
            else:
                q_values[0][i]=-10000
                
        if np.random.rand() < epsilon:
            action = random.choice(legal_actions)
        else:
            action= np.argmax(q_values)
        return action


def main(smartMovePlayer1):
    
    i=1
    episodes=1000
    arr= [
      (1, 0), (0.8, 0), (0.8, .1), (0.8, .2), (0.8, .3), (0.7, .4), (0.7, .5),
        (0.6, .5), (0.6, .6), (0.5, .6), (0.5, .7), (0.4, .7), (0.4, .8), (0.4, .8),
        (0.3, .8), (0.2, .8), (0.1, .9)
    ]

    #UNCOMMENT TO TRAIN
    # for epsilon, smartness in arr:
    #     print(i)
    #     Train(episodes,smartness,epsilon,i)
    #     print(datetime.datetime.now())
    #     i+=1
       
    playerSQN = PlayerSQN()
    game = TicTacToe(smartMovePlayer1,playerSQN)
    game.play_game()
    reward = game.get_reward()
    print(f"Reward for Player 2 (You): {reward}")
    
if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0<=smartMovePlayer1<=1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1Probability>")
        print("Example: python 2020A7PS0001.py 0.5")
        print("There is an error. Probability must lie between 0 and 1.")
        sys.exit(1)
    
    main(smartMovePlayer1)