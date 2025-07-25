#building from scartch 2000 episodes
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
 # First hidden layer
policy_model.add(Dense(64, activation='relu'))     
policy_model.add(Dense(64, activation='relu'))            # Second hidden layer
policy_model.add(Dense(9, activation='linear'))              # Output layer

# Step 2: Compile the policy_model with Mean Squared Error loss and Adam optimizer
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
            action = random.choice(legal_actions)  # Random action
        else:
            action= np.argmax(q_values)
        return action

def Train(episodes,smartness,epsilon,cnt):

    replay_buffer = deque(maxlen=10000)    # Old experiences are removed when full

    for ep in range(episodes):
        #play game and enter the replay buffer
        playerSQN = PlayerSQN_train()
        # print("apisode",ep)
        state_action=[]
        game = TicTacToe(smartness,playerSQN)
        player_turn = -1  # Player 1 starts
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
        # print("rewards is",reward)
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

    BATCH_SIZE =75 # Size of mini-batch
    GAMMA = 0.99     # Discount factor
    EPOCHS = 5    # Number of training epochs

    # Step 5: Train the Neural Network Using Mini-Batches
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}",datetime.datetime.now())

        # Sample mini-batches from replay buffer
        for step in range(len(replay_buffer) // BATCH_SIZE):
            # Randomly sample a mini-batch of experiences
            mini_batch = random.sample(replay_buffer, BATCH_SIZE)

            # Prepare input and target batches
            states = np.zeros((BATCH_SIZE, 9))
            q_values_batch = np.zeros((BATCH_SIZE, 9))

            for j, (state, action, reward, next_state, done) in enumerate(mini_batch):
                # Predict Q-values for current state and next state
                q_values = policy_model.predict(state[np.newaxis, :], verbose=0)  # Shape (1, 9)
               
                ns=np.array(next_state)
                q_next = policy_model.predict(ns[np.newaxis, :], verbose=0)  # Shape (1, 9)

                # Compute the Q-target using Bellman equation
                if done:
                    q_target = reward  # No future rewards if the episode ended
                else:
                    q_target = reward + GAMMA * np.max(q_next[0])   # Incorrect if all actions from next state are not valid

                # Update the Q-value for the chosen action
                q_values[0, action] = q_target

                # Store the updated values in the batch
                states[j] = state
                q_values_batch[j] = q_values

            # Perform gradient descent step on the mini-batch
            policy_model.fit(states, q_values_batch, epochs=1, verbose=0)
            # if(step%NETWORK_SYNC==0):
            #     target_model.set_weights(policy_model.get_weights()) 
        


    print("Training completed.")
    filename="Extra_Layer"+str(cnt)+".h5"
    policy_model.save(filename)
    
    




    

# Step 3: Initialize Replay Buffer and Generate Random Experience Data

def main(smartMovePlayer1):
    
#    random.seed(42)
    i=1
    episodes=1000
    arr= [
      (1, 0), (0.8, 0), (0.8, .1), (0.7, .2), (0.7, .3), (0.6, .4), (0.6, .5),
        (0.6, .5), (0.6, .6), (0.5, .6), (0.5, .7), (0.4, .7), (0.4, .8), (0.4, .8),
        (0.3, .8), (0.2, .8), (0.1, .9)
    ]
    for epsilon, smartness in arr:
        print(i)
        Train(episodes,smartness,epsilon,i)
        print(datetime.datetime.now())
        i+=1
       
    # playerSQN = PlayerSQN()
    # game = TicTacToe(smartMovePlayer1,playerSQN)
    # game.play_game()
    
    # # Get and print the reward at the end of the episode
    # reward = game.get_reward()
    # print(f"Reward for Player 2 (You): {reward}")
    
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