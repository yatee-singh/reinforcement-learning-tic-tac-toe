import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Define the neural network
policy_model = Sequential()
policy_model.add(Dense(64, input_dim=9, activation='relu'))  # First hidden layer
policy_model.add(Dense(64, activation='relu'))               # Second hidden layer
policy_model.add(Dense(9, activation='linear'))              # Output layer

# Step 2: Compile the policy_model with Mean Squared Error loss and Adam optimizer
policy_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

target_model = Sequential()
target_model.add(Dense(64, input_dim=9, activation='relu'))  # First hidden layer
target_model.add(Dense(64, activation='relu'))               # Second hidden layer
target_model.add(Dense(9, activation='linear'))              # Output layer

# Step 2: Compile the policy_model with Mean Squared Error loss and Adam optimizer
target_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
target_model.set_weights(policy_model.get_weights()) 
# Step 3: Initialize Replay Buffer and Generate Random Experience Data
replay_buffer = deque(maxlen=10000)    # Old experiences are removed when full

# Generate random experience data: (s, a, r, s', done)
np.random.seed(42)  # For reproducibility
for _ in range(10):
    state = np.random.rand(9)         # Random state (9-dimensional input)
    action = np.random.randint(0, 9)  # Random action (0 to 8)
    reward = np.random.rand()         # Random reward
    next_state = np.random.rand(9)    # Random next state
    done = np.random.choice([True, False])  # Randomly mark if episode ended
    replay_buffer.append((state, action, reward, next_state, done))

print(replay_buffer)
# Step 4: Training Parameters
BATCH_SIZE = 5  # Size of mini-batch
GAMMA = 0.99     # Discount factor
EPOCHS = 10  
NETWORK_SYNC=2    # Number of training epochs

# Step 5: Train the Neural Network Using Mini-Batches
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

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
            q_next =target_model.predict(next_state[np.newaxis, :], verbose=0)  # Shape (1, 9)

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
        if(step% NETWORK_SYNC==0):
            target_model.set_weights(policy_model.get_weights()) 




print("Training completed.")
# policy_model.save('m1.h5')
# Step 6: Test the trained policy_model with a sample input
test_input = np.random.rand(1, 9)  # A new random input
predicted_q_values = policy_model.predict(test_input, verbose=0)
print("Predicted Q-values for the test input:", predicted_q_values)
