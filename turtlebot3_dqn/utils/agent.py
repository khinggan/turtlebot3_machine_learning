import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.models import ReplayMemory, DQN, Transition
# If you want to use CUDA. But, make sure all machines has CUDA compatibility. Otherwise, use cpu
# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using {device} device")

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.global_step = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025

        self.epsilon = 1.0
        self.epsilon_start = 0.95
        self.epsilon_decay = 10000
        self.epsilon_end = 0.05

        # self.epsilon = 1.0
        # self.epsilon_decay = 0.99
        # self.epsilon_min = 0.05

        self.target_update = 300

        self.batch_size = 128
        self.train_start = 128
        self.memory = ReplayMemory(10000)

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.updateTargetModel()

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-06)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            # return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_value_output = self.model(state)
                self.q_value = q_value_output.max(1).indices.view(1, 1).item()
                return self.q_value
            # q_value = self.model(state.reshape(1, len(state)))
            # self.q_value = q_value
            # return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def trainModel(self):
        if self.memory.__len__() < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # state_batch = torch.cat(torch.from_numpy(np.array(batch.state)))
        # action_batch = torch.cat(torch.from_numpy(np.array(batch.action)))
        # reward_batch = torch.cat(torch.from_numpy(np.array(batch.reward)))
        # next_state_batch = torch.cat(torch.from_numpy(np.array(batch.next_state)))

        state_batch = torch.from_numpy(np.array(batch.state)).to(device=device, dtype=torch.float32)
        action_batch = torch.from_numpy(np.array(batch.action)).to(device=device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.from_numpy(np.array(batch.reward)).to(device=device, dtype=torch.float32)
        next_state_batch = torch.from_numpy(np.array(batch.next_state)).to(device=device, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        model_output = self.model(state_batch)
        state_action_values = model_output.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_model(next_state_batch).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # unsqueeze(1): change to vertical vector

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

