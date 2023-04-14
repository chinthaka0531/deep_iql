import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from utils import *
from networks.nets import Agent, Phi, Q, Rho



gamma = 0.99
lr = 0.0001
tau = 0.0001
batch_size = 1000

dataset = np.load('data_collection/dataset_2000_epi_d_max_80.npy', allow_pickle=True)
processed_dataset = pre_processing_data(dataset)
new_dataset = get_balanced_dataset(processed_dataset)

agent = Agent()
agent_target = deepcopy(agent)

opt = optim.Adam(agent.parameters(), lr=lr)

epochs = 1000
epoch_history = []
for epoch in range(epochs):
    epi_history = []
    batch = sample_batch(new_dataset, batch_size)
    # for epi in batch:
    (dyn_s1, static_s1, dyn_s2, static_s2, action_list,reward_list) = batch# selected an episode
    # print(action_list) # 0: keep_lane | -1: right | 1:left
    action_list = action_list.numpy().astype(int)+1
    # print(action_list) # 1: keep_lane | 0: right | 2:left
    

    q_from_target_net = agent_target.forward(dyn_s2, static_s2)
    
    y = reward_list + gamma*torch.max(q_from_target_net, dim=1)[0]

    q_out = agent.forward(dyn_s1, static_s1)
    print(np.argmax(q_from_target_net.detach().cpu().numpy(),axis=1), q_out[10].detach().cpu().numpy())
    q_out = q_out[torch.arange(len(action_list)), action_list]

    mse = torch.nn.functional.mse_loss(q_out, y)
    epi_history.append(float(mse.detach()))

    mse.backward()

    opt.step()

    agent_target.zero_grad()
    agent.zero_grad()

    agent_target = transfer_weights(from_net = agent, to_net=agent_target, tau = tau)
        
    print(f"loss for epoch {epoch}: ",sum(epi_history)/len(epi_history))
    epoch_history.append(sum(epi_history)/len(epi_history))

torch.save(agent, "test_agent.pt")
torch.save(agent_target, "test_agent_target.pt")