import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import yaml
from utils import *
from networks.nets import Agent, Phi, Q, Rho
import pandas as pd
import os
import time
import matplotlib.pyplot as plt


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")

    return arg_parser

def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf


if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    # Configuration file
    conf_file = args.conf_file
    conf = get_conf(conf_file)

    # Configurations

    dataset_path = conf['train']['dataset_path']
    sensor_range = conf['env']['sensor_range']

    weight_folder = conf['train']['weight_folder']
    gamma = conf['train']['gamma']
    lr = conf['train']['lr']
    tau = conf['train']['tau']
    batch_size = conf['train']['batch_size']
    freq = conf['train']['model_saving_frequency']
    n_optimization = conf['train']['n_optimization']+1
    resume_from = conf['train']['resume_from']

    weight_folder = os.path.join(weight_folder, f'run_id_{str(int(time.time()))}')
    os.makedirs(weight_folder, exist_ok=True)

    if os.path.exists(resume_from):
        print(f"Training from '{resume_from}'")
        [agent, agent_target] = torch.load(resume_from)
    else:
        print("Training from scratch..")
        agent = Agent()
        agent_target = deepcopy(agent)

    opt = optim.Adam(agent.parameters(), lr=lr)

    dataset = np.load(dataset_path, allow_pickle=True)[0:5]
    balanced_dataset = get_balanced_dataset(dataset)

    history = []
    for n in range(n_optimization):
    
        batch = sample_batch(balanced_dataset, batch_size)
        # for epi in batch:
        (dyn_s1, static_s1, dyn_s2, static_s2, action_list,reward_list) = batch# selected an episode
        # print(action_list) # 0: keep_lane | -1: right | 1:left
        action_list = action_list.numpy().astype(int)+1
        # print(action_list) # 1: keep_lane | 0: right | 2:left
        

        q_from_target_net = agent_target.forward(dyn_s2, static_s2)
        
        y = reward_list + gamma*torch.max(q_from_target_net, dim=1)[0]

        q_out_raw = agent.forward(dyn_s1, static_s1)
        # print(np.argmax(q_from_target_net.detach().cpu().numpy(),axis=1), q_out[10].detach().cpu().numpy())
        q_out = q_out_raw[torch.arange(len(action_list)), action_list]

        mse = torch.nn.functional.mse_loss(q_out, y)
        

        mse.backward()

        opt.step()

        agent_target.zero_grad()
        agent.zero_grad()

        agent_target = transfer_weights(from_net = agent, to_net=agent_target, tau = tau)
            
        print(f"loss for optimization_step {n}: ",round(float(mse.detach()), 4), '  | Sample prediction: ',q_out_raw[0].detach().cpu().numpy())
        history.append([n,float(mse.detach())])

        if n%freq==0:
            file_name = os.path.join(weight_folder, f'step_{n}_loss_{str(round(float(mse.detach()), 4))}.pt')
            models = [agent, agent_target]
            torch.save(models, file_name)

    # Saving history and plots
    csv_name = os.path.join(weight_folder, 'history.csv')
    plot_name = os.path.join(weight_folder, 'history.png')
    df = pd.DataFrame(history, columns=['optimization_step','loss'])
    df.to_csv(csv_name)
    # plot
    df.plot(x='optimization_step', y='loss', kind='line')
    plt.title('History')
    plt.grid()
    plt.savefig(plot_name)


