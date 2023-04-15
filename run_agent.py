import numpy as np
import torch
from copy import deepcopy
import traci.constants as tc
import traci
from utils import collect_step_data, get_vehicle_attr_dict_list
from networks.nets import Agent, Phi, Q, Rho
import argparse
import yaml


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")

    return arg_parser

def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf

def get_reward(action, v_ego, v_desired):
    if action == 1:
        Plc = 0
    else:
        Plc = 0.01

    reward = (1 - (np.abs(v_ego - v_desired)/v_desired)) - Plc

    return reward

def test_agent(agent, vehicle_attr_list, d_max):
    episode_data = []
    reward_list = []
    i = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()
        if i<len(vehicle_attr_list):
            v_param = vehicle_attr_list[i]

            if v_param['id']=='ego':
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setLaneChangeMode('ego', 265) # Turn off sumo auto lane change
            else:
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setMaxSpeed(v_param['id'], v_param['max_speed'])
                traci.vehicle.setParameter(v_param['id'], tc.LCA_SPEEDGAIN, v_param['ic_speed_gain']) # set the icSpeedGain of the new vehicle
        i = i+1

    
        vehicle_list = collect_step_data(traci, d_max = d_max)
        if len(vehicle_list)>0:
            episode_data.append(vehicle_list)
        
        if len(episode_data)>1 and len(vehicle_list)==0:
            break

        if 'ego' in traci.vehicle.getIDList():
            static_state = torch.Tensor([vehicle_list[0]])
            dyn_state = torch.Tensor([vehicle_list[1:]])
            v_ego = traci.vehicle.getSpeed('ego')
            v_desired = traci.vehicle.getMaxSpeed('ego')
            
            prediction = agent.forward(dyn_state, static_state).detach().cpu().numpy()[0]
            
            thresh = 0.6 # Confidance thresh
            if prediction.max()<thresh:
                a = 1
            else:
                a = np.argmax(prediction)

            lane_idx = traci.vehicle.getLaneIndex('ego')
            
            target_lane_idx = lane_idx + (a-1) 

            print("action: ", A[a])
            # print(prediction, "action: ", A[a])
            
            if target_lane_idx<3 and target_lane_idx>=0:
                traci.vehicle.changeLane('ego', target_lane_idx, duration=10)
            
            reward = get_reward(a, v_ego, v_desired)
            reward_list.append(reward)
        
    epi_return = sum(reward_list)/len(reward_list)

    return epi_return, reward_list

if __name__=="__main__":
    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    # Configuration file
    conf_file = args.conf_file
    conf = get_conf(conf_file)

    # Configurations

    sensor_range = conf['env']['sensor_range']
    weight_file = conf['run']['weight_file']
    env_conf_path = conf['run']['env_conf_path']
    num_vehicles = conf['run']['num_vehicles']

    agent = torch.load(weight_file)[0]

    A = ['right','keep_lane','left']

    traci.start(["sumo-gui", "-c", env_conf_path])

    vehicle_types = traci.vehicletype.getIDList()
    vehicle_attr_list = get_vehicle_attr_dict_list(traci, vehicle_types, num_vehicles=num_vehicles)

    epi_return, reward_list = test_agent(agent, vehicle_attr_list, d_max=sensor_range)
    print("Return: ", epi_return)
    exit()
