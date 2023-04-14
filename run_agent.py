import numpy as np
import torch
from copy import deepcopy
import traci.constants as tc
import traci
from utils import *
from networks.nets import Agent, Phi, Q, Rho


def run_episode(agent, vehicle_attr_list, d_max):
    episode_data = []
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

            
            prediction = agent.forward(dyn_state, static_state).detach().cpu().numpy()[0]
            
            thresh = 0.7 # Confidance thresh
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
        

    return np.array(episode_data, dtype=object)

agent = torch.load('weights/agent_loss_0.17_epoch_999.pt')
A = ['right','keep_lane','left']

traci.start(["sumo-gui", "-c", "data_collection/sumo_circuler_net/circle_env.sumocfg"])

vehicle_types = traci.vehicletype.getIDList()
vehicle_attr_list = get_vehicle_attr_dict_list(traci, vehicle_types, num_vehicles=30)

episode_data = run_episode(agent, vehicle_attr_list, d_max=80)

traci.close()