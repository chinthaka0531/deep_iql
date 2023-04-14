import numpy as np
import torch
import random
import argparse

def get_vehicle_attr_dict_list(traci, vehicle_types, num_vehicles=50, num_lanes = 3):
    vehicle_types = list(vehicle_types)
    vehicle_types.remove('ego')
    vehicle_types = [word for word in vehicle_types if not word.isupper()]

    vehicle_attr_list = []
    v_num_ego = np.clip(int(np.random.normal(num_vehicles//2, num_vehicles//2)), 1, num_vehicles-1)

    for v_id in range(num_vehicles):
        u = np.random.randint(-5,6)
        # v_id, typeID, max_speed, ic_speed_gain, pos, lane
        typeID = np.random.choice(vehicle_types)
        max_speed = traci.vehicletype.getMaxSpeed(typeID) + u
        ic_speed_gain = np.random.randint(10,21)
        pos = 0
        lane = np.random.randint(0,num_lanes)
        vehicle = {'id':str(v_id), 'typeID':typeID, 'max_speed':str(max_speed), 'ic_speed_gain':str(ic_speed_gain), 'pos':pos, 'lane':str(lane)}
        vehicle_attr_list.append(vehicle)


        if v_id == v_num_ego:
            ego_vehicle = {'id':'ego', 'typeID':'ego', 'max_speed':"", 'ic_speed_gain':"", 'pos':pos, 'lane':str(np.random.randint(0,num_lanes))}
            vehicle_attr_list.append(ego_vehicle)

    return vehicle_attr_list


def collect_step_data(traci, d_max):

    vehicle_list = []
    if 'ego' in traci.vehicle.getIDList():
        v_ego = traci.vehicle.getSpeed('ego')
        ego_lane = int(traci.vehicle.getLaneID('ego')[-1])
        if ego_lane == 0:
            left_lane_av = 1
            right_lane_av = 0
        elif ego_lane == 1:
            left_lane_av = 1
            right_lane_av = 1
        elif ego_lane == 2:
            left_lane_av = 0
            right_lane_av = 1

        vehicle_list.append([v_ego, left_lane_av, right_lane_av])

        for vehicle_id in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_lane = int(traci.vehicle.getLaneID(vehicle_id)[-1])

            # dr = (vehicle_pos-ego_pos)/d_max
            dr = (traci.vehicle.getDistance(vehicle_id)-traci.vehicle.getDistance('ego'))/d_max
            dv = (vehicle_speed-v_ego)/(v_ego+0.001)
            dl = vehicle_lane - ego_lane

            if np.abs(dr)<1 and not(vehicle_id=="ego"):
                
                vehicle_list.append([dv, dr, dl])

    return np.array(vehicle_list)


def cal_action_reward(static_state, next_static_state, v_desired):

    if static_state[1] and not(static_state[2]):
        l1 = 0
    elif static_state[1] and static_state[2]:
        l1 = 1
    elif not(static_state[1]) and static_state[2]:
        l1 = 2

    if next_static_state[1] and not(next_static_state[2]):
        l2 = 0
    elif next_static_state[1] and next_static_state[2]:
        l2 = 1
    elif not(next_static_state[1]) and next_static_state[2]:
        l2 = 2

    action = l2-l1

    v_ego = next_static_state[0]

    if action == 0:
        Plc = 0
    else:
        Plc = 0.01

    reward = (1 - (np.abs(v_ego - v_desired)/v_desired)) - Plc

    return action, reward


def transfer_weights(from_net, to_net, tau):

    from_state_dict = from_net.state_dict()
    to_state_dict = to_net.state_dict()
    
    # Scale the parameters in the state dictionary
    final_state_dict = {}
    for (k1, v1), (k2, v2) in zip(to_state_dict.items(), from_state_dict.items()):
        final_state_dict[k1] = tau* v2 + (1-tau)*v1

    
    # Load the scaled state dictionary into the destination model
    to_net.load_state_dict(final_state_dict)

    return to_net


def pre_processing_data(dataset):

    epi_list = []
    for episode in dataset:
        dyn_state_list = []
        static_state_list = [] 
        next_dyn_state_list = []
        next_static_state_list = []
        action_list = []
        reward_list = []

        for i, step in enumerate(episode[0:-1]):
            static_state_list.append(step[0])
            dyn_state_list.append(torch.Tensor(step[1:]))
            next_dyn_state_list.append(torch.Tensor(episode[i+1][1:]))
            next_static_state_list.append(episode[i+1][0])

            a,r = cal_action_reward(step[0], episode[i+1][0], v_desired=24)
            action_list.append(a)
            reward_list.append(r)


        epi_list.append((dyn_state_list, torch.Tensor(static_state_list), next_dyn_state_list, torch.Tensor(next_static_state_list), torch.Tensor(action_list),torch.Tensor(reward_list)))
        
    return epi_list


def get_balanced_dataset(dataset):

    processed_dataset = pre_processing_data(dataset)
    full_dataset_a0 = []
    full_dataset_lane_changed = []
    for epi in processed_dataset:
        (dyn_s1, static_s1, dyn_s2, static_s2, action_list,reward_list)=epi

        for dyn_s1s, static_s1s, dyn_s2s, static_s2s, action_s,reward_s in zip(dyn_s1, static_s1, dyn_s2, static_s2, action_list,reward_list):
            if action_s == 0:
                full_dataset_a0.append([dyn_s1s, static_s1s, dyn_s2s, static_s2s, action_s,reward_s])
            else:
                full_dataset_lane_changed.append([dyn_s1s, static_s1s, dyn_s2s, static_s2s, action_s,reward_s])

    num_samples_lane_changed = len(full_dataset_lane_changed)
    full_dataset_a0_small = full_dataset_a0[0:num_samples_lane_changed]

    new_dataset = full_dataset_a0_small + full_dataset_lane_changed

    return new_dataset


def sample_batch(new_dataset, batch_size):
    sample_batch = random.sample(new_dataset,batch_size)
    dyn_s1, static_s1, dyn_s2, static_s2, action_list, reward_list = [],[],[],[],[],[]
    for dyn_s1s, static_s1s, dyn_s2s, static_s2s, action_lists, reward_lists in sample_batch:
        dyn_s1.append(dyn_s1s)
        static_s1.append(list(static_s1s))
        dyn_s2.append(dyn_s2s)
        static_s2.append(list(static_s2s))
        action_list.append(action_lists)
        reward_list.append(reward_lists)
    
    return dyn_s1, torch.Tensor(static_s1), dyn_s2, torch.Tensor(static_s2), torch.Tensor(action_list), torch.Tensor(reward_list)
        



