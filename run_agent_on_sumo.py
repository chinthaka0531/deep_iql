import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import traci.constants as tc
import traci

A = ['right','keep_lane','left']
class Phi(nn.Module):

    def __init__(self):
        super(Phi, self).__init__()
        self.fc1 = nn.Linear(3, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 80)
        self.relu = nn.ReLU()

    def forward(self, X_dyn):
        Phi_out = torch.zeros((len(X_dyn), self.fc3.out_features))
        for i, sample in enumerate(X_dyn):
            for x_dyn in sample:
                x = self.fc1(x_dyn)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                x = self.relu(x)
                Phi_out[i] = Phi_out[i]+x

        return Phi_out

class Rho(nn.Module):

    def __init__(self):
        super(Rho, self).__init__()
        self.fc1 = nn.Linear(80, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 40)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(43, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.phi = Phi()
        self.rho = Rho()
        self.Q = Q()

    def forward(self, dyn, static):
        static= torch.Tensor(static)
        x = self.phi(dyn)
        x = self.rho(x)
        x = torch.cat((x, static), dim=1)
        x = self.Q(x)

        return x

def get_vehicle_attr_dict_list(vehicle_types, num_vehicles=50, num_lanes = 3):
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

        # print(vehicle_attr_list[-1])

    return vehicle_attr_list


def collect_step_data(d_max):

    vehicle_list = []
    if 'ego' in traci.vehicle.getIDList():
        ego_pos = np.array(traci.vehicle.getPosition('ego'))
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
            vehicle_pos = np.array(traci.vehicle.getPosition(vehicle_id))
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_lane = int(traci.vehicle.getLaneID(vehicle_id)[-1])

            # dr = (vehicle_pos-ego_pos)/d_max
            dr = (traci.vehicle.getDistance(vehicle_id)-traci.vehicle.getDistance('ego'))/d_max
            dv = (vehicle_speed-v_ego)/(v_ego+0.001)
            dl = vehicle_lane - ego_lane

            if np.abs(dr)<1 and not(vehicle_id=="ego"):
                
                vehicle_list.append([dv, dr, dl])

    return np.array(vehicle_list)


def run_episode(agent, vehicle_attr_list, d_max):
    episode_data = []
    i = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()
        # time.sleep(0.00000001)
        
        
        if i<len(vehicle_attr_list):
            v_param = vehicle_attr_list[i]

            if v_param['id']=='ego':
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setLaneChangeMode('ego', 265)
            else:
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setMaxSpeed(v_param['id'], v_param['max_speed'])
                traci.vehicle.setParameter(v_param['id'], tc.LCA_SPEEDGAIN, v_param['ic_speed_gain']) # set the icSpeedGain of the new vehicle
        i = i+1

    
        vehicle_list = collect_step_data(d_max = d_max)
        if len(vehicle_list)>0:
            episode_data.append(vehicle_list)
        
        if len(episode_data)>1 and len(vehicle_list)==0:
            break

        if 'ego' in traci.vehicle.getIDList():
            static_state = torch.Tensor([vehicle_list[0]])
            dyn_state = torch.Tensor([vehicle_list[1:]])

            
            prediction = agent.forward(dyn_state, static_state).detach().cpu().numpy()[0]
            
            a = np.argmax(prediction)
            lane_idx = traci.vehicle.getLaneIndex('ego')
            
            target_lane_idx = lane_idx + (a-1) 

            print("action: ", A[a])
            
            if target_lane_idx<3 and target_lane_idx>=0:
                traci.vehicle.changeLane('ego', target_lane_idx, duration=10)
        

    return np.array(episode_data, dtype=object)

agent = torch.load('agent_loss_0.17_epoch_999.pt')

traci.start(["sumo-gui", "-c", "data_collection/sumo_circuler_net/circle_env.sumocfg"])

vehicle_types = traci.vehicletype.getIDList()
vehicle_attr_list = get_vehicle_attr_dict_list(vehicle_types, num_vehicles=40)

episode_data = run_episode(agent, vehicle_attr_list, d_max=80)

traci.close()