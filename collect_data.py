import traci
import time
import numpy as np
import traci.constants as tc
import argparse
import os
import yaml

#https://sumo.dlr.de/pydoc/traci._vehicle.html


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")

    return arg_parser

def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf

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
        lane = 0 #np.random.randint(0,num_lanes)
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
        ego_speed = traci.vehicle.getSpeed('ego')
        ego_lane = traci.vehicle.getLaneID('ego')
        vehicle_list.append(['ego', ego_pos, ego_speed, ego_lane, 0])

        for vehicle_id in traci.vehicle.getIDList():
            vehicle_pos = np.array(traci.vehicle.getPosition(vehicle_id))
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_lane = traci.vehicle.getLaneID(vehicle_id)

            rel_dis = np.sqrt(np.sum(np.power(vehicle_pos - ego_pos, 2)))

            if rel_dis<d_max and not(vehicle_id=="ego"):
                vehicle_list.append([vehicle_id, vehicle_pos, vehicle_speed, vehicle_lane, rel_dis])

    return vehicle_list


def run_episode(vehicle_attr_list):
    episode_data = []
    i = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()
        # time.sleep(0.00000001)
        
        
        if i<len(vehicle_attr_list):
            v_param = vehicle_attr_list[i]

            if v_param['id']=='ego':
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
            else:
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setMaxSpeed(v_param['id'], v_param['max_speed'])
                traci.vehicle.setParameter(v_param['id'], tc.LCA_SPEEDGAIN, v_param['ic_speed_gain']) # set the icSpeedGain of the new vehicle
        i = i+1

    
        vehicle_list = collect_step_data(d_max = 30)
        if len(vehicle_list)>0:
            episode_data.append(vehicle_list)
        
        if len(episode_data)>1 and len(vehicle_list)==0:
            break

    return np.array(episode_data, dtype=object)



if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    # Configuration file
    conf_file = args.conf_file
    num_episodes = 1


    dataset = []
    for epi in range(num_episodes):

        print(f'episode {epi}')
        traci.start(["sumo-gui", "-c", "data_collection/sumo_circuler_net/circle_env.sumocfg"])
    
        vehicle_types = traci.vehicletype.getIDList()
        vehicle_attr_list = get_vehicle_attr_dict_list(vehicle_types, num_vehicles=50)

        episode_data = run_episode(vehicle_attr_list)
        dataset.append(episode_data)
        traci.close()

    np.save("dataset.npy",np.array(dataset, dtype=object))

    print("Data collection is done!")
