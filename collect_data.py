import traci
import numpy as np
import traci.constants as tc
import argparse
import os
import yaml
from utils import *

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


def run_episode(vehicle_attr_list, d_max):
    episode_data = []
    i = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()

        if i<len(vehicle_attr_list):
            v_param = vehicle_attr_list[i]

            if v_param['id']=='ego':
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
            else:
                traci.vehicle.addLegacy(v_param['id'], "route_0",pos=v_param['pos'], lane=v_param['lane'], speed=0, typeID=v_param['typeID'])
                traci.vehicle.setMaxSpeed(v_param['id'], v_param['max_speed'])
                traci.vehicle.setParameter(v_param['id'], tc.LCA_SPEEDGAIN, v_param['ic_speed_gain']) # set the icSpeedGain of the new vehicle
        i = i+1
    
        vehicle_list = collect_step_data(traci=traci, d_max = d_max)
        if len(vehicle_list)>0:
            episode_data.append(vehicle_list)
        
        if len(episode_data)>1 and len(vehicle_list)==0:
            break

    return episode_data

if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    # Configuration file
    conf_file = args.conf_file
    conf = get_conf(conf_file)

    # Configurations
    sumo_conf_path = conf['dataset']['env_conf_path']
    num_episodes = conf['dataset']['num_episodes']
    num_vehicles = conf['dataset']['num_vehicles']
    dataset_folder = conf['dataset']['folder']
    sensor_range = conf['env']['sensor_range']

    dataset_path = os.path.join(dataset_folder, f'dataset_epi_{num_episodes}_num_vehicles{num_vehicles}_sensor_range_{sensor_range}.npy')
    os.makedirs(dataset_folder, exist_ok=True)

    dataset = []
    for epi in range(num_episodes):

        traci.start(["sumo", "-c", sumo_conf_path])
    
        vehicle_types = traci.vehicletype.getIDList()
        vehicle_attr_list = get_vehicle_attr_dict_list(traci, vehicle_types, num_vehicles=num_vehicles)

        episode_data = run_episode(vehicle_attr_list, d_max=sensor_range)

        print(f'epi length for epi {epi}: {len(episode_data)}')
        dataset.append(episode_data)
        traci.close()

    np.save(dataset_path, np.array(dataset, dtype=object))

    print("Data collection is done!")
