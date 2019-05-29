from glob import glob
from pathlib import Path
import re 
import carla
import configparser

from agents.tools.enums import RoadOption
from agents.tools.misc import distance_vehicle


def get_lstm_config(path): 
    config = configparser.ConfigParser()
    config.read(path)
    steer_scale = float(config.get("ModelConfig", "scale", fallback=1.0))
    seq_length = int(config.get("ModelConfig","sequence_length",fallback=None))
    sampling_interval = int(config.get("ModelConfig","sampling_interval", fallback=None))
    model_type = config.get("ModelConfig","model", fallback=None)

    return steer_scale, seq_length, sampling_interval, model_type


def set_green_traffic_light(player):

    traffic_light_state = player.get_traffic_light_state()
    if traffic_light_state is not None and traffic_light_state ==  carla.TrafficLightState.Red: 
        traffic_light = player.get_traffic_light()
        traffic_light.set_state(carla.TrafficLightState.Green)


def get_best_models(models_path): 
    """
    input: path to folder where different models has been tested 
    return: 
        best_model_paths: a list of paths for each model - where it had the best val loss 
        model_parameter_paths: a list of paths to each model's parameter text 
    """

    best_model_paths = []
    model_parameter_paths = []
    config_paths = []

    # For each model 
    for model_path in sorted(glob(str(models_path / "*"))):
        min_val_loss = float('inf')
        min_val_loss_model_path = ""
        # For each file in model folder 
        for model_file_path in glob(str(Path(model_path) / "*.h5")):
            # Get val loss of model
            match = re.search("val(\d+\.*\d*)", model_file_path)
            if match: 
                val_loss = float(match.group(1))
            else:
                print("ERROR: in ", model_path)
                print("Validation loss was not found in the model's file name")
                return None 

            # check if this is min val loss 
            if val_loss <= min_val_loss: 
                min_val_loss = val_loss
                min_val_loss_model_path = Path(model_file_path)

        # Add best model to list 
        best_model_paths.append(min_val_loss_model_path)
        config_paths.append(model_path+"/config.ini")

        # Get txt-file of model 
        txt_paths = glob(str(Path(model_path) / "*parameters.txt"))
        if len(txt_paths) != 1: 
            print("ERROR in ", model_path)
            print("model folder should have exactly one parameters.txt file")
            return None 
        else: 
            model_parameter_paths.append(Path(txt_paths[0]))

    return best_model_paths, model_parameter_paths, config_paths
        

def get_parameter_text(path):
    f = open(str(path), "r")
    params = []
    for line in f:
        if '___' in line: 
            break
        if 'dataset' not in line and 'epochs' not in line and "batch" not in line:  
            params.append(line.replace("\n", ""))
        
    return params
    

def is_valid_lane_change(road_option, world, proximity_threshold=8): 

        if road_option != RoadOption.CHANGELANELEFT and road_option != RoadOption.CHANGELANERIGHT: 
            print("ERROR: road option is not a lane change")
            return False 
        
        player_waypoint = world.map.get_waypoint(world.player.get_location())
        player_lane_id = player_waypoint.lane_id
        player_transform = world.player.get_transform()
        player_yaw = player_transform.rotation.yaw
        
        vehicle_list = world.world.get_actors().filter('vehicle.*')     

        # Lane change left  
        if road_option == RoadOption.CHANGELANELEFT: 
            # check if lane change is valid 
            if not player_waypoint.lane_change & carla.LaneChange.Left:
                #print("Traffic rules does not allow left lane change here")
                return False  

            # Only look at vehicles in left adjecent lane 
            for vehicle in vehicle_list: 
                #vehicle = actorvehicle_list_list.find(vehicle_id)
                vehicle_lane_id = world.map.get_waypoint(vehicle.get_location()).lane_id
                # Check if lane_id is in the same driving direction 
                if (player_lane_id < 0 and vehicle_lane_id <0) or (player_lane_id > 0 and vehicle_lane_id > 0):
                    if abs(player_lane_id)-abs(vehicle_lane_id) == 1: 
                        # check the vehicle|s proximity to the player
                        vehicle_waypoint = world.map.get_waypoint(vehicle.get_location())
                        if distance_vehicle(vehicle_waypoint, player_transform) < proximity_threshold:   
                            return False  
        # Lane change right 
        else: 
            # check if lane change is valid 
            if not player_waypoint.lane_change & carla.LaneChange.Right:
                #print("Traffic rules does not allow right lane change here")
                return False 
            # Only look for vehicles in right adjencent lane 
            for vehicle in vehicle_list:
                #vehicle = vehicle_list.find(vehicle_id)
                vehicle_lane_id = world.map.get_waypoint(vehicle.get_location()).lane_id
                # Check if lane_id is in the same driving direction 
                if (player_lane_id < 0 and vehicle_lane_id <0) or (player_lane_id > 0 and vehicle_lane_id > 0):
                    if abs(player_lane_id)-abs(vehicle_lane_id) == -1: 

                        # check the vehicle|s proximity to the player
                        vehicle_waypoint = world.map.get_waypoint(vehicle.get_location())
                        if distance_vehicle(vehicle_waypoint, player_transform) < proximity_threshold:   
                            return False           
        return True            
