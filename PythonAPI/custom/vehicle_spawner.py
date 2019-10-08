#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random


class VehicleSpawner(object):

    def __init__(self, client, world): 
        self.client = client
        self.world = world
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        self._spawned_vehicles = []
        self._bad_colors = [
            "255,255,255", "183,187,162", "237,237,237", 
            "134,134,134", "243,243,243", "127,130,135", 
            "109,109,109", "181,181,181", "140,140,140", 
            "181,178,124", "171,255,0", "251,241,176",
            "158,149,129", "233,216,168", "233,216,168",
            "108,109,126", "193,193,193", "227,227,227",
            "151,150,125", "206,206,206", "255,222,218",
            "211,211,211", "191,191,191"
            ]

    
    def spawn_nearby(self, hero_spawn_point_index, number_of_vehicles_min,number_of_vehicles_max, radius):

       
        number_of_vehicles = random.randint(number_of_vehicles_min,number_of_vehicles_max)
        print(number_of_vehicles)
        hero_spawn_point = self.spawn_points[hero_spawn_point_index]

        hero_x = hero_spawn_point.location.x
        hero_y = hero_spawn_point.location.y

        self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('t2')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('coupe')]


        valid_spawn_points = []
        for spawn_point in self.spawn_points:
            # Distance between spaw points 
            loc = hero_spawn_point.location
            dx = spawn_point.location.x - loc.x
            dy = spawn_point.location.y - loc.y
            distance = math.sqrt(dx * dx + dy * dy)
            min_distance = 10
            if spawn_point == hero_spawn_point or distance < min_distance: 
                continue
            if radius != 0:
                x = spawn_point.location.x
                y = spawn_point.location.y
                yaw = spawn_point.rotation.yaw
                angle_diff = hero_spawn_point.rotation.yaw - yaw 
                angle_diff = abs((angle_diff + 180) % 360 - 180)
                
                if abs(hero_x-x)<= radius and abs(hero_y-y)<=radius and angle_diff < 50: 
                    valid_spawn_points.append(spawn_point)
            else: 
                valid_spawn_points.append(spawn_point)

            
        number_of_spawn_points = len(valid_spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(valid_spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(valid_spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):

                color = "255,255,255"
                while color in self._bad_colors: 
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            

        for response in self.client.apply_batch_sync(batch):
            if response.error is not None:
                self._spawned_vehicles.append(response.actor_id)

        print('spawned %d vehicles, press Ctrl+C to exit.' % len(self._spawned_vehicles))


    def destroy_vehicles(self): 
        print('\ndestroying %d actors' % len(self._spawned_vehicles))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self._spawned_vehicles if x is not None])
        self._spawned_vehicles = []


