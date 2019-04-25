#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.
"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function
from pathlib import Path
from drive_models import CNNKeras, LSTMKeras

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import cv2
import time
import ast 
import numpy as np
import pandas as pd

# Import ConfigParser for wheel 
if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from client_autopilot import ClientAutopilot
from enums import RoadOption, Enviornment, ControlType

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b    
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_KP1
    from pygame.locals import K_KP3
    from pygame.locals import K_KP4
    from pygame.locals import K_KP5
    from pygame.locals import K_KP6
    from pygame.locals import K_KP8
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [
        x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)
    ]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, history, actor_filter, settings):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.history = history
        self.player = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._spawn_point_start = 52
        self._spawn_point_destination = 102
        self._actor_filter = actor_filter
        self._settings = self._initialize_routes(settings)
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self._current_traffic_light = 0
        self._client_ap_active = False 

    def _initialize_routes(self, settings):
        s = {}
        routes = settings.get("Carla", "Routes", fallback=[]).split()        
        routes = [ast.literal_eval(r) for r in routes]
        s["routes"] = [[int(r[0]), int(r[1])] for r in routes]

        auto_record = settings.get("Carla", "AutoRecord")
        print(auto_record)
        s["auto_record"] = True if auto_record.strip() == "Yes" else False
        print(s["auto_record"])

        
        return s

    def restart(self):
        self.history._initiate()

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0

        # Get a vehilce blueprint.
        blueprint = self.world.get_blueprint_library().filter(
            'vehicle.tesla.*')[0]

        blueprint.set_attribute('role_name', 'hero')

        if self.player is not None:
            self.destroy()
            self.player = None

        if len(self._settings["routes"])>0:
            route = self._settings["routes"].pop(0)
            self._spawn_point_start = route[0]
            self._spawn_point_destination = route[1]
        
        spawn_point = self.map.get_spawn_points()[self._spawn_point_start]

        while self.player is None:
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self._client_ap = ClientAutopilot(self.player)

        destination_point = self.map.get_spawn_points()[self._spawn_point_destination]
        self._client_ap.set_destination((destination_point.location.x,
                                destination_point.location.y,
                                destination_point.location.z))
        
        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self._client_ap, self.hud,
                                            self.history)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager._initiate_recording()
        print(self._settings["auto_record"])
        
        if self._settings["auto_record"]: 
            self.camera_manager.toggle_recording()
        actor_type = get_actor_display_name(self.player)
        # self.hud.notification(actor_type)      

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_spawn_point(self):
        self._spawn_point_start += 1
        self.restart()

    def previous_spawn_point(self):
        if self._spawn_point_start > 0: 
            self._spawn_point_start -= 1
        self.restart()

    def tick(self, clock):

        self.hud.tick(self, clock)
        self.camera_manager.tick()

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroySensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager._index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.player
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.camera_manager._destroy_sensors()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, use_steering_wheel=False, start_control_type=ControlType.MANUAL, drive_model=None):
        self._drive_model = drive_model
        self._steering_wheel_enabled = use_steering_wheel
        self._control_type = start_control_type
        self._control = carla.VehicleControl()
        world.player.set_autopilot(self._control_type==ControlType.SERVER_AP)
        self._steer_cache = 0.0
        # world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        if self._steering_wheel_enabled:
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                raise ValueError("Please Connect Just One Joystick")

            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

            self._parser = ConfigParser()
            self._parser.read('wheel_config.ini')
            self._steer_idx = int(
                self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(
                self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.camera_manager.toggle_camera()
                elif event.button == 10:
                    world.next_spawn_point()
                elif event.button == 11:
                    world.previous_spawn_point()
                elif event.button == 2:
                    world.camera_manager.toggle_recording()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == 4:
                    world.restart()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h:
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_n:
                    world.next_spawn_point()
                    world.hud.notification('Next spawn point')
                elif event.key == K_b:
                    world.previous_spawn_point()
                    world.hud.notification('Previous spawn point')
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control_type = ControlType.MANUAL
                    world.hud.notification('Control Mode: Manual')
                    world.player.set_autopilot(False)
                elif event.key == K_o:
                    if self._drive_model is not None:
                        self._control_type = ControlType.DRIVE_MODEL
                        world.hud.notification('Control Mode: Drive Model')
                        world.player.set_autopilot(False)
                elif event.key == K_p:
                    self._control_type = ControlType.CLIENT_AP
                    world.hud.notification('Control Mode: Client Autopilot')
                    world.player.set_autopilot(False)
                    world._client_ap_active = not world._client_ap_active 
                elif event.key == K_p and pygame.key.get_mods() & KMOD_SHIFT:
                    #TODO: shift + P does not work 
                    self._control_type = ControlType.SERVER_AP
                    world.player.set_autopilot(True)
                    world.hud.notification('Control Mode: Server Autopilot')

        world.history.control_type = self._control_type

        if self._control_type == ControlType.MANUAL:
            if self._steering_wheel_enabled: 
                self._parse_vehicle_wheel()
            else:
                self._parse_vehicle_keys(world, pygame.key.get_pressed(),
                                            clock.get_time())
            self._control.reverse = self._control.gear < 0


        elif self._control_type == ControlType.DRIVE_MODEL:
            self._parse_drive_model_commands(world)
        elif self._control_type == ControlType.CLIENT_AP:
            world._client_ap.set_target_speed(world.player.get_speed_limit()-10)
            self._parse_client_ap(world)
            # Change route if client AP has reached its destination
            position = world.player.get_transform().location
            destination  = world.map.get_spawn_points()[world._spawn_point_destination].location
            
            if abs(position.x-destination.x < 30) and abs(position.y-destination.y)<30:
                world.hud.notification("Route Complete")
                if world._settings["auto_record"]:
                    world.camera_manager.toggle_recording()
                if len(world._settings["routes"])>0:
                    world.restart()

        world.player.apply_control(self._control)

        

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                    range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])/3

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        

    def _parse_drive_model_commands(self, world):
        images = {}
        info = {}

        images["forward_center_rgb"] = world.history._latest_images[
            "forward_center_rgb"]

        player = world.player

        v = player.get_velocity()
        red_light = 0 if player.get_traffic_light_state(
        ) == carla.TrafficLightState.Red else 1
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        info["speed"] = speed
        info["traffic_light"] = red_light
        info["speed_limit"] = player.get_speed_limit() / 3.6
        info["hlc"] = 0
        info["environment"] = 0

        steer = 0
        throttle = 0
        brake = 0

        if images["forward_center_rgb"] is not None:
            steer, throttle, brake = self._drive_model.get_prediction(
                images, info)
            print(steer, throttle, brake)

        self._control.steer = float(steer)
        
        self._control.throttle = max(min(float(throttle), 1),0)
        

        """if speed*3.6 < 80:
            self._control.throttle = 1.0 if throttle > 0.5 else 0.0
        else:
            self._control.throttle = 0"""
        self._control.brake = 0

    def _parse_vehicle_keys(self, world, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        # Update HLC 
        if keys[K_KP1]: 
            world.history.update_hlc(RoadOption.CHANGELANELEFT.value)
        elif keys[K_KP3]: 
            world.history.update_hlc(RoadOption.CHANGELANERIGHT.value)
        elif keys[K_KP4]: 
            world.history.update_hlc(RoadOption.LEFT.value)
        elif keys[K_KP5]: 
            world.history.update_hlc(RoadOption.LANEFOLLOW.value)
        elif keys[K_KP6]: 
            world.history.update_hlc(RoadOption.RIGHT.value)
        elif keys[K_KP8]: 
            world.history.update_hlc(RoadOption.STRAIGHT.value)
        
        else: 
            world.history.update_hlc(RoadOption.LANEFOLLOW.value)


        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_client_ap(self, world):
        noise = np.random.uniform(-0.05, 0.05)
        client_autopilot_control = world._client_ap.run_step() 
        world.history.update_client_autopilot_control(client_autopilot_control)
        self._control.brake = client_autopilot_control.brake
        self._control.throttle = client_autopilot_control.throttle
        self._control.steer = client_autopilot_control.steer + noise


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q
                                     and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        self._mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(self._mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(self._mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
       
        vehicles = world.world.get_actors().filter('vehicle.*')
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_limit = world.player.get_speed_limit()
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(), '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Spawn:   % 20s' % str(world._spawn_point_start),
            'Simulation time: % 12s' %
            datetime.timedelta(seconds=int(self.simulation_time)), '',
            'Speed:   % 15.0f km/h' % speed,
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z, ''
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [('Throttle:', c.throttle, 0.0, 1.0),
                                ('Steer:', c.steer, -1.0, 1.0),
                                ('Brake:', c.brake, 0.0, 1.0),
                                ('Reverse:', c.reverse),
                                ('Hand brake:', c.hand_brake),
                                ('Manual:', c.manual_gear_shift),
                                'Gear:        %s' % {
                                    -1: 'R',
                                    0: 'N'
                                }.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [('Speed:', c.speed, 0.0, 5.556),
                                ('Jump:', c.jump)]
        self._info_text += [
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        self._info_text.append(('Speed: ', '%s/%s'%(int(speed), int(speed_limit))))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            speed = self._info_text[-1]
            for item in self._info_text[:-1]:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False,
                                          points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8),
                                           (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect,
                                         0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8),
                                                  (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border,
                                         1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6),
                                 v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8),
                                               (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True,
                                                     (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
            font = pygame.font.Font(self._mono, 32)
            surface = font.render(speed[0], True, (255, 255, 255))
            display.blit(surface, (8, self.dim[1]-200))    
            
            font = pygame.font.Font(self._mono, 60)
            surface = font.render(speed[1], True, (255, 255, 255))
            display.blit(surface, (8, self.dim[1]-150))     
            
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0],
                    0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, client_ap, hud, history):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._client_ap = client_ap
        self._hud = hud
        self._history = history
        self._recording = False 
        self._last_recorded_frame = 0
        self._camera_transforms = [
            carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))
        ]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            [
                'sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'
            ],
            [
                'sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'
            ],
            [
                'sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'
            ], ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']
        ]

        self._recording_sensors = []

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self._index = None

    def _initiate_recording(self):

        sensor_bp = self._parent.get_world().get_blueprint_library().find(
            'sensor.camera.rgb')
        sensor_bp.set_attribute('image_size_x', "350")
        sensor_bp.set_attribute('image_size_y', "160")

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(carla.Location(x=0.5, z=1.7)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "forward_center", "rgb"))
        self._recording_sensors.append(sensor)

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(carla.Location(x=0.5, y=-0.7, z=1.7)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "forward_left", "rgb"))
        self._recording_sensors.append(sensor)

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(carla.Location(x=0.5, y=0.7, z=1.7)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "forward_right", "rgb"))
        self._recording_sensors.append(sensor)

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(
                carla.Location(x=0, y=-0.5, z=1.8),
                carla.Rotation(pitch=-20, yaw=-90)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "left_center", "rgb"))
        self._recording_sensors.append(sensor)

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(
                carla.Location(x=0, y=0.5, z=1.8),
                carla.Rotation(pitch=-20, yaw=90)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "right_center", "rgb"))
        self._recording_sensors.append(sensor)

    def _destroy_sensors(self):
        for sensor in self._recording_sensors:
            sensor.destroy()

        if self.sensor is not None:
            self.sensor.destroy

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(
            self._camera_transforms)
        self.sensor.set_transform(
            self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(
                weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        if self._recording:
            self._hud.notification('Writing data to disk, please wait..')
            self._history.save_to_disk()
            self._hud.notification('Writing complete!')
        else:
            self._history._active = True

        self._recording = not self._recording
        self._hud.notification(
            'Recording %s' % ('On' if self._recording else 'Off'))

    def tick(self):
        if self._recording:
            timestamp = time.time()
            if timestamp - self._last_recorded_frame > 0.1:
                self._history.record_frame(self._parent, self._client_ap)
                self._last_recorded_frame = timestamp

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class History:
    def __init__(self, output_folder):
        self._latest_images = {}
        self._image_history = []
        self._measurements_history = []
        self._frame_number = 0
        self._output_folder = output_folder
        self._timestamp = None
        self._driving_log = None
        self._active = False
        self._latest_client_autopilot_control = None
        self.control_type = None


    def _initiate(self):

        if self._active:
            self.save_to_disk()
        
        self._driving_log = pd.DataFrame(columns=[
            "ForwardCenter", "ForwardLeft", "ForwardRight", "LeftCenter",
            "RightCenter", "Location", "Velocity", "Controls", "ClientAutopilotControls","ControlType","TrafficLight",
            "SpeedLimit", "HLC", "Environment"
        ])
        self._timestamp = time.strftime("%Y-%m-%d_%H-%M-%S",
                                        time.localtime(time.time()))

        self._latest_images = {}
        self._image_history = []
        self._measurements_history = []
        self._frame_number = 0
        self._latest_client_autopilot_control = None
        self._latest_hlc = -1 

    def update_image(self, image, position, sensor_type):
        if image.raw_data:
            img = np.reshape(np.array(image.raw_data), (160, 350, 4))[:, :, :3]
            self._latest_images[position + "_" + sensor_type] = img

    def update_client_autopilot_control(self, control): 
        self._latest_client_autopilot_control = control

    def update_hlc(self, hlc):
        self._latest_hlc = hlc

    def record_frame(self, player, client_ap):
        images = []
        self._frame_number += 1

        v = player.get_velocity()
        t = player.get_transform()
        c = player.get_control()

        if self.control_type == ControlType.CLIENT_AP:
            client_ap_c = self._latest_client_autopilot_control
            hlc = client_ap._local_planner._target_road_option.value
        else:
            client_ap_c = None
            hlc = self._latest_hlc

        for name, image in self._latest_images.items():
            images.append((name + "_%08d.png" % self._frame_number, image))

        self._image_history.append(images)

        output_path = Path(self._output_folder + '/' + self._timestamp)
        image_path = output_path / "imgs"

        red_light = 0 if player.get_traffic_light_state(
        ) == carla.TrafficLightState.Red else 1

        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        self._driving_log = self._driving_log.append(
            pd.Series([
                "imgs/forward_center_rgb_%08d.png" % self._frame_number,
                "imgs/forward_left_rgb_%08d.png" % self._frame_number,
                "imgs/forward_right_rgb_%08d.png" % self._frame_number,
                "imgs/left_center_rgb_%08d.png" % self._frame_number,
                "imgs/right_center_rgb_%08d.png" % self._frame_number,
                (t.location.x, t.location.y), 
                speed,
                (c.throttle, c.steer, c.brake), 
                (client_ap_c.throttle, client_ap_c.steer, client_ap_c.brake) if self.control_type==ControlType.CLIENT_AP else -1, 
                self.control_type.value,
                red_light,
                player.get_speed_limit() / 3.6, 
                hlc, 
                Enviornment.HIGHWAY.value
            ],
                      index=self._driving_log.columns),
            ignore_index=True)

    def save_to_disk(self):
        output_path = Path(self._output_folder + '/' + self._timestamp)
        image_path = output_path / "imgs"
        image_path.mkdir(parents=True, exist_ok=True)

        for frame in self._image_history:
            for name, image in frame:
                cv2.imwrite(str(image_path / name), image)

        csv_path = str(output_path / "driving_log.csv")
        if not os.path.isfile(csv_path):
            self._driving_log.to_csv(csv_path)
        else:
            self._driving_log.to_csv(csv_path, mode="a", header=False)

        self._active = False
        self._initiate()

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args, settings):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)

        display = pygame.display.set_mode((args.width, args.height),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        history = History(args.output)

        model = None
        if args.model is not None:
            model = CNNKeras()
            model.load_model(args.model)

        world = World(client.get_world(), hud, history, args.filter, settings)
        controller = KeyboardControl(world, use_steering_wheel=args.joystick, drive_model=model)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if world is not None:
            print("Destroying world")
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-o',
        '--output',
        metavar='O',
        dest='output',
        default='output',
        help='output-folder for recordings')
    argparser.add_argument(
        '-m',
        '--model',
        dest='model',
        default=None,
        help='model file for autonomouse driving')
    argparser.add_argument(
        '-j',
        '--joystick',
        action='store_true',
        default=False,
        help='use steering wheel to control vehicle')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    settings = ConfigParser()
    settings.read("settings.ini")

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args, settings)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
