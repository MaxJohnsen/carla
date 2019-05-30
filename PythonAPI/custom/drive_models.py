from abc import ABC, abstractmethod
import cv2
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.backend import set_session
import tensorflow.keras.losses
import re
from scipy.optimize import curve_fit

class ModelInterface(ABC):
    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_prediction(self, images, info):
        pass


def encoder(x, angle):
    return np.sin(((2*np.pi*(x-1))/(9))-((angle*np.pi)/(2*1*70)))

class CNNKeras(ModelInterface):
    def __init__(self):
        self._model = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        set_session(sess)
        self._steer_scale = 1.0
        self.hlc_one_hot = { 1: [1,0,0,0,0,0], 2:[0,1,0,0,0,0], 3:[0,0,1,0,0,0], 4:[0,0,0,1,0,0], 5:[0,0,0,0,1,0], 6:[0,0,0,0,0,1]}

    
    def load_model(self, path):
        self._model = tf.keras.models.load_model(path, compile=False)
        match = re.search("scale(\d+\.*\d*)", str(path))
        if match: 
            self._steer_scale = float(match.group(1))

        print("CNN drive model loaded with steer scale ", self._steer_scale)


    def get_prediction(self, images, info):
        if self._model is None:
            return False

        img_center = cv2.cvtColor(images["forward_center_rgb"],
                                 cv2.COLOR_BGR2LAB)
        img_left = cv2.cvtColor(images["left_center_rgb"],
                                 cv2.COLOR_BGR2LAB)
        img_right = cv2.cvtColor(images["right_center_rgb"],
                                 cv2.COLOR_BGR2LAB)
        info_input = [
            float(info["speed"] * 3.6 / 100),
            float(info["speed_limit"] * 3.6 / 100),
            int(info["traffic_light"])
        ]

        hlc_input = self.hlc_one_hot[info["hlc"].value]

        prediction = self._model.predict({
            "forward_image_input": np.array([img_center]),
            "info_input": np.array([info_input]),
        })
        steer, throttle, brake = prediction[0][0], prediction[1][0], prediction[2][0]
        #steer = prediction[0]
        steer_curve_parameters = curve_fit(encoder, np.arange(1, 11, 1), steer)[0]

        steer_angle = steer_curve_parameters[0]

        step_brake = 1 if brake > 0.5 else 0

        return (steer_angle, throttle, step_brake)  

class LSTMKeras(ModelInterface):
    def __init__(self, seq_length, sampling_interval, capture_rate=3, late_hlc=False):
        self._model = None
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        set_session(sess)

        self._late_hlc = late_hlc

        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = [] 
        self._hlc_history = []

        self._seq_length = seq_length
        self._sampling_interval = sampling_interval + capture_rate - 1
        self.hlc_one_hot = { 1: [1,0,0,0,0,0], 2:[0,1,0,0,0,0], 3:[0,0,1,0,0,0], 4:[0,0,0,1,0,0], 5:[0,0,0,0,1,0], 6:[0,0,0,0,0,1]}

    def _init_history(self):
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = [] 
        self._hlc_history = []
    
    def load_model(self, path):
        self._model = tf.keras.models.load_model(path, compile=False)
        match = re.search("scale(\d+\.*\d*)", str(path))
        if match: 
            self._steer_scale = float(match.group(1))
        self._init_history()

        print("LSTM drive model loaded with steer scale ", self._steer_scale)


    def get_prediction(self, images, info):
        if self._model is None:
            return False

        req = (self._seq_length - 1) * (self._sampling_interval + 1) + 1

        img_center = cv2.cvtColor(images["forward_center_rgb"], cv2.COLOR_BGR2LAB)
        img_left = cv2.cvtColor(images["left_center_rgb"], cv2.COLOR_BGR2LAB)
        img_right = cv2.cvtColor(images["right_center_rgb"], cv2.COLOR_BGR2LAB)
        info_input = [
            float(info["speed"] * 3.6 / 100),
            float(info["speed_limit"] * 3.6 / 100),
            int(info["traffic_light"])
        ]
        hlc_input = self.hlc_one_hot[(info["hlc"].value)]

        self._img_center_history.append(np.array(img_center))
        self._img_left_history.append(np.array(img_left))
        self._img_right_history.append(np.array(img_right))
        self._info_history.append(np.array(info_input))
        self._hlc_history.append(np.array(hlc_input))
        print(hlc_input)
        if len(self._img_center_history) > req:
            self._img_center_history.pop(0)
            self._img_left_history.pop(0)
            self._img_right_history.pop(0)
            self._info_history.pop(0)
            self._hlc_history.pop(0)
                
        if len(self._img_center_history) == req:
            imgs_center = np.array([self._img_center_history[0::self._sampling_interval + 1]])
            imgs_left = np.array([self._img_left_history[0::self._sampling_interval + 1]])
            imgs_right = np.array([self._img_right_history[0::self._sampling_interval + 1]])

            infos = np.array([self._info_history[0::self._sampling_interval + 1]])
            hlcs = np.array([self._hlc_history[0::self._sampling_interval + 1]])

            prediction = self._model.predict({
                "image_center_input": imgs_center,
                "image_left_input": imgs_left,
                "image_right_input": imgs_right,
                "info_input": infos,
                "hlc_input": hlcs
            })

            """if info["hlc"].value == 4:
                prediction = prediction[0]
            elif info["hlc"].value == 5:
                prediction = prediction[1]
            elif info["hlc"].value == 6:
                prediction = prediction[2]"""
            prediction = prediction[0]
            throttle = prediction[0]
            steer = prediction[1] / self._steer_scale
            brake = prediction[2]

            # print(brake)

            return (steer, throttle, brake)
        return (0, 0, 0)