from abc import ABC, abstractmethod
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import set_session


class ModelInterface(ABC):
    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_prediction(self, images, info):
        pass


class CNNKeras(ModelInterface):
    def __init__(self):
        self._model = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        set_session(sess)

    def load_model(self, path):
        print("drive model: " + path)
        self._model = tf.keras.models.load_model(path)
        print("Drive model loaded")

    def get_prediction(self, images, info):
        if self._model is None:
            return False

        img_input = cv2.cvtColor(images["forward_center_rgb"],
                                 cv2.COLOR_BGR2LAB)

        info_input = [
            float(info["speed"] * 3.6 / 100),
            float(info["speed_limit"] * 3.6 / 100),
            int(info["traffic_light"])
        ]

        # hlc_input = self._one_hot_hlc[int(info["hlc"])]

        prediction = self._model.predict({
            "image_input": np.array([img_input]),
            "info_input": np.array([info_input])
        })

        prediction = prediction[0]
        throttle = prediction[0]
        steer = prediction[1]
        brake = prediction[2]
        return (steer, throttle, brake)


class LSTMKeras(ModelInterface):
    def __init__(self, seq_length, seq_space, late_hlc=False):
        self._model = None
        self._img_history = []
        self._info_history = []
        self._hlc_history = []
        self._late_hlc = late_hlc
        self._seq_length = seq_length
        self._seq_space = seq_space
        self._one_hot_hlc = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                             [0, 0, 0, 1]]

    def load_model(self, path):
        self._model = tf.keras.models.load_model(path)

    def get_prediction(self, images, info):
        if self._model is None:
            return False
        img_input = cv2.cvtColor(images["rgb_center"], cv2.COLOR_BGR2LAB)
        info_input = [
            info["speed"] / 100,
            info["speed_limit"] / 100,
            1 if info["traffic_light"] == 2 else 0,
        ]
        hlc_input = self._one_hot_hlc[int(info["hlc"])]

        self._img_history.append(np.array(img_input))
        self._info_history.append(np.array(info_input))
        self._hlc_history.append(np.array(hlc_input))

        req = (self._seq_length - 1) * (self._seq_space + 1)
        if len(self._img_history) > req:
            start = len(self._img_history) - 1 - req
            imgs = np.array([self._img_history[start::self._seq_space + 1]])
            infos = np.array([self._info_history[start::self._seq_space + 1]])

            if self._late_hlc:
                hlcs = np.array([self._hlc_history[-1]])
            else:
                hlcs = np.array(
                    [self._hlc_history[start::self._seq_space + 1]])

            prediction = self._model.predict({
                "image_input": imgs,
                "info_input": infos,
                "hlc_input": hlcs
            })
            prediction = prediction[0]
            steer = prediction[0]
            throttle = prediction[1]
            brake = prediction[2]
            return (steer, throttle, brake)
        return (0, 0, 0)