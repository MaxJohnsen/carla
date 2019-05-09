from enum import Enum

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class ControlType(Enum):
    MANUAL = 0
    CLIENT_AP = 1
    SERVER_AP = 2
    DRIVE_MODEL = 3

class Enviornment(Enum):
    VOID = -1
    HIGHWAY = 0

    