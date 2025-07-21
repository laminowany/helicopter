import math
import random
from enum import Enum

class TextureType(Enum):
    TRIANGLE = 1 # trójkąt równoboczny
    CROSS = 2 # krzyż
    CROSS_WITH_CENTER = 3 # krzyż ze środkiem

class Noise:
    def __init__(self, pos_std, yaw_std = 0):
        self.pos_std = pos_std
        self.yaw_std = yaw_std
class Pose:
    def __init__(self, x, y, z, yaw = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

PLATFORM_POSITION = Pose(0.0, 0.0, 0.0)
class Problem:
    """
    Klasa reprezentujaca cały problem lądowania helikopterem na platformie

    Zawiera funkcje sprawdzające czy helikopter już wylądował oraz generuje obserwacje platformy.
    """
    def landed(position: Pose, accuracy) -> bool:
        dx = PLATFORM_POSITION.x - position.x
        dy = PLATFORM_POSITION.y - position.y
        dz = PLATFORM_POSITION.z - position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        epsilon = 1e-6
        return distance < accuracy + epsilon
    
    def calculate_move(position: Pose, step_size) -> Pose:
        """
        Kalkulacja następnego ruchu w stronę platformy o maksymalnej zadanej długości kroku.
        """
        dx = PLATFORM_POSITION.x - position.x
        dy = PLATFORM_POSITION.y - position.y
        dz = PLATFORM_POSITION.z - position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance == 0:
            return Pose(
                0, 0, 0, position.yaw
            )
        step = min(step_size, distance)
        direction_x = dx / distance
        direction_y = dy / distance
        direction_z = dz / distance
        move_x = direction_x * step
        move_y = direction_y * step
        move_z = direction_z * step
        yaw = math.atan2(dy, dx)

        return Pose(
            move_x, move_y, move_z, yaw
        )
    
    def rotate_point(x, y, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cos_a * x - sin_a * y, sin_a * x + cos_a * y

    def get_observation(position: Pose, texture: TextureType, noise: Noise = None) -> list[Pose]:
        """
        Zwraca obserwacje, a więc punkty charakterystyczne platformy (x,y,z) w kordynatach globalnych.
        """
        if texture == TextureType.TRIANGLE:
            r = 1
            z = 0
            angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
            platform_points = [(r * math.cos(a), r * math.sin(a), z) for a in angles]
        elif texture == TextureType.CROSS:
            platform_points = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)]
        elif texture == TextureType.CROSS_WITH_CENTER:
            platform_points = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
        observations = []
        for px, py, pz in platform_points:
            dx = position.x - px
            dy = position.y - py
            dz = position.z - pz
            if noise is not None:
                dx += random.gauss(0, noise.pos_std)
                dy += random.gauss(0, noise.pos_std)
                dz += random.gauss(0, noise.pos_std)
            observations.append(Pose(dx, dy, dz))
        return observations

    def observation_to_camera_coordinates(fixed_yaw, points: list[Pose])  -> list[Pose]:
        """
        Transformuje punkty w przestrzeni 3D to perspektywy kamery.
        """
        observations = []
        for p in points:
            dx = -p.x
            dy = -p.y
            dz = -p.z
            rx, ry = Problem.rotate_point(dx, dy, -fixed_yaw)
            observations.append(Pose(rx, ry, dz))
        return observations

class Helicopter:
    """
    Reprezentuje helikopter w symulacji lądowania.

    Atrybuty:
        x (float): Współrzędna X pozycji helikoptera.
        y (float): Współrzędna Y pozycji helikoptera.
        z (float): Wysokość helikoptera nad platformą.
        yaw (float): Kąt obrotu helikoptera (w radianach).
        id (int): Unikalny identyfikator helikoptera.
        distance_traveled (float): Pokonana odległość.
    """
    id_counter = 0

    def __init__(self, x=None, y=None, z=None, yaw=None):
        def rand_side():
            return random.uniform(-100.0, -90.0) if random.random() < 0.5 else random.uniform(90.0, 100.0)
        self.x = x if x is not None else rand_side()
        self.y = y if y is not None else rand_side()
        self.z = z if z is not None else random.uniform(50.0, 60.0)
        self.yaw = yaw if yaw is not None else random.uniform(-math.pi, math.pi)
        self.distance_traveled = 0.0
        self.id = Helicopter.id_counter
        Helicopter.id_counter += 1

    def move(self, move: Pose, noise: Noise):
        """
        Realizuje ruch helikopterem z zadanym szumem.
        """
        self.yaw = random.gauss(move.yaw, noise.yaw_std)
        xy_move_length = math.sqrt(move.x**2 + move.y**2)
        dx = random.gauss(math.cos(self.yaw) * xy_move_length, noise.pos_std)
        dy = random.gauss(math.sin(self.yaw) * xy_move_length, noise.pos_std)
        dz = random.gauss(move.z, noise.pos_std)
        self.x += dx
        self.y += dy
        self.z += dz
        self.distance_traveled += math.sqrt(dx**2 + dy**2 + dz**2)
        
    def position(self) -> Pose:
        return Pose(self.x, self.y, self.z, self.yaw)
    
    def shortest_distance_to_platform(self):
        return math.sqrt((self.x - PLATFORM_POSITION.x)**2 + 
                         (self.y - PLATFORM_POSITION.y)**2 + 
                         (self.z - PLATFORM_POSITION.z)**2)
    


