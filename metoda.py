import random
import math
from problem import *

class Particle:
    """
    Klasa pomocnicza która reprezentuje cząstke w filtrze.
    """
    def __init__(self, x, y, z, yaw, weight=None):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw if yaw is not None else random.uniform(-math.pi, math.pi)
        self.weight = 1.0 if weight is None else weight
    def position(self) -> Pose:
        return Pose(self.x, self.y, self.z, self.yaw)

class ParticleFilter:
    """
    Ta klasa implementuje filtr cząsteczkowy.
    """
    def __init__(self, hel, num_particles, noise: Noise):
        self.particles = []
        for _ in range(num_particles):
            x = random.gauss(hel.x, noise.pos_std)
            y = random.gauss(hel.y, noise.pos_std)
            z = random.gauss(hel.z, noise.pos_std)
            yaw = random.gauss(hel.yaw, noise.yaw_std)
            p = Particle(x, y, z, yaw)
            p.weight = 1.0 / num_particles
            self.particles.append(p)
    
    def estimate(self) -> Pose:
        total = sum(p.weight for p in self.particles)
        if total == 0:
            n = len(self.particles)
            if n == 0:
                return Pose(0.0, 0.0, 0.0, 0.0)
            x = sum(p.x for p in self.particles) / n
            y = sum(p.y for p in self.particles) / n
            z = sum(p.z for p in self.particles) / n
            yaw = sum(p.yaw for p in self.particles) / n
            return Pose(x, y, z, yaw)
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        z = sum(p.z * p.weight for p in self.particles)
        yaw = sum(p.yaw * p.weight for p in self.particles)

        return Pose(x / total, y / total, z / total, yaw / total)
    
    def predict(self, expected_move: Pose, motion_noise: Noise):
        for p in self.particles:
            p.x += random.gauss(expected_move.x, motion_noise.pos_std)
            p.y += random.gauss(expected_move.y, motion_noise.pos_std)
            p.z += random.gauss(expected_move.z, motion_noise.pos_std)
            p.yaw = random.gauss(expected_move.yaw, motion_noise.yaw_std)

    def update_weights(self, helikopter: Helicopter, actual_observation: list[Pose],
                       texture, observation_noise: Noise, motion_noise: Noise):
        for p in self.particles:
            predicted_obs = Problem.observation_to_camera_coordinates(helikopter.yaw, Problem.get_observation(p.position(), texture))
            if len(predicted_obs) != len(actual_observation):
                return 0.0
            log_weight = 0.0
            epsilon = 1e-12
            for i, (po, ao) in enumerate(zip(predicted_obs, actual_observation)):
                for val_name, po_val, ao_val, sigma in [
                    ("x", po.x, ao.x, observation_noise.pos_std),
                    ("y", po.y, ao.y, observation_noise.pos_std),
                    ("z", po.z, ao.z, observation_noise.pos_std),
                    ("yaw", p.yaw, helikopter.yaw, motion_noise.yaw_std),
                ]:
                    error = ao_val - po_val
                    sigma = max(sigma, epsilon)
                    log_prob = - (error ** 2) / (2 * sigma ** 2)
                    log_prob -= math.log(math.sqrt(2 * math.pi) * sigma)
                    log_weight += log_prob
            p.weight = math.exp(log_weight)
    
    def resample(self):
        num_particles = len(self.particles)
        weights = [p.weight for p in self.particles]
        total = sum(weights)
        if total == 0:
            return
        weights = [w / total for w in weights]
        new_particles = random.choices(self.particles, weights=weights, k=num_particles)
        self.particles = [Particle(p.x, p.y, p.z, p.yaw, 1.0 / num_particles) for p in new_particles]
