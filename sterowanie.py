import math
from metoda import *
from problem import *
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

NUM_PARTICLES = 100
ITERATIONS = 10
STEP_SIZE = 1.0
OBSERVATION_NOISE = Noise(pos_std = 0.1)
MOTION_NOISE = Noise(pos_std = 0.1, yaw_std=0.1)
PARTICLE_FITLER_INIT_NOISE = Noise(pos_std = 0.5, yaw_std=0.1)

class Statistics:
    def __init__(self):
        self.data = defaultdict(list) 
        self.observed_positions = {}
        self.estimated_positions = {}
        self.true_positions = {}
        self.file = open('raport.txt', "w")

    def __del__(self):
        self.file.close()    

    def record_single_flight(self, texture: TextureType, shortest_distance, 
                             distance_traveled, landed: bool):
        self.data[texture].append((shortest_distance, distance_traveled, landed))

    def record_observation(self, helicopter, observation, estimated_state):
        if helicopter.id not in self.observed_positions:
            self.observed_positions[helicopter.id] = []
            self.estimated_positions[helicopter.id] = []
            self.true_positions[helicopter.id] = []  
        avg_obs = (
            sum(p.x for p in observation) / len(observation),
            sum(p.y for p in observation) / len(observation),
            sum(p.z for p in observation) / len(observation)
        )

        self.observed_positions[helicopter.id].append(avg_obs)
        self.estimated_positions[helicopter.id].append((estimated_state.x, estimated_state.y, estimated_state.z))
        self.true_positions[helicopter.id].append((helicopter.x, helicopter.y, helicopter.z))

    def print_summary(self):
        for texture, flights in self.data.items():
            n = len(flights)
            total_landed = sum(1 for f in flights if f[2])
            print(f"Tekstura: {texture.name}")
            print(f"  Liczba lotów: {n}")
            successful_flights = [f for f in flights if f[2]] 
            success_n = len(successful_flights)
            print(f"  Średnia nadmiarowość trasy: { '-' if success_n == 0 else f'{sum((f[1] - f[0]) / f[0] * 100 for f in successful_flights) / success_n:.2f}%'}")
            print(f"  Odsetek udanych lądowań: {100 * total_landed / n:.1f}%")
            print()

    def plot_all_positions(self, helicopter_id):
        true = self.true_positions[helicopter_id]
        est = self.estimated_positions[helicopter_id]
        obs = self.observed_positions[helicopter_id]

        labels = ['x', 'y', 'z']
        for i, axis in enumerate(labels):
            true_vals = [p[i] for p in true]
            est_vals = [p[i] for p in est]
            obs_vals = [p[i] for p in obs]

            plt.figure(figsize=(10, 6))
            plt.plot(est_vals, label=f"Estymowana {axis}")
            plt.plot(true_vals, label=f"Rzeczywista {axis}")
            plt.plot(obs_vals, label=f"Obserwacja {axis}")

            plt.xlabel("Czas (iteracje)")
            plt.ylabel(f"Pozycja {axis.upper()}")
            plt.title(f"Przebieg pozycji {axis.upper()} w czasie (helikopter ID={helicopter_id})")
            plt.legend()

            filename = f"wykres_{axis}.jpg"
            plt.savefig(filename, format='jpg')
            plt.close()

    def generate_report(self):
        print('PARAMETRY SYMULACJI:', file=self.file)
        print(f'Liczba cząstek: {NUM_PARTICLES}', file=self.file)
        print(f'Długość kroku: {STEP_SIZE}', file=self.file)
        print(f'Szum pozycyjny w obserwacji: {OBSERVATION_NOISE.pos_std}', file=self.file)
        print(f'Szum pozycyjny w ruchu: {MOTION_NOISE.pos_std}', file=self.file)
        print(f'Szum kąta w ruchu: {MOTION_NOISE.yaw_std}', file=self.file)
        print(f'Szum pozycyjny generowania cząstek: {PARTICLE_FITLER_INIT_NOISE.pos_std}', file=self.file)
        print(f'Szum kąta generowania cząstek: {PARTICLE_FITLER_INIT_NOISE.yaw_std}', file=self.file)
        print('---------------------------------------------------------', file=self.file)
        for texture, flights in self.data.items():
            print(f'Tekstura: {texture.name}', file=self.file)
            for i, (shortest, traveled, landed) in enumerate(flights, 1):
                redundancy = ((traveled - shortest) / shortest * 100) if shortest > 0 else 0
                status = 'Udany' if landed else 'Nieudany'
                print(f'  Lot {i}: {status}, Długość: {traveled:.2f}, Nadmiarowość: {redundancy:.2f}%', file=self.file)
            print()
        print('---------------------------------------------------------', file=self.file)
        for texture, flights in self.data.items():
            n = len(flights)
            total_landed = sum(1 for f in flights if f[2])
            print(f"Tekstura: {texture.name}", file=self.file)
            print(f"  Liczba lotów: {n}", file=self.file)
            successful_flights = [f for f in flights if f[2]] 
            success_n = len(successful_flights)
            print(f"  Średnia nadmiarowość trasy: { '-' if success_n == 0 else f'{sum((f[1] - f[0]) / f[0] * 100 for f in successful_flights) / success_n:.2f}%'}", file=self.file)
            print(f"  Odsetek udanych lądowań: {100 * total_landed / n:.1f}%", file=self.file)


def run_simulation():
    textures = [TextureType.TRIANGLE, TextureType.CROSS, TextureType.CROSS_WITH_CENTER]
    stats = Statistics()
    for texture in textures:
        for _ in range(ITERATIONS):
            helicopter = Helicopter()
            shortest_distance = helicopter.shortest_distance_to_platform()
            particle_filter = ParticleFilter(helicopter, NUM_PARTICLES, PARTICLE_FITLER_INIT_NOISE)
            estimated_state = particle_filter.estimate()
            while not Problem.landed(estimated_state, accuracy=0.5) and helicopter.distance_traveled < shortest_distance*5:
                expected_move = Problem.calculate_move(estimated_state, STEP_SIZE)
                helicopter.move(expected_move, MOTION_NOISE)
                particle_filter.predict(expected_move, MOTION_NOISE)

                observation = Problem.get_observation(helicopter.position(), texture, OBSERVATION_NOISE)
                observation_cam_coords = Problem.observation_to_camera_coordinates(helicopter.yaw, observation)
                particle_filter.update_weights(helicopter, observation_cam_coords, texture, OBSERVATION_NOISE, MOTION_NOISE)
                particle_filter.resample()
                estimated_state = particle_filter.estimate()
                
                stats.record_observation(helicopter, observation, estimated_state)
            stats.record_single_flight(texture, shortest_distance, helicopter.distance_traveled, 
                                    Problem.landed(helicopter.position(), accuracy=0.5))
    stats.plot_all_positions(0)
    stats.print_summary()
    stats.generate_report()
   

if __name__ == "__main__":
    print('Projekt N4 na przedmiot Inteligentne Techniki Obliczeniowe.')
    print('(filtr cząsteczkowy dla symulacji lądowania helikopterem)')
    print('---------------------------------------------------------')
    NUM_PARTICLES = int(input(f'Liczba cząstek [{NUM_PARTICLES}]: ') or NUM_PARTICLES)
    STEP_SIZE = float(input(f'Długość kroku [{STEP_SIZE}]: ') or STEP_SIZE)
    OBSERVATION_NOISE.pos_std = float(input(f'Szum pozycyjny w obserwacji [{OBSERVATION_NOISE.pos_std}]: ') or OBSERVATION_NOISE.pos_std)
    MOTION_NOISE.pos_std = float(input(f'Szum pozycyjny w ruchu [{MOTION_NOISE.pos_std}]: ') or MOTION_NOISE.pos_std)
    MOTION_NOISE.yaw_std = float(input(f'Szum kąta w ruchu [{MOTION_NOISE.yaw_std}]: ') or MOTION_NOISE.yaw_std)
    if MOTION_NOISE.yaw_std  > math.pi: MOTION_NOISE.yaw_std  = math.pi
    PARTICLE_FITLER_INIT_NOISE.pos_std = float(input(f'Szum pozycyjny generowania cząstek [{PARTICLE_FITLER_INIT_NOISE.pos_std}]: ') or PARTICLE_FITLER_INIT_NOISE.pos_std)
    PARTICLE_FITLER_INIT_NOISE.yaw_std = float(input(f'Szum kąta generowania cząstek [{PARTICLE_FITLER_INIT_NOISE.yaw_std}]: ') or PARTICLE_FITLER_INIT_NOISE.yaw_std)
    if PARTICLE_FITLER_INIT_NOISE.yaw_std  > math.pi: PARTICLE_FITLER_INIT_NOISE.yaw_std  = math.pi
    print('---------------------------------------------------------')
    print('LICZENIE SYMULACJI...')
    print()
    run_simulation()
