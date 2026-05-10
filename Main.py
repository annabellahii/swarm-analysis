import pybullet as p
import pybullet_data
import time
import random
import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def run_simulation(population_genomes, generation_id, noise_std=0.0):
   
    #Connect to Physics Server
    physicsClient = p.connect(p.GUI) #changed from GUI to DIRECT for testing without visualization
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

    #Global World Settings
    p.resetDebugVisualizerCamera(cameraDistance=25.0, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0]) 
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0) 

    #Load assets
    planeId = p.loadURDF("plane.urdf")
    hive_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.5, length=0.1, rgbaColor=[0.5, 0.3, 0, 1])
    hive_id = p.createMultiBody(baseVisualShapeIndex=hive_visual, basePosition=[0, 0, 0.05])

    #Bee swarm configuration
    num_bees = 100
    bee_ids = []
    bees_memories = [None] * num_bees 
    bee_states = [0] * num_bees #0=searching, 1=returning/dancing, 2=returning empty
    bee_dance_timers = [0] * num_bees 
    bee_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 0, 1])
    bee_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
    individual_pollen = [0] * num_bees #Track individual fitness

    #Track misguided trips
    misguided_trips = 0

    for i in range(num_bees):
        if i < 30: #Scouts
            start_pos = [random.uniform(-5, 5), random.uniform(-5, 5), 0.15]
        else: #Lazy
            start_pos = [random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4), 0.2]
        
        bee_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=bee_collision, 
                                baseVisualShapeIndex=bee_visual, basePosition=start_pos)
        p.changeDynamics(bee_id, -1, lateralFriction=0.1, linearDamping=0.5)
        bee_ids.append(bee_id)

    #Pollen configuration
    pollen_ids = []
    num_patches = 6
    pollen_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.2, length=0.01, rgbaColor=[0, 0.8, 0, 0.7]) #radius changed for testing communication
    pollen_collision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.2, height=0.01)
    patch_capacities = [50] * num_patches
    patch_active = [True] * num_patches

    for i in range(num_patches):
        random.seed(i + 42) #Consistent patch placement across generations
        dist = random.uniform(15, 25) #Far to force communication
        angle = random.uniform(0, 2 * 3.14159)
        px, py = dist * math.cos(angle), dist * math.sin(angle)
        p_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=pollen_collision, 
                                baseVisualShapeIndex=pollen_visual, basePosition=[px, py, 0.01])
        pollen_ids.append(p_id)

    start_time = time.time()
    sim_duration = 120 #Enough time for multiple foraging trips and dances to occur

    #Simulation Loop
    while (time.time() - start_time) < sim_duration:
        p.stepSimulation()
        
        for i, b_id in enumerate(bee_ids):
            pos, _ = p.getBasePositionAndOrientation(b_id)
            #Genome data [dance prob, dance duration, recruitment radius]
            genome = population_genomes[i]
            
            #Memory forgetting logic - if pollen patch exhausted
            if bee_states[i] == 0 and bees_memories[i] is not None:
                mem_x, mem_y = bees_memories[i]
                still_exists = False
                for idx, p_id in enumerate(pollen_ids):
                    if patch_active[idx]:
                        p_pos, _ = p.getBasePositionAndOrientation(p_id)
                        if math.sqrt((mem_x-p_pos[0])**2 + (mem_y-p_pos[1])**2) < 1.5:
                            still_exists = True
                
                if not still_exists:
                    bees_memories[i] = None
                    bee_states[i] = 2 #Return home empty

            #FSM Logic
            
            #State 0: Searching / Flying to Patch
            if bee_states[i] == 0:
                if bees_memories[i] is None:
                    if i < 50: #Scout movement - Random walk
                        #Random walk - Randomize x and y forces
                        force_x = random.uniform(-4.0, 4.0)
                        force_y = random.uniform(-4.0, 4.0)
                        force_z = 0.5 #Constant lift to keep them airborne
                        p.applyExternalForce(b_id, -1, [force_x, force_y, force_z], pos, p.WORLD_FRAME)
                    else: #Lazy hover
                        p.applyExternalForce(b_id, -1, [0, 0, 0.5], pos, p.WORLD_FRAME)
                else:
                    mx, my = bees_memories[i]
                    p.applyExternalForce(b_id, -1, [(mx-pos[0])*0.6, (my-pos[1])*0.6, 0], pos, p.WORLD_FRAME)

                    #Tracker Logic
                    dist_to_mem = math.sqrt((mx-pos[0])**2 + (my-pos[1])**2)
                    if dist_to_mem < 0.8: #If bee reaches memory spot
                        hit_pollen = False
                        for p_id in pollen_ids:
                            if p.getContactPoints(bodyA=b_id, bodyB=p_id):
                                hit_pollen = True
                                break
                        if not hit_pollen:
                            misguided_trips += 1        
                            bees_memories[i] = None
                            bee_states[i] = 2 #Return home empty

                #Harvest Check
                for idx, p_id in enumerate(pollen_ids):
                    if patch_active[idx] and p.getContactPoints(bodyA=b_id, bodyB=p_id):
                        patch_capacities[idx] -= 1
                        p_pos, _ = p.getBasePositionAndOrientation(p_id)
                        bees_memories[i] = [p_pos[0], p_pos[1]]
                        bee_states[i] = 1 # GOT POLLEN
                        p.changeVisualShape(b_id, -1, rgbaColor=[1.0, 0.75, 0.8, 1.0]) #PINK - pollen found
                        
                        if patch_capacities[idx] <= 0:
                            patch_active[idx] = False
                            p.resetBasePositionAndOrientation(p_id, [0, 0, -10], [0,0,0,1])
                            print(f"Patch {idx} exhausted!")
                        break

            #State 1: Returning with Pollen 
            elif bee_states[i] == 1:
                dx, dy = 0 - pos[0], 0 - pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist > 0.5 and bee_dance_timers[i] == 0:
                    p.applyExternalForce(b_id, -1, [dx*0.5, dy*0.5, 0], pos, p.WORLD_FRAME)
                else:
                    #Dance/Recruit
                    #Gene 1 = Dance duration
                    if bee_dance_timers[i] < genome[1]: 
                        bee_dance_timers[i] += 1

                        #Gene 0 = Dance probability
                        if random.random() < genome[0]: 
                            curr_mem = bees_memories[i]

                            #Gaussian noise to simulate imperfect communication
                            noise_x = curr_mem[0] + np.random.normal(0, noise_std)
                            noise_y = curr_mem[1] + np.random.normal(0, noise_std)
                            noisy_mem = [noise_x, noise_y]
                    
                            #Gene 2 = Recruitment radius
                            recruit_radius = genome[2] 

                            for j, other_id in enumerate(bee_ids):
                                if bee_states[j] == 0 and bees_memories[j] is None:
                                    o_pos, _ = p.getBasePositionAndOrientation(other_id)
                                    if math.sqrt((pos[0]-o_pos[0])**2 + (pos[1]-o_pos[1])**2) < recruit_radius:
                                        bees_memories[j] = noisy_mem
                                        p.changeVisualShape(other_id, -1, rgbaColor=[1, 0.5, 0, 1]) #orange - recruited 
                    else:
                        individual_pollen[i] += 1 #Add to fitness score
                        bee_states[i] = 0
                        bee_dance_timers[i] = 0
                        bees_memories[i] = None #Clear memory after delivery
                        p.changeVisualShape(b_id, -1, rgbaColor=[1, 1, 0, 1]) #Back to yellow

            #State 2: Returning Empty
            elif bee_states[i] == 2:
                dx, dy = 0 - pos[0], 0 - pos[1]
                dist_to_hive = math.sqrt(dx**2 + dy**2)
                if dist_to_hive> 0.6:
                    p.applyExternalForce(b_id, -1, [dx*0.5, dy*0.5, 0], pos, p.WORLD_FRAME)
                else:
                    #Back at hive - reset everything
                    bee_states[i] = 0
                    bees_memories[i] = None
                    p.changeVisualShape(b_id, -1, rgbaColor=[1, 1, 0, 1]) #Back to YELLOW

        p.stepSimulation()
        
    p.disconnect()
    #Large bonus for bringing pollen back
    fitness_scores = [p * 10 for p in individual_pollen] 
    return fitness_scores, misguided_trips

def evolve_population(current_population, fitness_scores, mutation_rate=0.1):
    new_population = []
    #Selection - top 20% elites
    indices = np.argsort(fitness_scores)[::-1]
    elites = current_population[indices[:20]]
    new_population.extend(elites)

    #Crossover & Mutation
    while len(new_population) < len(current_population):
        parent1, parent2 = random.sample(list(elites), 2)
        child = (parent1 + parent2)/2
        #Mutation
        if random.random() < mutation_rate:
            child += np.random.normal(0, [0.1, 10.0, 0.5], size=3)

        child[0] = np.clip(child[0], 0, 1) #Dance prob 0-1
        child[1] = np.clip(child[1], 10, 500) #Dance duration 10-500 steps
        child[2] = np.clip(child[2], 0.5, 10) #Recruitment radius 0.5-10
        new_population.append(child)

    return np.array(new_population)

population = np.random.rand(100, 3)
population[:, 1] *=300 #Scale duration
population[:, 2] *= 10 #Scale radius
reality_noise = 0.8
results_file = 'small_patch_results.csv'

if os.path.exists(results_file): os.remove(results_file)

for gen in range(10):
    print(f"Evaluating Generation {gen} with Noise={reality_noise} and Patch Radius 0.2...")
    scores, misguided_trips = run_simulation(population, gen, noise_std=reality_noise)

    avg_fitness = np.mean(scores)
    avg_dance_prob = np.mean(population[:, 0])
    avg_radius = np.mean(population[:, 2])  
    print(f"Generation {gen} Results: Avg Pollen={avg_fitness:.2f}, Avg Dance Prob={avg_dance_prob:.2f}, Avg Radius={avg_radius:.2f}")

    #Save generation data to a CSV
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if gen == 0:
            writer.writerow(['Generation', 'Avg_Pollen', 'Avg_Dance_Prob', 'Avg_Radius', 'Misguided_Trips'])
        writer.writerow([gen, avg_fitness, avg_dance_prob, avg_radius, misguided_trips])

    #Evolve for next generation
    population = evolve_population(population, scores)

print("\n Evolution Experiment Completed!")

#GA Results Visualization - changed for different tolerance experiments
def plot_results():
    data = np.genfromtxt(results_file, delimiter=',', names=True)
    plt.figure(figsize=(15, 5))
    
    #Fitness
    plt.subplot(1, 3, 1)
    plt.plot(data['Generation'], data['Avg_Pollen'], marker='o', color='green')
    plt.title('Swarm Efficiency (with 0.2 Patch Radius)')
    plt.xlabel('Generation')
    plt.ylabel('Avg Pollen Collected')

    #DNA Changes
    plt.subplot(1, 3, 2)
    plt.plot(data['Generation'], data['Avg_Dance_Prob'], label='Dance Prob', color='blue')
    plt.plot(data['Generation'], data['Avg_Radius']/10, label='Radius (Scaled)', color='orange')
    plt.title('Evolution of Communication')
    plt.xlabel('Generation')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(data['Generation'], data['Misguided_Trips'], color='red', alpha=0.6)
    plt.title("Misguided Trips (Reality Gap Indicator)")
    plt.xlabel('Generation')
    plt.ylabel('Misguided Trips')

    plt.tight_layout()
    plt.savefig('small_patch_experiment_analysis.png')
    plt.show()

plot_results()