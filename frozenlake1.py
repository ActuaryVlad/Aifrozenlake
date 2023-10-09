import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

random_map = generate_random_map(size=4)  # Genera un mapa aleatorio de tamaño 4x4
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

# Guardar los pesos de la red política y objetivo
ruta_guardado_policy_net = r"C:\Users\monte\Desktop\juegos\carlpolepolicy_net.pth"
ruta_guardado_target_net = r"C:\Users\monte\Desktop\juegos\carlpoletarget.pth"

#Red neuronal 
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# Crear el Buffer 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    


BATCH_SIZE = 50
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# Get the number of state observations otra forma
state, info = env.reset()
#n_observations = len(state)

# Obtener numero de acciones 
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

#crear las instancias de la red 
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #print(f"Episode {transitions} ")
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_network(policy_net, target_net, env, memory, num_episodes, train):
    if not train:
        return

    policy_net.train()

    for i_episode in range(num_episodes):
        # Inicializar el entorno y obtener su estado
        env = CustomReward(env)
        state, info = env.reset()
        current_position = state
        values = env.desc.tolist()
        cell_values = {
        b'S': 1,
        b'F': 18,
        b'H': 19,
        b'G': 20
        }

        state_numeric = [cell_values[cell] for row in values for cell in row]
        state_tensor = torch.tensor(state_numeric).reshape(1,16).float()
        state = state_tensor
        cnt = 0  # Para llevar un registro de cuánto dura el juego
        for t in count():
            env.render()  # Renderizar el entorno
            action = select_action(state).view(1, 1)
            observation, reward, done, info = env.step(action.item())
            #print(reward)

            reward = torch.tensor([reward])
            terminated = done
            cnt += 1  # Aumentar el contador por cada movimiento

            if terminated:
                next_state = None
            else:
                next_state = update_state_tensor(state, current_position, observation)
                current_position = observation
            
            #print(reward)
            # Almacenar la transición en la memoria
            memory.push(state, action, next_state, reward)

            # Pasar al siguiente estado
            state = next_state

            # Realizar un paso de la optimización (en la red de políticas)
            optimize_model()

            # Actualización suave de los pesos de la red objetivo
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                if reward > 0:  # El agente ganó
                    episode_outcomes.append(1)
                else:  # El agente perdió
                    episode_outcomes.append(0)  # Actualizar el gráfico
                #print(f"Episode {i_episode + 1} lasted {cnt} moves")
                break


def test_network(policy_net, env, num_episodes=10, test=False):
    if not test:
        return
    policy_net.eval()
    
    # Correr el entorno por un número de episodios
    for i_episode in range(num_episodes):
        # Generar un mapa nuevo en cada episodio
        random_map = generate_random_map(size=4)

        # Inicializar el entorno y obtener su estado
        state, info = env.reset()
        current_position = state
        values = env.desc.tolist()
        cell_values = {
        b'S': 1,
        b'F': 18,
        b'H': 19,
        b'G': 20
        }

        state_numeric = [cell_values[cell] for row in values for cell in row]
        state_tensor = torch.tensor(state_numeric).reshape(1,16).float()
        state = state_tensor
        for t in count():
            env.render()  # Renderizar el entorno

            # Seleccionar la acción usando la red
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

            # Tomar la acción y obtener la recompensa y el nuevo estado
            observation, reward, terminated, info = env.step(action.item())[:4]
            done = terminated

            print(observation)
            if terminated:
                next_state = None
            else:
                state = update_state_tensor(state, current_position, observation)
                current_position = observation

            # Agregar un break aquí para terminar el ciclo del episodio cuando se termine
            if done:
                break
    env.close()



try:
    policy_net.load_state_dict(torch.load(ruta_guardado_policy_net))
    target_net.load_state_dict(torch.load(ruta_guardado_target_net))
    print("Weights loaded successfully.")
except FileNotFoundError:
    print("No weights found. Training from scratch.")

# Probar la red
train_network(policy_net, target_net, env, memory, num_episodes=100, train=True)
test_network(policy_net, env, num_episodes=10, test=False)

torch.save(policy_net.state_dict(), ruta_guardado_policy_net)
torch.save(target_net.state_dict(), ruta_guardado_target_net)
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
