import numpy as np
import random

# Paramètres de la grille
GRID_SIZE = 5

# Définition des récompenses (initialisation à -1 partout)
REWARDS = np.full((GRID_SIZE, GRID_SIZE), -1)

# Position des pièges (☠️) avec une pénalité de -10
traps = [(1, 1), (3, 3)]
for trap in traps:
    REWARDS[trap] = -10

# Position du trésor (🏆) avec une récompense de +10
treasure = (3, 2)
REWARDS[treasure] = 10

# Paramètres du Q-Learning
ALPHA = 0.1  # Taux d'apprentissage
GAMMA = 0.9  # Facteur de discount
EPSILON = 0.3  # Taux d'exploration

# Initialisation de la Q-table (5x5x4 pour les 4 actions)
Q_TABLE = np.zeros((GRID_SIZE, GRID_SIZE, 4))

# Actions possibles (Haut, Bas, Gauche, Droite)
ACTIONS = ["up", "down", "left", "right"]
ACTION_MAP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Fonction pour choisir une action (epsilon-greedy)
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(range(4))  # Exploration aléatoire
    else:
        return np.argmax(Q_TABLE[state[0], state[1]])  # Exploitation

# Fonction pour obtenir le prochain état
def get_next_state(state, action):
    move = ACTION_MAP[ACTIONS[action]]
    new_state = (state[0] + move[0], state[1] + move[1])
    
    # Vérifier les limites de la grille
    if new_state[0] < 0 or new_state[0] >= GRID_SIZE or new_state[1] < 0 or new_state[1] >= GRID_SIZE:
        return state  # Rester sur place si sortie de la grille
    
    return new_state

# Entraînement de l'agent avec Q-Learning
EPISODES = 1000  # Nombre d'épisodes
for episode in range(EPISODES):
    state = (0, 0)  # Position initiale
    done = False

    while not done:
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = REWARDS[next_state]

        # Mettre à jour la Q-table avec la formule de Bellman
        Q_TABLE[state[0], state[1], action] = Q_TABLE[state[0], state[1], action] + \
            ALPHA * (reward + GAMMA * np.max(Q_TABLE[next_state[0], next_state[1]]) - Q_TABLE[state[0], state[1], action])

        state = next_state

        # Arrêter l'épisode si l'agent atteint le trésor ou un piège
        if state == treasure or state in traps:
            done = True

# Affichage de la Q-table après entraînement
print("Q-Table après entraînement :")
print(Q_TABLE)
