import torch
import torch.nn as nn
import torch.optim as optim

# Modello Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, latent_dim, action_dim, window_size=3):
        super(ActorCritic, self).__init__()
        
        self.input_dim = latent_dim * window_size  # Input = finestra temporale concatenata

        # Actor: genera un'azione a partire dallo stato latente
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Output dimensione = azione
        )
        
        # Critic: valuta il valore dello stato
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output dimensione = scalare (valore dello stato)
        )
    
    def forward(self, state_sequence):

        # Appiattisce la finestra temporale
        state_sequence = state_sequence.view(state_sequence.size(0), -1)  # [batch_size, input_dim]

        action = self.actor(state_sequence)  # Azione predetta
        value = self.critic(state_sequence)  # Valore dello stato
        return action, value


# Funzione di reward
def compute_reward(predicted_state, real_state):
    # Reward = -distanza euclidea tra stato predetto e stato reale
    return -torch.norm(predicted_state - real_state, dim=1)

# Funzione di aggiornamento
def update(actor_critic, optimizer, latent_state, real_next_state, gamma=0.99):
    
    state_sequence = latent_states.view(1, -1)

    # Predizione del Critic e Actor
    action, state_value = actor_critic(state_sequence)
    
    last_state = latent_states[:, -1, :]  # Ultimo stato [1, latent_dim]

    # Predici il prossimo stato dato l'azione
    predicted_next_state = last_state + action  # Es.: modello semplice
    
    # Calcola il reward
    reward = compute_reward(predicted_next_state, real_next_state)
    
    # Concatenazione degli ultimi due stati con lo stato reale successivo
    last_two_states = latent_states[:, -2:, :]  # Ultimi due stati concatenati [1, 2 * latent_dim]
    last_two_states = last_two_states.view(1, -1)
    
    real_next_sequence = torch.cat([last_two_states, real_next_state.unsqueeze(0)], dim=1)  # [1, 3 * latent_dim]

    # Calcola il valore del prossimo stato
    with torch.no_grad():
        _, next_state_value = actor_critic(real_next_sequence)
    
    # Target per Critic: Bellman equation
    target_value = reward + gamma * next_state_value.squeeze()
    
    # Loss Critic
    critic_loss = nn.MSELoss()(state_value.view(-1), target_value.view(-1))
    
    # Loss Actor: Policy gradient
    advantage = (target_value - state_value).detach()
    #baseline = torch.mean(reward).detach()  # Calcola un baseline sul batch
    #advantage = reward - baseline
    actor_loss = -torch.mean(advantage * torch.norm(action, dim=1))
    
    # Aggiorna i pesi
    optimizer.zero_grad()
    loss = actor_loss + critic_loss
    loss.backward()
    optimizer.step()
    
    return loss.item(), actor_loss.mean().item(), critic_loss.item(), reward.mean().item()