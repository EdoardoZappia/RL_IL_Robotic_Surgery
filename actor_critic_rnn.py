import torch
import torch.nn as nn

class ActorCriticRNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128, rnn_type='gru'):
        super(ActorCriticRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # RNN per l'Actor
        if rnn_type == 'gru':
            self.actor_rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.actor_rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Output dimensione = azione
        )
        
        # RNN per il Critic
        if rnn_type == 'gru':
            self.critic_rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.critic_rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output dimensione = scalare (valore dello stato)
        )
    
    def forward(self, state):
        
        # Actor
        actor_output, _ = self.actor_rnn(state)
        action = self.actor_fc(actor_output[:, -1, :])  # Usa l'ultimo hidden state

        # Critic
        critic_output, _ = self.critic_rnn(state)
        value = self.critic_fc(critic_output[:, -1, :])  # Usa l'ultimo hidden state
        
        return action, value

# Funzione di reward
def compute_reward(predicted_state, real_state):
    # Reward = -distanza euclidea tra stato predetto e stato reale
    return -torch.norm(predicted_state - real_state, dim=1)

# Funzione di aggiornamento
def update(actor_critic_rnn, optimizer, state, real_next_state, gamma=0.99):

    # Predizione del Critic e Actor
    action, state_value = actor_critic_rnn(state)

    # Predici il prossimo stato dato l'azione
    predicted_next_state = state + action
    
    # Calcola il reward
    reward = compute_reward(predicted_next_state, real_next_state)
    
    # Calcola il valore del prossimo stato
    with torch.no_grad():
        _, next_state_value = actor_critic_rnn(real_next_state)
    
    # Target per Critic: Bellman equation
    target_value = reward + gamma * next_state_value
    
    # Loss Critic
    critic_loss = nn.MSELoss()(state_value, target_value)
    
    # Loss Actor: Policy gradient
    advantage = (target_value - state_value.detach())
    #baseline = torch.mean(reward).detach()  # Calcola un baseline sul batch
    #advantage = reward - baseline
    actor_loss = -torch.mean(advantage * torch.norm(action, dim=1))
    
    loss = actor_loss + critic_loss

    # Aggiorna i pesi solo se l'optimizer è fornito
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item(), actor_loss.mean().item(), critic_loss.item(), reward.mean().item()