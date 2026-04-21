"""
Actor-Critic Hedge (ACH) / NW-CFR Algorithm for OpenSpiel
PyTorch implementation based on:

"Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"
(Fu et al., ICLR 2022)

Core idea: Neural Network-based Weighted Counterfactual Regret Minimization.
- A Q-network (critic) estimates action values Q(s, a).
- A regret network approximates cumulative counterfactual regret R(s, a).
- The policy is derived via the Hedge formula: π(a|s) ∝ exp(η · R(s, a)).
- Training is on-policy and iterative (Algorithm 1 / NW-CFR).
"""

import os
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pyspiel


class QNetwork(nn.Module):
    """Action-value (Q) network: maps state → Q-values for all actions."""

    def __init__(self, state_size, action_size, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = state_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)]
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """Returns Q(s, ·) of shape (batch, action_size)."""
        return self.net(state)


class RegretNetwork(nn.Module):
    """Regret network: approximates cumulative counterfactual regret R(s, a)."""

    def __init__(self, state_size, action_size, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = state_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)]
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """Returns R(s, ·) of shape (batch, action_size)."""
        return self.net(state)


def get_state_size(game):
    """Get the state tensor size, preferring information_state_tensor."""
    game_type = game.get_type()
    if game_type.provides_information_state_tensor:
        return game.information_state_tensor_shape()[0]
    elif game_type.provides_observation_tensor:
        return game.observation_tensor_shape()[0]
    else:
        raise ValueError(
            f"Game {game_type.short_name} provides neither "
            "information_state_tensor nor observation_tensor.")


def get_state_tensor(state, player, game):
    """Get the state tensor for a player, preferring information_state_tensor."""
    if game.get_type().provides_information_state_tensor:
        return state.information_state_tensor(player)
    return state.observation_tensor(player)


def _collect_episodes_worker(worker_args):
    """
    Worker function for multiprocessing trajectory collection.

    Must be a top-level function (picklable). Each worker creates its own
    game instance and lightweight CPU-only agents, then collects episodes
    independently.
    """
    (game_name, num_episodes, state_size, action_size,
     agent_weights, eta) = worker_args

    game = pyspiel.load_game(game_name)
    num_players = game.num_players()

    # Build lightweight CPU agents for inference only
    agents = []
    for p in range(num_players):
        agent = ACHAgent(state_size, action_size, device='cpu')
        agent.eta = eta
        agent.regret_net.load_state_dict(agent_weights[p]['regret_net'])
        agent.q_net.load_state_dict(agent_weights[p]['q_net'])
        agents.append(agent)

    # Reuse the sequential collection logic
    return ACHAgent.collect_trajectories(game, agents, num_episodes)


class ACHAgent:
    """
    Actor-Critic Hedge (ACH) Agent implementing NW-CFR (Algorithm 1).

    Key components:
      - Q-network (critic): trained with TD or MC targets.
      - Regret network: trained to predict cumulative counterfactual regret.
      - Policy: derived from regret network via Hedge formula, NOT a separate
        trainable policy network.

    Training flow per iteration:
      1. Generate on-policy trajectories with the current Hedge policy.
      2. Train the Q-network on collected (s, a, r, s', done) tuples.
      3. Compute counterfactual advantages for all legal actions at each
         visited state: A(s,a) = Q(s,a) − Σ_b π(b|s)·Q(s,b).
      4. Accumulate regrets: target_R(s,a) = prev_R(s,a) + A(s,a).
      5. Train regret network to predict the new cumulative regret targets.
      6. Derive next iteration's policy: π(a|s) = softmax(η · R(s,a)).
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr_critic=1e-3,
        lr_regret=1e-3,
        gamma=1.0,
        eta=0.1,
        entropy_coef=0.01,
        regret_update_epochs=5,
        critic_update_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.regret_update_epochs = regret_update_epochs
        self.critic_update_epochs = critic_update_epochs
        self.device = device

        # Critic (Q-network)
        self.q_net = QNetwork(state_size, action_size).to(device)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_critic)

        # Regret network
        self.regret_net = RegretNetwork(state_size, action_size).to(device)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=lr_regret)

    # ------------------------------------------------------------------
    # Policy derivation (Hedge)
    # ------------------------------------------------------------------

    def get_policy(self, state_tensor, legal_actions_mask):
        """
        Derive policy from regret network via Hedge formula.

        π(a|s) = softmax(η · R(s, a))  masked to legal actions.

        Args:
            state_tensor: (batch, state_size) or (state_size,)
            legal_actions_mask: bool tensor, same leading dims as state_tensor

        Returns:
            action_probs: (batch, action_size) probability distribution
        """
        squeeze = state_tensor.dim() == 1
        if squeeze:
            state_tensor = state_tensor.unsqueeze(0)
            legal_actions_mask = legal_actions_mask.unsqueeze(0)

        with torch.no_grad():
            regrets = self.regret_net(state_tensor)  # (B, A)

        logits = self.eta * regrets
        logits = logits.masked_fill(~legal_actions_mask, -1e9)
        probs = F.softmax(logits, dim=-1)

        if squeeze:
            probs = probs.squeeze(0)
        return probs

    def select_action(self, state, legal_actions_mask, training=True):
        """Select action using current Hedge-derived policy."""
        state_t = torch.FloatTensor(state).to(self.device)
        mask_t = torch.BoolTensor(legal_actions_mask).to(self.device)

        probs = self.get_policy(state_t, mask_t)

        if training:
            action = torch.distributions.Categorical(probs).sample().item()
        else:
            action = probs.argmax().item()
        return action

    # ------------------------------------------------------------------
    # Trajectory collection
    # ------------------------------------------------------------------

    @staticmethod
    def collect_trajectories(game, agents, num_episodes, state_fn=None):
        """
        Play *num_episodes* games using each agent's current policy and
        collect per-player transition data.

        Returns a list (one per player) of lists of transition dicts.
        Each dict:  {state, action, reward, next_state, done, legal_mask}
        """
        num_players = game.num_players()
        action_size = game.num_distinct_actions()
        all_transitions = [[] for _ in range(num_players)]

        for _ in range(num_episodes):
            state = game.new_initial_state()

            # Per-player pending transitions (state, action, mask) awaiting
            # reward / next_state once the game progresses or ends.
            pending = [None] * num_players

            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    state.apply_action(np.random.choice(actions, p=probs))
                    continue

                player = state.current_player()
                info = np.array(get_state_tensor(state, player, game),
                                dtype=np.float32)
                legal = state.legal_actions(player)
                mask = np.zeros(action_size, dtype=bool)
                mask[legal] = True

                action = agents[player].select_action(info, mask, training=True)

                # If there was a pending transition for this player, we can
                # now fill in its next_state (the current info state).
                if pending[player] is not None:
                    p = pending[player]
                    p['next_state'] = info.copy()
                    p['done'] = False
                    p['reward'] = 0.0
                    all_transitions[player].append(p)

                pending[player] = {
                    'state': info.copy(),
                    'action': action,
                    'legal_mask': mask.copy(),
                }

                state.apply_action(action)

            # Terminal — flush all pending transitions with terminal rewards.
            returns = state.returns()
            for p_idx in range(num_players):
                if pending[p_idx] is not None:
                    pend = pending[p_idx]
                    pend['next_state'] = np.zeros(agents[p_idx].state_size,
                                                  dtype=np.float32)
                    pend['done'] = True
                    pend['reward'] = returns[p_idx]
                    all_transitions[p_idx].append(pend)

        return all_transitions

    @staticmethod
    def collect_trajectories_parallel(
        game_name, agents, num_episodes, num_workers=None, pool=None
    ):
        """
        Parallel version of collect_trajectories using multiprocessing.

        Splits episodes across *num_workers* processes. Each worker creates
        its own game instance and lightweight CPU agents loaded with the
        current network weights, collects its share of episodes, and returns
        the transitions.

        Args:
            game_name: string name of the game (workers recreate the game).
            agents: list of ACHAgent (main-process agents, used for weights).
            num_episodes: total episodes to collect.
            num_workers: number of worker processes (default: min(cpu_count, 4)).

        Returns:
            Same format as collect_trajectories: list of transition lists,
            one per player.
        """
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, 4)

        # Fall back to sequential if only 1 worker or very few episodes
        if num_workers <= 1 or num_episodes <= 1:
            game = pyspiel.load_game(game_name)
            return ACHAgent.collect_trajectories(game, agents, num_episodes)

        num_players = len(agents)
        state_size = agents[0].state_size
        action_size = agents[0].action_size
        eta = agents[0].eta

        # Extract CPU-side weight dicts (picklable)
        agent_weights = []
        for agent in agents:
            agent_weights.append({
                'regret_net': {k: v.cpu() for k, v in
                               agent.regret_net.state_dict().items()},
                'q_net': {k: v.cpu() for k, v in
                          agent.q_net.state_dict().items()},
            })

        # Split episodes across workers
        base = num_episodes // num_workers
        remainder = num_episodes % num_workers
        episode_counts = [base + (1 if i < remainder else 0)
                          for i in range(num_workers)]
        # Filter out workers with 0 episodes
        episode_counts = [c for c in episode_counts if c > 0]

        worker_args = [
            (game_name, count, state_size, action_size, agent_weights, eta)
            for count in episode_counts
        ]

        # Use an existing pool if provided, otherwise create a temporary one.
        # Creating a pool per call is expensive with 'spawn' context (Linux).
        if pool is not None:
            results = pool.map(_collect_episodes_worker, worker_args)
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=len(episode_counts)) as tmp_pool:
                results = tmp_pool.map(_collect_episodes_worker, worker_args)

        # Aggregate: merge transition lists from all workers
        all_transitions = [[] for _ in range(num_players)]
        for worker_result in results:
            for p in range(num_players):
                all_transitions[p].extend(worker_result[p])

        return all_transitions

    # ------------------------------------------------------------------
    # NW-CFR update step (Algorithm 1)
    # ------------------------------------------------------------------

    def update(self, transitions):
        """
        Perform one NW-CFR iteration given a list of transition dicts
        collected under the current policy.

        Steps:
          1. Train Q-network on (s, a, r, s', done) via TD(0).
          2. Compute counterfactual advantages A(s, a) for **all** actions.
          3. Compute regret targets = prev_regret + advantage.
          4. Train regret network on new targets.

        Returns (critic_loss, regret_loss) averages.
        """
        if not transitions:
            return None, None

        # Prepare tensors ------------------------------------------------
        states = torch.FloatTensor(
            np.array([t['state'] for t in transitions])).to(self.device)
        actions = torch.LongTensor(
            [t['action'] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor(
            [t['reward'] for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(
            np.array([t['next_state'] for t in transitions])).to(self.device)
        dones = torch.FloatTensor(
            [float(t['done']) for t in transitions]).to(self.device)
        masks = torch.BoolTensor(
            np.array([t['legal_mask'] for t in transitions])).to(self.device)

        # 1. Critic (Q-network) update -----------------------------------
        critic_losses = []
        for _ in range(self.critic_update_epochs):
            q_values = self.q_net(states)                        # (N, A)
            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = self.q_net(next_states)                 # (N, A)
                # Use max over legal actions of next state as bootstrap.
                # For terminal states this is ignored (dones mask).
                q_next_max = q_next.max(dim=1).values
                td_target = rewards + self.gamma * q_next_max * (1 - dones)

            critic_loss = F.mse_loss(q_taken, td_target)

            self.q_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.q_optimizer.step()
            critic_losses.append(critic_loss.item())

        # 2. Compute counterfactual advantages for all actions ------------
        with torch.no_grad():
            q_all = self.q_net(states)                           # (N, A)

            # Current policy probabilities
            regrets_pred = self.regret_net(states)               # (N, A)
            logits = self.eta * regrets_pred
            logits = logits.masked_fill(~masks, -1e9)
            policy_probs = F.softmax(logits, dim=-1)             # (N, A)

            # Baseline: V(s) = Σ_a π(a|s) Q(s,a)
            baseline = (policy_probs * q_all).sum(dim=1, keepdim=True)  # (N,1)

            # Counterfactual advantage for every action
            advantages = q_all - baseline                        # (N, A)

            # 3. Regret targets: accumulate
            # prev regret + new advantage (only for legal actions)
            regret_targets = regrets_pred + advantages           # (N, A)
            # Zero out illegal actions
            regret_targets = regret_targets.masked_fill(~masks, 0.0)

        # 4. Train regret network on new targets --------------------------
        regret_losses = []
        for _ in range(self.regret_update_epochs):
            pred = self.regret_net(states)                       # (N, A)
            # Only train on legal actions
            diff = (pred - regret_targets) ** 2
            diff = diff * masks.float()
            regret_loss = diff.sum() / masks.float().sum()

            self.regret_optimizer.zero_grad()
            regret_loss.backward()
            nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
            self.regret_optimizer.step()
            regret_losses.append(regret_loss.item())

        return np.mean(critic_losses), np.mean(regret_losses)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'regret_net': self.regret_net.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.regret_net.load_state_dict(ckpt['regret_net'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])
        self.regret_optimizer.load_state_dict(ckpt['regret_optimizer'])


# ======================================================================
# Training loop
# ======================================================================

def train_ach_openspiel(
    game_name='leduc_poker',
    num_iterations=5000,
    episodes_per_iter=20,
    eval_freq=100,
    save_freq=1000,
    num_workers=None,
):
    """
    Train ACH agents via iterative self-play (NW-CFR Algorithm 1).

    Each iteration:
      1. Both players generate on-policy trajectories.
      2. Each player updates its Q-network and regret network.
      3. Policy is re-derived from updated regret network (Hedge).
    """
    game = pyspiel.load_game(game_name)
    state_size = get_state_size(game)
    action_size = game.num_distinct_actions()
    num_players = game.num_players()

    agents = [ACHAgent(state_size, action_size) for _ in range(num_players)]

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 4)
    use_parallel = num_workers > 1
    if use_parallel:
        print(f"Using {num_workers} worker processes for trajectory collection")

    # Create a persistent pool once — avoids per-iteration process startup
    # overhead, which is very expensive with 'spawn' context.
    pool = None
    if use_parallel:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=num_workers)

    try:
        for iteration in range(num_iterations):
            # --- Collect on-policy data under current policies ---------------
            if use_parallel:
                all_transitions = ACHAgent.collect_trajectories_parallel(
                    game_name, agents, episodes_per_iter, num_workers,
                    pool=pool)
            else:
                all_transitions = ACHAgent.collect_trajectories(
                    game, agents, episodes_per_iter)

            # --- Update each player's networks -------------------------------
            losses = []
            for p in range(num_players):
                c_loss, r_loss = agents[p].update(all_transitions[p])
                losses.append((c_loss, r_loss))

            # --- Logging -----------------------------------------------------
            if (iteration + 1) % eval_freq == 0:
                # Quick evaluation: play 100 games, agent 0 vs random
                wins = 0
                total_return = 0.0
                for _ in range(100):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        if st.is_chance_node():
                            outcomes = st.chance_outcomes()
                            acts, prbs = zip(*outcomes)
                            st.apply_action(np.random.choice(acts, p=prbs))
                            continue
                        cp = st.current_player()
                        la = st.legal_actions(cp)
                        mask = np.zeros(action_size, dtype=bool)
                        mask[la] = True
                        if cp == 0:
                            a = agents[0].select_action(
                                get_state_tensor(st, 0, game), mask,
                                training=False)
                        else:
                            a = np.random.choice(la)
                        st.apply_action(a)
                    ret = st.returns()
                    total_return += ret[0]
                    if ret[0] > ret[1]:
                        wins += 1

                c0, r0 = losses[0] if losses[0][0] is not None else (0, 0)
                print(f"Iter {iteration+1:>5}/{num_iterations}  "
                      f"WinRate={wins/100:.0%}  AvgRet={total_return/100:+.3f}  "
                      f"CriticL={c0:.4f}  RegretL={r0:.4f}")

            # --- Checkpoint ---------------------------------------------------
            if (iteration + 1) % save_freq == 0:
                for i, ag in enumerate(agents):
                    ag.save(f'ach_player{i}_iter{iteration+1}.pth')
                print(f"  [checkpoint saved at iter {iteration+1}]")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return agents


if __name__ == "__main__":
    print("Training ACH (NW-CFR) on Leduc Poker …")
    agents = train_ach_openspiel(
        game_name='leduc_poker',
        num_iterations=5000,
        episodes_per_iter=20,
        eval_freq=100,
        save_freq=1000,
        num_workers=4,
    )
    print("Training completed!")
