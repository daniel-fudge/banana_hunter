"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from collections import deque
from hunter.dqn_agent import Agent
import numpy as np
import torch
from unityagents import UnityEnvironment

# !!!!!!!!! YOU MAY NEED TO EDIT THIS !!!!!!!!!!!!!!!
env = UnityEnvironment(file_name=r"Banana_Windows_x86_64\Banana.exe")


def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """This function trains the given agent in the given environment.

    Args:
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = list()
    scores_window = deque(maxlen=100)
    eps = eps_start
    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes + 1):
        brain_info = env.reset(train_mode=True)[brain_name]
        state = brain_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps).astype(int)
            brain_info = env.step(action)[brain_name]
            next_state = brain_info.vector_observations[0]
            reward = brain_info.rewards[0]
            done = brain_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            np.savez('scores.npz', scores)
            break


def setup():

    # Setup the environment and print of some information for reference
    # -----------------------------------------------------------------------------------
    print('Setting up the environment.')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Setup the agent and return it
    # -----------------------------------------------------------------------------------
    print('Setting up the agent.')
    hidden_layer_sizes = (state_size, state_size, int((state_size + action_size) / 2.0))
    return Agent(state_size=state_size, action_size=action_size, seed=0, h_sizes=hidden_layer_sizes)


if __name__ == "__main__":
    # Setup the environment and agent
    # -----------------------------------------------------------------------------------
    agent = setup()

    # Perform the training
    # -----------------------------------------------------------------------------------
    print('Training the agent.')
    train()
