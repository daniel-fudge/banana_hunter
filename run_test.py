"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from hunter.test import make_plot, setup, train
import os
from unityagents import UnityEnvironment

# !!!!!!!!! YOU MAY NEED TO EDIT THIS !!!!!!!!!!!!!!!
env = UnityEnvironment(file_name=r"Banana_Windows_x86_64\Banana.exe")


if __name__ == "__main__":

    # Delete the old output files
    # -----------------------------------------------------------------------------------
    for name in ['score.png', 'scores.npz', 'checkpoint.pth']:
        if os.path.isfile(name):
            os.remove(name)

    # Setup the environment and agent
    # -----------------------------------------------------------------------------------
    agent = setup(env)

    # Perform the training
    # -----------------------------------------------------------------------------------
    print('Training the agent.')
    train(agent=agent, env=env)

    # Make some pretty plots
    # -----------------------------------------------------------------------------------
    print('Make training plot called scrore.png.')
    make_plot()
