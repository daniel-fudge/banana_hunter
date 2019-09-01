[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
# Project 1: Navigation
 
**_THE GREAT BANANA HUNTER!!!_**  
![The_Hunter](the_hunter.png)
### Introduction

This is the 1st project in the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).  
The goal of the project is to train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

_Note this GitHub repository and project was completed in a Windows 10 environment_  :poop:  

### Instructions
This project can be executed from the `Navigation.ipynb` Jupyter Notebook or executed (and debugged) from a terminal or an IDE with the `run_test.py` script.  
  
The Jupyter Notebook is a good way to step through the process with a user friendly description.  

The `run_test.py` script performs the training and testing of the DRL model and can be used if you don't like Notebooks or would like to dig into the code with an IDE.  It can be executed with the following command:

    python run_test.py 

### Pretty plot
The raw scores are save as `scores.npz` and can be plotted and saved as `scores.png` with the command:

    python plot.py  

### Saved model Weights
The trained model is saved as `checkpoint.pth` in the root directory and can be loaded with the command:  

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
### Report
A [report](Report.md) providing a description of the implementation, a plot of the rewards and ideas for future work can be found in `Report.md`.