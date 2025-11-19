# Autonomous Trailer Truck Parking with DRL

[![wakatime](https://wakatime.com/badge/user/53e0bfc9-ae89-4cb3-99fe-c6cbc6359857/project/74e6f5cb-cfe5-4699-8ead-ae3e38671130.svg)](https://wakatime.com/badge/user/53e0bfc9-ae89-4cb3-99fe-c6cbc6359857/project/74e6f5cb-cfe5-4699-8ead-ae3e38671130)

The main problem of the project is to train an agent that controls a vehicle to park in the desired spot while avoiding collision with obstacles. This problem arises from the rapid development of autonomous vehicles. Compared to parking manually, autonomous parking not only saves time, but also achieves more compact parking spaces: it can significantly boost the operation efficiency of both private cars and commercial trucks.

We aim to train an agent that is able to park the vehicle in the target spot from any feasible starting position, under time or space constraints. We approach the problem by modeling it as an environment with continuous state space (position, angle, etc.) and discrete action space (steering, direction, etc.); then we will attempt to train the agent with two algorithms: DQN and PPO. While traditional studies often focus on the parking of cars, we will attempt to park trailer trucks, which have more complex mechanics than cars and are more common in industrial contexts. A real-world application of the problem would be autonomous parking of trailer trucks to facilitate efficient cargo loading and unloading. To start with, we aim to train the agent to back a trailer truck straight into a parking spot that is directly behind it (this is easy for cars, but tricky for trailer trucks).

## Requirements

To start the simulation, create python environment using commands below:

```bash
conda create --name ATTPDRL python=3.13.1
conda activate ATTPDRL
```

Then install pytorch and torchvision using commands below:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then install the required packages using commands below:

```bash
pip install -r ./requirements.txt
```


The demonstration above uses CUDA 13.0, you may modify to your desired version.

## Usage

To run the simulation, use command below:

```bash
python <Name-of-Agent>.py
```

For example, to run Random movement agent, use command below:

```bash
python randomAction.py
```

To run DDPG agent, use command below:

```bash
python DDPGAgent.py
```