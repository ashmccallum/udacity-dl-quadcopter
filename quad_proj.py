# TODO: Train your agent here.
import numpy as np
from agents.agent123 import Agent
from task import Task
import matplotlib.pyplot as plt
# % matplotlib notebook

num_episodes = 100  # number of episodes
init_pose = np.array([0., 0., 5., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])  # initial velocities
init_angle_velocities = np.array([0., 0., 0.])  # initial angle velocities

task = Task(init_pose, None, None)
agent = Agent(task)
done = False

display_graph = True
display_freq = 10


# generate plot function
def plt_dynamic(x, y1, y2, color_y1='g', color_y2='b'):
    sub1.plot(x, y1, color_y1)
    sub2.plot(x, y2, color_y2)
    fig.canvas.draw()


# create plots
fig, sub1 = plt.subplots(1, 1)
sub2 = sub1.twinx()

# set plot boundaries. y1 = z, y2 = reward
time_limit = 5
y1_lower = 0
y1_upper = 20
y2_lower = -100
y2_upper = 50

sub1.set_xlim(0, time_limit)  # this is typically time
sub1.set_ylim(y1_lower, y1_upper)  # limits to your y1
sub2.set_xlim(0, time_limit)  # time, again
sub2.set_ylim(y2_lower, y2_upper)  # limits to your y2

# set labels and colors for the axes
sub1.set_xlabel('time (s)', color='k')
sub1.set_ylabel('y1-axis label', color='g')
sub1.tick_params(axis='x', colors='k')

sub1.tick_params(axis='y', colors="g")

sub2.set_ylabel('y2-axis label', color='b')
sub2.tick_params(axis='y', colors='b')

for episode in range(num_episodes + 1):
    state = agent.reset_episode()

    x, y1, y2 = [], [], []

    while done is False:

        if (episode % display_freq == 0) and (display_graph is True):
            x.append(task.sim.time)
            y1.append(task.sim.pose[2])
            y2.append(agent.total_reward)

        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

    if (episode % display_freq == 0) and (display_graph is True):
        plt_dynamic(x, y1, y2)

    print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
        episode, agent.score, agent.best_score, agent.noise_scale))
