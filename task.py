import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)

        # """
        #     task.sim.pose (the position of the quadcopter in ( x,y,z ) dimensions and the Euler angles)
        #     task.sim.v (the velocity of the quadcopter in ( x,y,z ) dimensions)
        #     task.sim.angular_v (radians/second for each of the three Euler angles)
        # """
        
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 300
        self.action_high = 1000
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, old_angular_v, old_v):
        """Uses current pose of sim to return reward."""
        distance_from_target = sigmoid(sum(abs(self.sim.pose[:3] - np.float32(self.target_pos))) / 3)

        x_distance_from_target = abs(self.sim.pose[0] - np.float32(self.target_pos[0]))
        y_distance_from_target = abs(self.sim.pose[1] - np.float32(self.target_pos[1]))
        z_distance_from_target = abs(self.sim.pose[2] - np.float32(self.target_pos[2]))
        
        # punish large deltas in euler angles and velocity to produce smooth flight
        euler_change = sigmoid(sum(abs(old_angular_v - self.sim.angular_v)))
        velocity_change = sigmoid(sum(abs(old_v - self.sim.v)))
        
        reward = 1.0 - distance_from_target

        # penalise rewards further from end time
        # TODO: implement this so it works for each time step. Also, the logic of this is just wrong...
        # if self.sim.time < self.sim.runtime:
            # reward -= reward / max(5 - (self.sim.runtime - self.sim.time), 2)
            # reward -= reward / 2

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            old_pose = self.sim.pose
            old_angular_v = self.sim.angular_v
            old_v = self.sim.v
            
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(old_angular_v, old_v)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
