
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2 # communication channel dimensionality
        num_agents = 5
        world.num_agents = num_agents
        num_adversaries = 2
        num_landmarks = 1 # num_agents - num_adversaries
        execution_time = 20
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.10 if i < num_adversaries else 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1, 0.15, 0.15])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.35, 0.35, 0.85])
        
        sheep = np.random.choice(world.landmarks)
        while sheep == goal: 
            sheep = np.random.choice(world.landmarks)
        sheep.color = np.array([0.15, 0.15, 0.15])
        
        for agent in world.agents:
            agent.goal_a = goal
            
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_pos_series = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel_series = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c) # communication noise
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on STREL robustness score
        good_agents = self.good_agents(world)
        result = self.adversary_reward(agent, world) if agent.adversary else np.sum([self.agent_reward(a, world) for a in good_agents])/len(good_agents)
        #print(agent.name, ', adv? ',agent.adversary, ', reward = ',result)
        return result

        # robustness formula
        # \phi = Eventually_{[5,10]} good_agents surround_{r<min_gd2ad_dist} landmark 
        #       AND Eventually_{[5,10]} sheep reach_{r<min_dist} goal
        #       AND Always_{[0,20]} ~(adversary_agents reach_{r<min_dist} landmark)
        
        
    def agent_reward(self, agent, world):

        min_gd2gl_dist = 1.5 * agent.goal_a.size
        max_gd2gl_dist = 3 * agent.goal_a.size
        min_gl2ad_dist = agent.size + 2 * agent.goal_a.size
        min_gd2gd_dist = 1.5 * agent.size
        max_gd2gd_dist = 1.5 * agent.size
        
        # Calculate distance to goal from adversary_agents
        adversary_agents = self.adversaries(world)
        adv2gl_dist = np.mean([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])

        # Calculate distance from agent to goal 
        gd2gl_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        #gd2gl_dist = max([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        
        # Calculate minimum distance between good_agents
        gd2gd_dist_temp = []
        good_agents = self.good_agents(world)
        n_good_agents = len(good_agents)
        for a in good_agents:
            if a.name != agent.name:
                gd2gd_dist_temp.append( np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) )

        gd2gd_dist = min(gd2gd_dist_temp)
        #rob_score = min([ (gd2gl_dist - min_gd2gl_dist)/gd2gl_dist, (max_gd2gl_dist - gd2gl_dist)/gd2gl_dist, 2*(adv2gl_dist - min_gl2ad_dist)/(adv2gl_dist+min_gl2ad_dist),
        #                  (gd2gd_dist - min_gd2gd_dist)/gd2gd_dist, (max_gd2gd_dist - gd2gd_dist)/gd2gd_dist])
        rob_score = np.mean([ (max_gd2gl_dist - gd2gl_dist), (adv2gl_dist - min_gl2ad_dist),(max_gd2gd_dist - gd2gd_dist)])
        
        return rob_score
        
    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
