import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import json
import logging

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from GridShield import GridShield
from copy import deepcopy
# from CustomLogger import CustomLogger

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread2", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--grid", type=bool, default="True", help="Grid snapping")
    parser.add_argument("--shielding", type=bool, default="True", help="Enable shielding")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--collisions", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, config=None, benchmark=False, multi_goal=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    print('************ Loading scenario: {}'.format(scenario_name))

    # create world
    if multi_goal:
        with open('config/'+config["particle_config"]) as f:
            config_particle = json.load(f)
        n_agents = config_particle['n_agents']
        world = scenario.make_world(n_agents, config_particle)
    else:
        world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def make_parallel(env, scenario_name=None, config=None, benchmark=False, multi_goal=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    if multi_goal:
        with open('config/'+config["particle_config"]) as f:
            config_particle = json.load(f)
        n_agents = config_particle['n_agents']
        world = scenario.make_world(n_agents, config_particle)
    else:
        world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        para = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        para = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return update_parallel(env, para)

def update_parallel(env, parallel_env):
    parallel_env.agents = deepcopy(env.agents)
    parallel_env.world = deepcopy(env.world)

    return parallel_env

def train(arglist):
    with U.single_threaded_session():

        # logger =CustomLogger(file=arglist.exp_name)
        logging.basicConfig(filename='logs/'+arglist.exp_name+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # arglist.shielding=False
        print('shielding : ', arglist.shielding)
        logging.info(f'shielding: {arglist.shielding}')

        # print('shielding : ', arglist.shielding)
        config = None
        multi_goal=False

        if arglist.scenario == 'multi_goal_spread':
            multi_goal = True
            with open('config/config.json', 'r') as f:
                config = json.load(f)
                print(f'sub-scenario: {config["particle_config"][23:]}')

        # Create environment
        env = make_env(arglist.scenario, arglist, config, (arglist.benchmark or arglist.collisions), multi_goal)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        collisions_info = []
        collisions_info_ag = [[0]*env.n]
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        if arglist.debug:
            pass
            # env.agents[0].state.p_pos = [-0., -0.6]
            # env.agents[1].state.p_pos = [0, -0.9]
            # env.agents[2].state.p_pos = [-0.1, -0.4]
            #
            # obs_n[0][2:4] = [-0. , -0.6]
            # obs_n[1][2:4] = [0, -0.9]
            # obs_n[2][2:4] = [-0.1, -0.4]

        if arglist.shielding:
            # initialize grid shields
            gridshield = GridShield(nagents=env.n, c_start=np.array(obs_n)[:, 2:4])

        if arglist.display:
            parallel_env = make_parallel(env, arglist.scenario,config, (arglist.benchmark or arglist.collisions), multi_goal)

        # interference = np.zeros([env.n, arglist.num_episodes])

        print('Starting iterations...')
        logging.info('Starting iterations...')

        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            prev = [agent.state.p_pos.flatten() for agent in env.agents]

            if arglist.shielding:
                if arglist.display:
                    parallel_env = update_parallel(env, parallel_env)
                else:
                    parallel_env = deepcopy(env)

                pre_action_n = deepcopy(action_n)
                # get a_req +
                if arglist.grid:
                    alt_obs_n, alt_rew, alt_done, _ = parallel_env.step(pre_action_n, discretize=True)
                else:
                    alt_obs_n, alt_rew, alt_done, _ = parallel_env.step(pre_action_n)

                new_pos = [agent.state.p_pos for agent in parallel_env.agents]
                if arglist.debug:
                    print(f"shield states: [{gridshield.current_state[int(gridshield.agent_pos[0][1])]}, "
                          f"{gridshield.current_state[int(gridshield.agent_pos[1][1])]}, "
                          f"{gridshield.current_state[int(gridshield.agent_pos[2][1])]}] \t\t- pos : {new_pos} "
                          f" \t\t- ag pos : {gridshield.agent_pos.flatten()}")

                logging.info(f"shield states: [{gridshield.current_state[int(gridshield.agent_pos[0][1])]},"
                          f"{gridshield.current_state[int(gridshield.agent_pos[1][1])]},"
                          f"{gridshield.current_state[int(gridshield.agent_pos[2][1])]}]\t- pos: {new_pos}"
                          f" \t- ag pos: {gridshield.agent_pos.flatten()} - action : {np.array(action_n).tolist()}")
                valid = gridshield.step(pre_action_n, np.array(obs_n)[:, 2:4], alt_done, np.array(alt_obs_n)[:, 2:4])

                punish = (~valid)  # 1 for agents that need to be punished #
                if not np.all(punish == False):
                    for a in range(env.n):
                        if punish[a]:
                            action_n[a] = np.zeros([5])
                            action_n[a][1] = - obs_n[a][0] * 1.5  # reversing agent momentum and inertia to standstill
                            action_n[a][3] = - obs_n[a][1] * 1.5

                # if len(interference[punish]) > 0:
                #     idx_values = np.where(punish == True)[0]
                #     for idx in idx_values:
                #         interference[idx][e] += 1

            # environment step
            if arglist.grid:
                new_obs_n, rew_n, done_n, info_n = env.step(action_n, discretize=True)
            else:
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            post = [agent.state.p_pos for agent in env.agents]

            episode_step += 1

            # print(episode_step)
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            #  check rewards + do shield double update.
            # Exrta shield update
            if arglist.shielding:
                if not np.all(punish == False): # only if one agent or more was modified
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], pre_action_n[i], alt_rew[i], alt_obs_n[i], alt_done[i], terminal)

            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                if arglist.debug:
                    print('----------- end of episode ------ ')
                logging.info('------------------ end of episode -------------- ')
                # print('agent pos: ', np.array(obs_n)[:, 2:4].flatten())
                if arglist.shielding:
                    gridshield.reset(np.array(obs_n)[:, 2:4])  # TODO fix to take start positions

                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                collisions_info_ag.append([0]*env.n)

            # increment global step counter
            train_step += 1

            # aggregate collision information
            if arglist.collisions:
                col = []
                for i, info in enumerate(info_n['n']):
                    col.append(info_n['n'][i][1])
                    collisions_info_ag[-1][i] += info[1] # TODO test this
                    if info_n['n'][i][1] != 0 and arglist.shielding:
                        print('collision not 0')
                        logging.error(' collision not 0')
                collisions_info.append(col)

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                print ("file: ", rew_file_name)
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                if arglist.collisions: 
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    ag_file_name = arglist.benchmark_dir + arglist.exp_name + '_ag.pkl'
                    with open(file_name, 'wb') as fp:
                        pickle.dump(collisions_info[:-1], fp)
                    with open(ag_file_name, 'wb') as fp:
                        pickle.dump(collisions_info_ag[:-1], fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))

                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
