'''Pure Tensorflow implementation. Includes Basic Dueling Double DQN and Spatial Attention DQN.'''

from deeprl_prj.policy import *
from deeprl_prj.objectives import *
from deeprl_prj.preprocessors import *
from deeprl_prj.utils import *
from deeprl_prj.core import *
from helper import *
from tqdm import tqdm
import numpy as np
from tensorflow.python.ops.distributions.normal import Normal
import sys
from gym import wrappers
import tensorflow as tf
#import skimage.transform

"""Main DQN agent."""

class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, myScope):
        self.imageIn = tf.placeholder(shape=[None, 84, 84, num_frames], dtype=tf.float32)
        
        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,1])
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn, num_outputs=32, \
            kernel_size=[8, 8], stride=[4, 4], padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv1')
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1, num_outputs=64, \
            kernel_size=[4, 4], stride=[2, 2], padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv2')
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2, num_outputs=64, \
            kernel_size=[3, 3], stride=[1, 1], padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv3')  # (None, 10, 7, 7, 64)
        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)
        #batch_size = self.batch_size.get_shape().num_elements()
        #a = self.conv3.get_shape().num_elements()


        self.L = 7 * 7
        self.D = 64
        #self.T = 6
        self.H = 256
        #self.variance = 0.1

        self.channel = 64
        self.loc_dim = 5
        self.img_size = 7
        self.pth_size = 3
        #self.sigma = 1
        #self.g_size = 128
        #self.l_size = 128
        self.num_glimpses = 4
        self.glimpse_output_size = 256
        self.cell_size = 256
        #self.variance = 0.22
        #self.max_gradient_norm = 2.0

        # self.selector=args.selector
        # self.temporal=True
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        #self.features = tf.reshape(self.conv3, [self.batch_size, self.L, self.D])
        #self.features_list = tf.split(self.features, num_frames, axis=1)
        # print(len(self.features_list), self.features_list[0].get_shape().as_list()) # 10 [None, 1, 49, 64]
        self.alpha_list = []

        #lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.H)
        #c, h = lstm_cell.zero_state(self.batch_size, tf.float32)

        lstm_cell_att = tf.nn.rnn_cell.LSTMCell(num_units=self.H)
        c_att, h_att = lstm_cell_att.zero_state(self.batch_size, tf.float32)

        for t in range(self.num_glimpses):

            loc = self._attention_layer(h_att, myScope=myScope, reuse=(t != 0))

            context = self.GlimpseNetwork(self.conv3, loc, myScope=myScope)

            with tf.variable_scope(myScope + '_lstmCell', reuse=(t != 0)):
                
                _, (c_att, h_att) = lstm_cell_att(inputs=context, state=[c_att, h_att])
                #_, (c, h) = mcell(inputs=context, state=[c, h])
                # print("========== h ", h.get_shape().as_list())
                #hists.append(h)
	
        
        self.rnn = h_att
       

        self.Qout = tf.contrib.layers.fully_connected(self.rnn, num_actions, activation_fn=None)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        gradients = self.optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        self.updateModel = self.optimizer.apply_gradients(capped_gradients)

    def RetinaSensor(self, normLoc):
        A = B = self.img_size
        N = self.pth_size
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(normLoc,5, 1)
        gx = (A + 1) / 2 * (gx_ + 1)
        gy = (B + 1) / 2 * (gy_ + 1)
        sigma2 = tf.exp(log_sigma2)
        #sigma2 -= 0.9
        delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N


        return filterbank(gx, gy, sigma2, delta, N)+ (tf.exp(log_gamma),)



    def GlimpseNetwork(self, feature_map, locs, myScope=''):
        B=A=self.img_size
        N = self.pth_size
        # with tf.variable_scope(myScope,'GlimpseNetwork',reuse=tf.AUTO_REUSE):
        #     # layer 1
        #     g1_w = tf.get_variable('g1_w', [64 * 9, self.glimpse_output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #     g1_b = tf.get_variable('g1_b', [self.glimpse_output_size], dtype=tf.float32, initializer=tf.initializers.zeros)
        #     l1_w = tf.get_variable('l1_w', [2 * self.loc_dim, self.glimpse_output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #     l1_b = tf.get_variable('l1_b', [self.glimpse_output_size], dtype=tf.float32, initializer=tf.initializers.zeros)
        #     # layer 2
        #     # g2_w = tf.get_variable('g2_w', [self.g_size, self.glimpse_output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #     # g2_b = tf.get_variable('g2_b', [self.glimpse_output_size], dtype=tf.float32, initializer=tf.initializers.zeros)
        #     # l2_w = tf.get_variable('l2_w', [self.l_size, self.glimpse_output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #     # l2_b = tf.get_variable('l2_b', [self.glimpse_output_size], dtype=tf.float32, initializer=tf.initializers.zeros)

        Fx, Fy, gamma= self.RetinaSensor(locs)

        
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])    #[?, 7, 3]
        

        feature_map = tf.transpose(feature_map, perm=[0, 3, 1, 2]) #[?, 64, 7, 7]
        img = tf.reshape(feature_map, [-1, 64*B, A])

        img_Fxt = tf.matmul(img, Fxt)#[?, 64*7, 3]                  [?, 7, 3, 64]
        img_Fxt = tf.reshape(tf.transpose(tf.reshape(img_Fxt, [-1, 64, 7, 3]), [0, 2, 3, 1]), [-1, 7, 3*64])
        glimpse = tf.matmul(Fy, img_Fxt)
        
       
        glimpse = tf.reshape(glimpse, [-1, N * N * 64])


        x = glimpse * tf.reshape(gamma, [-1, 1]) #batch x (read_n*read_n)

        return glimpse

    def _attention_layer(self, h, myScope, reuse=False):
        with tf.variable_scope(myScope + '_attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.loc_dim], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.loc_dim], initializer=self.const_initializer)
            #w_att = tf.get_variable('w_att', [self.D, self.loc_dim], initializer=self.weight_initializer)

            loc = tf.matmul(h, w) + b  # (N, L, D)

            return loc

eps = 1e-8
def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    #gx += 12
    #gy += 12
    A = B = 7
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the class 
    provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self, args, num_actions):
        self.num_actions = num_actions
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)
        self.atari_processor = AtariPreprocessor()
        self.memory = ReplayMemory(args)
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon, args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.output_path_videos = args.output + '/videos/'
        self.output_path_images = args.output + '/images/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode

        self.h_size = 512
        self.tau = 0.001
        tf.reset_default_graph()
        #We define the cells for the primary and target q-networks
        #cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        #cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        self.q_network = Qnetwork(args=args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, myScope="QNet")
        self.target_network = Qnetwork(args=args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, myScope="TargetNet")
        
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        trainables = tf.trainable_variables()

        self.targetOps = updateTargetGraph(trainables, self.tau)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        updateTarget(self.targetOps, self.sess)
        self.writer = tf.summary.FileWriter(self.output_path)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        # return self.q_network.predict_on_batch(state)
        # print state.shape
        # Qout = self.sess.run(self.q_network.rnn_outputs,\
        #             feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        Qout = self.sess.run(self.q_network.Qout,\
                    feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        return Qout

    def select_action(self, state, is_training = True, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state)
        if is_training:
            if kwargs['policy_type'] == 'UniformRandomPolicy':
                return UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                return self.policy.select_action(q_values, is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            return GreedyPolicy().select_action(q_values)

    def update_policy(self, current_sample):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch_size = self.batch_size

        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = self.memory.sample(batch_size)
            samples = self.atari_processor.process_batch(samples)

            states = np.stack([x.state for x in samples])
            actions = np.asarray([x.action for x in samples])
            # action_mask = np.zeros((batch_size, self.num_actions))
            # action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            # next_qa_value = self.target_network.predict_on_batch(next_states)
            next_qa_value = self.sess.run(self.target_network.Qout,
                    feed_dict={self.target_network.imageIn: next_states, self.target_network.batch_size:batch_size})

        if self.enable_ddqn:
            # qa_value = self.q_network.predict_on_batch(next_states)
            qa_value = self.sess.run(self.q_network.Qout,
                    feed_dict={self.q_network.imageIn: next_states, self.q_network.batch_size:batch_size})
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        # print rewards.shape, mask.shape, next_qa_value.shape, batch_size
        target = rewards + self.gamma * mask * next_qa_value

        loss, _, rnn = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn],
                    feed_dict={self.q_network.imageIn: states, self.q_network.batch_size:batch_size,
                    self.q_network.actions: actions, self.q_network.targetQ: target})
        
        return loss, np.mean(target)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        self.is_training = True
        print("Training starts.")
        #self.save_model(0)
        eval_count = 0

        state = env.reset()
        burn_in = True
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0
        count = 0
        patience = 20
        max_episode_reward_mean = -1

        for t in tqdm(range(self.num_burn_in + num_iterations)):
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"
            action = self.select_action(action_state, self.is_training, policy_type = policy_type)
            processed_state = self.atari_processor.process_state_for_memory(state)

            state, reward, done, info = env.step(action)

            processed_next_state = self.atari_processor.process_state_for_network(state)
            action_next_state = np.dstack((action_state, processed_next_state))
            action_next_state = action_next_state[:, :, 1:]

            processed_reward = self.atari_processor.process_reward(reward)

            self.memory.append(processed_state, action, processed_reward, done)
            current_sample = Sample(action_state, action, processed_reward, action_next_state, done)
            
            if not burn_in: 
                episode_frames += 1
                episode_reward += processed_reward
                episode_raw_reward += reward
                if episode_frames > max_episode_length:
                    done = True

            if done:
                #with open('reward_spatial_dqn2lstm_global_seaquest4_4.txt', 'a') as file:
                    #file.write(str(episode_raw_reward) + '\n')
                # adding last frame only to save last state
                last_frame = self.atari_processor.process_state_for_memory(state)
                # action, reward, done doesn't matter here
                self.memory.append(last_frame, action, 0, done)
                if not burn_in:
                    avg_target_value = episode_target_value / episode_frames
                    print(">>> Training: time %d, episode %d, length %d, reward %.0f, raw_reward %.0f, loss %.4f, target value %.4f, policy step %d, memory cap %d" % 
                        (t, idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss, 
                        avg_target_value, self.policy.step, self.memory.current))
                    # sys.stdout.flush()
                    # save_scalar(idx_episode, 'train/episode_frames', episode_frames, self.writer)
                    # save_scalar(idx_episode, 'train/episode_reward', episode_reward, self.writer)
                    # save_scalar(idx_episode, 'train/episode_raw_reward', episode_raw_reward, self.writer)
                    # save_scalar(idx_episode, 'train/episode_loss', episode_loss, self.writer)
                    # save_scalar(idx_episode, 'train_avg/avg_reward', episode_reward / episode_frames, self.writer)
                    # save_scalar(idx_episode, 'train_avg/avg_target_value', avg_target_value, self.writer)
                    # save_scalar(idx_episode, 'train_avg/avg_loss', episode_loss / episode_frames, self.writer)
                    episode_frames = 0
                    episode_reward = .0
                    episode_raw_reward = .0
                    episode_loss = .0
                    episode_target_value = .0
                    idx_episode += 1
                burn_in = (t < self.num_burn_in)
                state = env.reset()
                self.atari_processor.reset()
                self.history_processor.reset()

            if not burn_in:
                if t % self.train_freq == 0:
                    loss, target_value = self.update_policy(current_sample)
                    episode_loss += loss
                    episode_target_value += target_value
                # update freq is based on train_freq
                if t % (self.train_freq * self.target_update_freq) == 0:
                    # self.target_network.set_weights(self.q_network.get_weights())
                    updateTarget(self.targetOps, self.sess)
                    print("----- Synced.")
                # if t % self.save_freq == 0:
                #     self.save_model(idx_episode)
                if t % (self.eval_freq * self.train_freq) == 0:
                    episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 10, eval_count, max_episode_length, True)
                    # save_scalar(t, 'eval/eval_episode_reward_mean', episode_reward_mean, self.writer)
                    # save_scalar(t, 'eval/eval_episode_reward_std', episode_reward_std, self.writer)
            




    def save_model(self, idx_episode):
        safe_path = self.output_path + "/qnet" + str(idx_episode) + ".cptk"
        self.saver.save(self.sess, safe_path)
        # self.q_network.save_weights(safe_path)
        print("Network at", idx_episode, "saved to:", safe_path)

    def restore_model(self, restore_path):
        model_file = tf.train.latest_checkpoint('ckpt/')
        #saver.restore(sess, model_file)
        self.saver.restore(self.sess, model_file)
        print("+++++++++ Network restored from: %s", model_file)

    def evaluate(self, env, num_episodes, eval_count, max_episode_length=None, monitor=True):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("Evaluation starts.")
        #plt.figure(1, figsize=(40, 20))

        is_training = False
        # if self.is_training is False:
        #     # self.q_network.load_weights(self.load_network_path)
        #     # print("Load network from:", self.load_network_path)
        #     self.restore_model(self.load_network_path)
        #if monitor:
            #env = wrappers.Monitor(env, self.output_path_videos, video_callable=lambda x:True, resume=True)
        state = env.reset()

        idx_episode = 1
        episode_frames = 0
        episode_reward = np.zeros(num_episodes)
        t = 0

        while idx_episode <= num_episodes:
            t += 1
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            action = self.select_action(action_state, is_training, policy_type = 'GreedyEpsilonPolicy')

            # action_state_ori = self.history_processor.process_state_for_network_ori(
            #     self.atari_processor.process_state_for_network_ori(state))
            # print "state.shape", state.shape
            # print "action_state_ori.shape", action_state_ori.shape

            #if np.random.random() < 1e-3:
                #alpha_list = self.sess.run(self.q_network.alpha_list,\#todo!!!!!
                            #feed_dict={self.q_network.imageIn: action_state[None, :, :, :], self.q_network.batch_size:1})
                # print alpha_list, len(alpha_list), alpha_list[0].shape #10 (1, 49)
                #for alpha_idx in range(len(alpha_list)):
                    #plt.subplot(2, len(alpha_list)//2, alpha_idx+1)
                    #img = action_state_ori[:, :, :, alpha_idx] #(210, 160, 3)
                    #plt.imshow(img)
                    #alp_curr = alpha_list[alpha_idx].reshape(7, 7)
                    #alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=22, sigma=20)
                    #plt.imshow(scipy.misc.imresize(alp_img, (img.shape[0], img.shape[1])), alpha=0.7, cmap='gray')
                    #plt.axis('off')
                # plt.show()
                # plt.canvas.draw()
                #plt.savefig('%sattention_ep%d-frame%d.png'%(self.output_path_images, eval_count, episode_frames))
            
            state, reward, done, info = env.step(action)
            episode_frames += 1
            episode_reward[idx_episode-1] += reward 
            if episode_frames > max_episode_length:
                done = True
            if done:
                print("Eval: time %d, episode %d, length %d, reward %.0f. @eval_count %s" %
                    (t, idx_episode, episode_frames, episode_reward[idx_episode-1], eval_count))
                eval_count += 1
                # save_scalar(eval_count, 'eval/eval_episode_raw_reward', episode_reward[idx_episode-1], self.writer)
                # save_scalar(eval_count, 'eval/eval_episode_raw_length', episode_frames, self.writer)
                # sys.stdout.flush()
                state = env.reset()
                episode_frames = 0
                idx_episode += 1
                self.atari_processor.reset()
                self.history_processor.reset()


        reward_mean = np.mean(episode_reward)
        reward_std = np.std(episode_reward)
        with open('reward_mean_local_dqn.txt','a') as file:
            file.write(str(reward_mean)+'\n')
        # with open('reward_std_spatial_2lstm_global_seaquest4_4.txt', 'a') as file:
        #     file.write(str(reward_mean) + '\n')
        print("Evaluation summury: num_episodes [%d], reward_mean [%.3f], reward_std [%.3f]" %
            (num_episodes, reward_mean, reward_std))
        #sys.stdout.flush()

        return reward_mean, reward_std, eval_count
