import pandas as pd
import numpy as np
import random
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
import math
import matplotlib.pyplot as plt

#
# Read in raw data
#
d_time = pd.read_excel("InputData_HouseholdStates.xls", sheet_name = "time", header=None)
d_p_base = pd.read_excel("InputData_HouseholdStates.xls", sheet_name = "pbase", header=None)
d_p_ctrl_min = pd.read_excel("InputData_HouseholdStates.xls", sheet_name = "pmin", header=None)
d_p_ctrl_max = pd.read_excel("InputData_HouseholdStates.xls", sheet_name = "pmax", header=None)

#
# Calculate number of homes
#
n = len(d_time.columns)

#
# Compile household data in usable format
#


input_data = np.empty((d_time.shape[0],5,n),dtype=object) # 5 is for the 3 power variables, time & electricity rate


def f_electricity_price(t):

    tod = t-(24*(math.ceil(t/24)-1))

    if tod < 1 or tod > 24:
            roh = "error"

    elif tod >= 19 and tod < 22:       
            roh = 4.5

    elif tod >= 16 and tod < 19:
            roh = 4

    elif (tod >= 7 and tod < 13 ) or (tod >= 22) or (tod < 4):
            roh = 2.5

    else:
            roh =1
                
    return roh

for i in range(0,n): # iterate ober household data
    for j in range (0,(len(d_time.index))): # iterate over timestep
        input_data[j][0][i] = d_time[i][j]# time stamp for house i
        input_data[j][1][i] = f_electricity_price(d_time[i][j])# electricity price for house i
        input_data[j][2][i] = d_p_base[i][j] # baseload power for house i
        input_data[j][3][i] = d_p_ctrl_min[i][j] # minimum controllable load for house i
        input_data[j][4][i] = d_p_ctrl_max[i][j] # maximum controllable load for house i
       
        
# TEST POINT
#print("Section 2: input_data table", input_data)
#
# Initalize values
#
t = 1
epsilon = 10**-3

#
# Federated learning variables
#
num_groups = int(4)
n_per_group = int(n/num_groups)
group_index = np.linspace(0, 24, 4)
household_data_groups = np.empty((4,5,8),dtype=object)

#
# DNN variables
#           
action_probs_history = []
state_output_history = []
critic_value_history = []
reward_history = []
value_history = []
prob_history = []
delta_history = []
P_ctrl_history = []
P_ctrl_cost_history = []
load_checkpoint = False

#
# Federated learning
#
for i in range(0,num_groups):
    t_time = input_data[t,0,int(group_index[i]):int(group_index[i]+n_per_group)]
    t_elec_price = input_data[t,1,int(group_index[i]):int(group_index[i]+n_per_group)]
    t_base = input_data[t,2,int(group_index[i]):int(group_index[i]+n_per_group)]
    t_min = input_data[t,3,int(group_index[i]):int(group_index[i]+n_per_group)]
    t_max = input_data[t,4,int(group_index[i]):int(group_index[i]+n_per_group)]
    household_data_groups[i] = [t_time, t_elec_price, t_base, t_min, t_max]
    
random_data = household_data_groups.shape
random_data = np.random.rand(*random_data)


system_data = household_data_groups
system_data = np.append(system_data, random_data, axis = 0)

system_data_clean = system_data
delete_index = []
it = 0
for i in range(system_data.shape[0]):
    for j in range(random_data.shape[0]):
        if (random_data [j,:,:] == system_data[i,:,:]).any():
            delete_index.insert(it,i)
            it +=1
                   
system_data_clean = np.delete(system_data,delete_index,0)            

lamda_bar = np.empty(3)
gamma_bar = np.empty(3)

base = np.sum(system_data[:,2,:])
min_ctrl = np.sum(system_data[:,3,:])
max_ctrl = np.sum(system_data[:,4,:])

lamda_bar = [base,min_ctrl,max_ctrl]
gamma_bar = lamda_bar

#
# Actor critic model
#

class ActorCriticNetwork(keras.Model):
    def __init__(self, num_actions, num_hidden1=6, num_hidden2=18,num_hidden3=6,
            name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden2 = num_hidden2
        self.num_actions = num_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_DR')

        self.fc1 = Dense(self.num_hidden1, activation='relu')
        self.fc2 = Dense(self.num_hidden2, activation='relu')
        self.fc3 = Dense(self.num_hidden2, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(num_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi
    
    
class Agent:
    def __init__(self, alpha=0.0003, beta=0.99, num_actions=3):
        self.beta = beta
        self.num_actions = num_actions
        self.action = None
        self.delta = 0
        self.action_space = [i for i in range(self.num_actions)]

        self.actor_critic = ActorCriticNetwork(num_actions=num_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
        
    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) 
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.beta*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
            self.delta = delta

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))


agent = Agent(alpha=0.003, num_actions=100)
for i in range (0,(len(d_time.index)-1)): 
    observation = [input_data[i,1:5,0]]
    action = agent.choose_action(observation)
    P_ctrl = input_data[i,2,0] + action * (input_data[i,4,0] - input_data[i,3,0])
    observation_ = [input_data[i+1,1:5,0]]
    # calculate discomfort rate as mentioned in paper equation (5)
    if (t < 6 or t > 21):
        discomfort = random.uniform(0.1,1)
    else:
        discomfort = 5
    reward = -(input_data[i,1,0]* P_ctrl + discomfort*abs(input_data[i,4,0]-P_ctrl))
    reward_history.append(reward)
    val, prob = agent.actor_critic.call(tf.convert_to_tensor([observation], dtype=tf.float32))
    P_ctrl_cost = input_data[i,1,0]* P_ctrl
    delta_history.append((agent.delta))
    value_history.append(val)
    prob_history.append(prob)
    P_ctrl_history.append(P_ctrl)
    P_ctrl_cost_history.append(P_ctrl_cost)
    done = False
    agent.learn(observation, reward, observation_, done)
    agent.save_models()
                
#
# Plotting figures for result
#

val_history = np.array(value_history)
val_history=((val_history.reshape(2399,1)))


tot_days = math.floor(input_data[-1,0,0]/24)
p_cntrl_day_hist = P_ctrl_history[2375:2399]
p_base_day_hist = input_data[2376:2400,2,0]
p_max_day_hist =input_data[2376:2400,4,0]

p_cntrl_day_hist = np.array(p_cntrl_day_hist).reshape(24,1)
p_base_day_hist = np.array(p_base_day_hist).reshape(24,1)
p_max_day_hist = np.array(p_max_day_hist).reshape(24,1)

p_base_days = []
p_max_days = []
p_ctrl_days = []
p_tot_days = []
p_tot_cost_days = []
p_max_cost_days = []
val_function_days = []
days = []
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
for i in range(1, tot_days):
    days.append(i)
    data_start = (((i-1)*24))
    data_end = i*24-1

    p_base_day = sum(input_data[data_start:data_end,2,0])
    p_base_days.append(p_base_day) 
    p_max_day = sum(input_data[data_start:data_end,4,0])
    p_max_days.append(p_max_day) 
    p_ctrl_day = sum(P_ctrl_history[data_start:data_end])
    p_ctrl_day_filt = min((p_ctrl_day+p_base_day), (p_max_day+p_base_day))
    p_ctrl_days.append(p_ctrl_day_filt) 
    p_tot_days.append(p_max_day+p_base_day)
    
    p_max_cost_day = sum(np.multiply((input_data[data_start:data_end,2,0] + input_data[data_start:data_end,4,0]), input_data[data_start:data_end,1,0]))
    p_tot_cost_day = sum(P_ctrl_cost_history[data_start:data_end]) + sum(np.multiply((input_data[data_start:data_end,2,0]), input_data[data_start:data_end,1,0]))
    p_tot_cost_filt = min(p_tot_cost_day, p_max_cost_day)/100
    p_tot_cost_days.append(p_tot_cost_filt)
    p_max_cost_days.append(p_max_cost_day/100)
    
    val_function_day = p_max_cost_day/100 + (p_max_cost_day/100)*np.average(val_history[data_start:data_end])/100
    val_function_days.append(val_function_day)

    


plt.figure(figsize=(15, 5))
plt.bar(days[0:39], p_max_cost_days[0:39], label="Without Load Scheduling")
plt.bar(days[0:39], p_tot_cost_days[0:39], label="With Load Scheduling")
plt.title("Expected Daily Total Cost [$/household]")
plt.xlabel("Day")
plt.ylabel("cost [$]")
plt.legend()
plt.show() 

plt.figure(figsize=(15, 5))
plt.plot(days[0:39], p_tot_cost_days[0:39])
plt.title("Cost per Household per Day [$]")
plt.xlabel("Day")
plt.ylabel("cost [$]")
plt.show() 

plt.figure(figsize=(15, 5))
plt.plot(days, val_function_days)
plt.xlabel("Day")
plt.ylabel("Value Function [$]")
plt.show() 


plt.figure(figsize=(15, 5))
x =np.arange(1,25,1)
y1 = (np.add(p_max_day_hist, p_base_day_hist))
y2 = (np.add(p_cntrl_day_hist, p_base_day_hist))

p1 = plt.plot(x, y1, label="Without Load Scheduling")
plt.plot(x, y2, label="With Load Scheduling")
plt.xlabel("Hour")
plt.ylabel("Load Demand [kW]")
plt.legend()
plt.show() 

demand_diff = np.multiply(np.divide(np.subtract(y1,y2), y1),100)
demand_diff_ave = (np.average(demand_diff))
max_demand_diff = max(demand_diff)
min_demand_diff = min(demand_diff)

print("Average electricity consumption reduced:",demand_diff_ave, "%")
