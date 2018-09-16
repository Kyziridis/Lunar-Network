#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:41:53 2018

@author: dead
"""

#from __future__ import print_function
import keras
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import gym
from keras.callbacks import ModelCheckpoint


    
class AGENT:
    def __init__(self, nx, ny, lr, gamma):
        self.nx = nx  #  Observation array length   nx = env.observation_space.shape[0]
        self.ny = ny  #   Action space length       ny = env.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.e = 1.0
        self.e_= 0.01
        self.dc= 0.95
        #self.l_link =l_link 
        #self.s_link =s_link 
        
        self.model = self.MODEL() # Call function model to build the model        
        #self.model.load_weights("weights.best.hdf5")
        # Load
        #self.load(self.l_link)
        self.ep_obs, self.ep_rewards, self.ep_action = [], [], []
            
    def choose_action(self,observation, dis):
        
        if np.random.rand() <= self.e : 
            probs = self.model.predict([observation,dis])
            action = np.random.choice(len(range(4)), p=probs[0])
            return action
            
        probs = self.model.predict([observation,dis])    
        action = np.argmax(probs[0])
        return action
        
    
    def storing(self, observation, action, reward ):
        self.ep_obs.append(observation)
        self.ep_action.append(to_categorical(action,4))
        self.ep_rewards.append(reward)
        
               
    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.ep_rewards)
        cumulative = 0
        for t in reversed(range(len(self.ep_rewards))):
            cumulative = cumulative * self.gamma + self.ep_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def myloss(self,y_true, y_pred): # (act,Z3)
        #y_true, y_pred = K.transpose(y_true), K.transpose(y_pred)        
        neg_log_prob = keras.losses.categorical_crossentropy(y_true,y_pred)
        lala = self.dis*neg_log_prob
        out = K.mean(lala)
        return out
    
    #def load(self,name):
    #    self.model.load_weights(name)

    #def save(self,name):
    #    self.model.save_weights(name)

    def MODEL(self):                     
        # Build Network
        
        self.ini = keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=666)
        #self.b_ini = keras.initializers.Constant(value=0.1)
        self.b_ini = keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=666)
        
        self.x = Input(shape=(8,))  
        self.l1 = Dense(10, activation='relu', input_shape=(8,), init='lecun_uniform')(self.x)
        self.l2 = Dense(10, activation='relu', init='lecun_uniform')(self.l1)
        self.Z3 = Dense(4, activation='softmax',init='lecun_uniform')(self.l2)
        
        # Create a second input for the new_normalized rewards to update myloss
        self.dis = Input(shape=(1,))
                
        self.model = Model(inputs=[self.x,self.dis], outputs=self.Z3)
        self.model.compile(loss=self.myloss, optimizer=Adam(lr=self.lr), metrics=['categorical_accuracy'])
            
        return self.model
    
    
    def TRAIN(self):
        dis = self.discount_and_norm_rewards()    
        dis = np.vstack(dis)
        
        
        
        filepath="weights2.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='self.myloss', verbose=1, save_best_only=True, mode='max', period=1)
        callbacks_list = [checkpoint]
        
        history = self.model.fit(x=[np.vstack(self.ep_obs), dis], y=np.vstack(self.ep_action),\
                            batch_size=1,\
                            verbose=0,\
                            epochs=1, callbacks=callbacks_list)        
        self.ep_obs, self.ep_rewards, self.ep_action = [], [], []
        
        if self.e > self.e_:
            self.e *= self.dc
        
        #self.save(self.s_link)
        return history



if __name__ == '__main__':
    
    rendering = input("Visualize rendering ? [y/n]:  ")
    
    #load_version = 1
    #save_version = 1 + load_version 
    #l_link = "output/weights/LunarLander/{}/LunarLander-v2.h5".format(load_version)
    #s_link = "output/weights/LunarLander/{}/LunarLander-v2.h5".format(save_version)
    
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment
    EPISODES = 5000    # Number of episodes
    
    env = gym.make('LunarLander-v2')
    env = env.unwrapped
    
    # Observation and Action array length
    nx = env.observation_space.shape[0] 
    ny = env.action_space.n
    dis = np.array([1,2,3]) # initial for randomness
    lr = 0.001
    gamma = 0.99
    agent = AGENT(nx,ny, lr, gamma)
    
    rewards_over_time = []
    error = []
    seed = np.random.seed(666)
        
    print("-----------------------------------")        
    print("Environment Observation_space: ", env.observation_space)
    print("Environment Action_space: ", env.action_space) 
    print("-----------------------------------\n")
    w = 0
    # Start running the episodes        
    for i in range(EPISODES): 
        observation = env.reset()         
                        
        start = time.time()
        
        while True:            
            if RENDER_ENV==True:
                env.render()
            
            observation = observation.reshape(1,-1)
            
            action = agent.choose_action(observation, dis)

            observation_new, reward, flag, inf = env.step(action)
            
            # Append
            agent.storing(observation, action, reward)   
            
            end = time.time()
            time_space = end - start
            
            if time_space > 15:
                flag = True
          
            # Sum the episode rewards
            ep_rew_total = sum(agent.ep_rewards) 
            if ep_rew_total < -250:
                flag = True
            
            if flag==True:
                rewards_over_time.append(ep_rew_total)
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                if ep_rew_total >=200 :
                    w = w + 1
                
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Episode: ", i)
                print("Time: ", np.round(time_space, 2),"secs")
                print("Reward:", ep_rew_total)
                print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(episode_max))
                print("Times win: " + str(w))
                
                # Start training the Neural Network
                hist= agent.TRAIN()
                
                                           
                error.append(hist.history['loss'])
                
                if max_reward > RENDER_REWARD_MIN: RENDER_ENV = True
                
                break
            
            observation = observation_new
            
            
            
            
plt.figure(figsize=(6,5))
plt.plot(l)
plt.xlabel("Episodes")
plt.ylabel("Epsilon value")
plt.title("Epsilon Vs Episodes")
plt.show()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


