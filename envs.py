import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy
import imutils
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image
import cv2
import os
train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])
import glob

class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_id:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, info = self.env.step(action)

            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                    info.get('episode', {}).get('visited_rooms', {})))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class MarioEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=False,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, sticky_action=True,
            p=0.25):
        super(MarioEnvironment, self).__init__()
        self.daemon = True
        self.env = JoypadSpace(
            gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()
    def run(self):
        super(MarioEnvironment, self).run()
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        if not os.path.exists("data"):
            os.makedirs("data")
        done = True
        min_area = 20
        max_area = 100
        subtractor = cv2.createBackgroundSubtractorKNN(history=5, detectShadows=True)
        while True:
            if done :
                self.reset()
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action
            # 4 frame skip
            reward = 0.0
            done = None
            # list_img  = os.listdir('data')
            # total_img = len(list_img)
            # img_last = total_img+1
            # #print(int(list_img[-1][:-4]),img_last)
            # img_last = str(img_last)+".jpg"
            # path = "data/"+img_last
            # print(path)
            for i in range(4):
                obs, r, done, info = self.env.step(action)
                temp_frame = cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
                temp_obs = np.array(cv2.cvtColor(obs,cv2.COLOR_BGR2RGB))
                temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
                temp_frame = cv2.GaussianBlur(temp_frame, (5, 5), 0)
                subtr_frame = subtractor.apply(temp_frame)
                #edges = cv2.Canny(temp_frame, 50, 100)
                mask_sub = cv2.bitwise_and(temp_frame, subtr_frame)
                _, thr = cv2.threshold(mask_sub, 100, 255, 0,cv2.THRESH_BINARY)
                cnts = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                for c in cnts:
                    if cv2.contourArea(c) < min_area or cv2.contourArea(c)>max_area :
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    #temp_obs = cv2.rectangle(temp_obs, (x-50, y-50), (x + w+50, y +50+ h), (0, 255, 0), 3)
                    temp_obs = temp_obs[y-5:y+h+5,x-5:x+w+5]
                    list_img = os.listdir('data')
                    total_img = len(list_img)
                    img_last = total_img + 1
                    img_last = str(img_last)+".jpg"
                    path = "data/"+img_last
                    try:
                        cv2.imwrite(path,cv2.resize(temp_obs, (40, 40)))
                    except:
                        pass
                # cv2.imshow("frame",temp_frame)
                # cv2.imshow("edge",edges)
                #cv2.imwrite(path,obs)
                if self.is_render:
                    self.env.render()
                reward += r
                if done:
                    break

            # when Mario loses life, changes the state to the terminal
            # state.
            if self.life_done:
                if self.lives > info['life'] and info['life'] > 0:
                    force_done = True
                    self.lives = info['life']
                else:
                    force_done = done
                    self.lives = info['life']
            else:
                force_done = done

            # reward range -15 ~ 15
            log_reward = reward / 15
            self.rall += log_reward

            r = int(info.get('flag_get', False))

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print(
                    "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                        self.episode,
                        self.env_idx,
                        self.steps,
                        self.rall,
                        np.mean(
                            self.recent_rlist),
                        info['stage'],
                        info['x_pos'],
                        self.max_pos))

                self.history = self.reset()

            self.child_conn.send([self.history[:, :, :], r, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)
