#!/usr/bin/env python


from __future__ import print_function

import os
import tensorflow as tf
from model import DoomNeuralNetwork
import random
from collections import namedtuple

import numpy
from tqdm import tqdm
from vizdoom import *
from visualization import plot_running_avg


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'new_state'])


class Memory:
    def __init__(self, experience_size=100, experience_sample=4):
        self.experience_size = experience_size
        self.experience_sample = experience_sample
        self.buffer = []

    def sample(self):
        return random.sample(self.buffer, self.experience_sample)

    def add_sample(self, state, action, reward, new_state):
        if len(self.buffer) >= self.experience_size:
            self.buffer.pop()
        self.buffer.insert(0, Experience(state, action, reward, new_state))


def init_game():
    game = DoomGame()
    game.set_doom_scenario_path(os.sep.join(['resources', 'scenarios', 'basic.wad']))
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
    #game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_game_variable(GameVariable.AMMO2)
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    game.init()
    return game


def convert_image(image):
    image = image / 255.0
    return image.reshape((1, image.shape[0], image.shape[1], 1)).astype(numpy.float32)


def to_one_hot(actions):
    max_positions = numpy.argmax(actions, axis=1)
    one_hot_actions = numpy.zeros_like(actions)
    one_hot_actions[range(actions.shape[0]), max_positions] = 1.0
    return one_hot_actions


def get_states(samples):
    state_shape = samples[0].new_state.shape
    states = numpy.empty((len(samples), state_shape[1], state_shape[2], state_shape[3]))
    for i in range(len(samples)):
        states[i] = samples[i].new_state
    return states


def get_actions(samples):
    actions = numpy.empty((len(samples), len(samples[0].action)))
    for i in range(len(samples)):
        actions[i] = samples[i].action
    return actions


def get_rewards(samples):
    rewards = numpy.empty((len(samples), 1))
    for i in range(len(samples)):
        rewards[i] = samples[i].reward
    return rewards


def update(model, target_model, memory, gamma):
    if len(memory.buffer) >= memory.experience_sample:
        samples = memory.sample()
        # Get states tensor
        states = get_states(samples)
        # Get action predictions
        next_action_predictions = target_model.predict(states)
        # Get actions tensor
        actions = get_actions(samples)
        # Get rewards tensor
        rewards = get_rewards(samples)
        state_action_values = numpy.zeros_like(actions)
        max_positions = numpy.argmax(actions, axis=1)
        state_action_values[range(len(samples)), max_positions] = \
            (rewards + gamma * numpy.max(next_action_predictions)).reshape((-1,))
        model.update(states, state_action_values, to_one_hot(actions))


def play_one_episode(session, game, epsilon, gamma=0.99, max_steps=10000, experience_size=2, experience_sample=2,
                     use_target_model=True):
    total_reward = 0
    dims = (None, 120, 160, 1)
    model = DoomNeuralNetwork(session, dims, game.get_available_buttons_size())
    if use_target_model:
        target_model = DoomNeuralNetwork(session, dims, game.get_available_buttons_size())
    else:
        target_model = model
    session.run(tf.global_variables_initializer())
    game.new_episode()
    time_step = 0
    memory = Memory(experience_size, experience_sample)
    state = convert_image(game.get_state().screen_buffer)
    while not game.is_episode_finished() and max_steps > time_step:
        action = model.sample_action(state, epsilon)
        reward = game.make_action(action)
        total_reward += reward
        if game.get_state() is None:
            break
        next_state = convert_image(game.get_state().screen_buffer)
        # Save experience
        memory.add_sample(state, action, reward, next_state)
        # Update model
        update(model, target_model, memory, gamma)
        # update target model
        if not model == target_model and time_step % 100 == 0:
            update(target_model, target_model, memory, gamma)
        state = next_state
    return total_reward


def play_multiple_episodes(episodes=10):
    total_rewards = numpy.empty(episodes)
    with tf.Session() as session:
        game = init_game()
        for i in tqdm(range(episodes), desc='Playing episode'):
            epsilon = 1.0 / numpy.sqrt(1 + i)
            total_rewards[i] = play_one_episode(session, game, epsilon)
        plot_running_avg(total_rewards)
    game.close()


if __name__ == '__main__':
    play_multiple_episodes()
