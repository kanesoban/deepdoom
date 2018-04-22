#!/usr/bin/env python


from __future__ import print_function

import os
import tensorflow as tf
from model import DoomNeuralNetwork

import numpy
from tqdm import tqdm
from vizdoom import *
from visualization import plot_running_avg


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


def to_one_hot(action):
    max_position = int(numpy.argmax(action))
    one_hot_action = numpy.zeros((len(action),))
    one_hot_action[max_position] = 1.0
    return one_hot_action.reshape((1, -1))


def update(model, reward, state, action, gamma):
    next_action_predictions = model.predict(state)
    state_action_values = numpy.zeros((len(action),))
    max_position = int(numpy.argmax(action))
    state_action_values[max_position] = reward + gamma * numpy.max(next_action_predictions)
    model.update(state, state_action_values.reshape((1, -1)), to_one_hot(action))


def play_one_episode(session, game, epsilon, gamma=0.99, max_steps=10000):
    total_reward = 0
    dims = (None, 120, 160, 1)
    model = DoomNeuralNetwork(session, dims, game.get_available_buttons_size())
    session.run(tf.global_variables_initializer())
    game.new_episode()
    time_step = 0
    state = convert_image(game.get_state().screen_buffer)
    while not game.is_episode_finished() and max_steps > time_step:
        action = model.sample_action(state, epsilon)
        reward = game.make_action(action)
        total_reward += reward
        if game.get_state() is None:
            break
        state = convert_image(game.get_state().screen_buffer)
        update(model, reward, state, action, gamma)
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
