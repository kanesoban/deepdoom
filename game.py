#!/usr/bin/env python


from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep
import os


def init_game():
    game = DoomGame()
    game.set_doom_scenario_path(os.sep.join(['resources', 'scenarios', 'basic.wad']))
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
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
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
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


def main():
    game = init_game()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10

    sleep_time = 1.0 / DEFAULT_TICRATE

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels

            r = game.make_action(choice(actions))

            print("State #" + str(n))
            print("Game variables:", vars)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

        game.close()


if __name__ == '__main__':
    main()