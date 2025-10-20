# dino_env.py (Final Production Version)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyautogui
import cv2
import webbrowser
import time
import sys
import mss
import pytesseract


class DinoEnv(gym.Env):
    def __init__(self):
        super(DinoEnv, self).__init__()

        self.sct = mss.mss()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.game_region = None
        self.game_over_region = None
        self.score_region = None
        self.current_score = 0

        self._locate_game_window()

    def _locate_game_window(self):
        print("Attempting to locate the game window...")
        webbrowser.open("https://www.crazygames.com/game/chrome-dino")
        play_button_coords = self._find_button('play_button.png', 0.8)
        if play_button_coords:
            pyautogui.click(play_button_coords)
        else:
            sys.exit("Error: Could not find 'play_button.png'.")
        time.sleep(4);
        pyautogui.press('space');
        time.sleep(2)
        replay_button_coords = self._find_button('replay_button.png', 0.8)
        if replay_button_coords:
            print(f"✅ Found 'replay_button.png'.")
        else:
            sys.exit("Error: Could not find 'replay_button.png'.")

        # Your custom coordinates
        self.game_region = {
            'left': int(replay_button_coords.left - 378), 'top': int(replay_button_coords.top - 89),
            'width': int(840), 'height': int(200)
        }
        self.game_over_region = {
            'left': int(replay_button_coords.left), 'top': int(replay_button_coords.top),
            'width': int(replay_button_coords.width), 'height': int(replay_button_coords.height)
        }
        self.score_region = {
            'left': self.game_region['left'] + 600, 'top': self.game_region['top'],
            'width': 235, 'height': 50
        }
        print(f"Game region calculated: {self.game_region}")
        print(f"Score region calculated: {self.score_region}")

    def _find_button(self, image_path, confidence_level):
        start_time = time.time()
        timeout = 30
        while time.time() - start_time < timeout:
            try:
                coords = pyautogui.locateOnScreen(image_path, confidence=confidence_level)
                if coords: return coords
            except pyautogui.ImageNotFoundException:
                pass
            time.sleep(1)
        return None

    def _get_score(self):
        try:
            score_img = np.array(self.sct.grab(self.score_region))
            gray_score = cv2.cvtColor(score_img, cv2.COLOR_BGRA2GRAY)
            text = pytesseract.image_to_string(gray_score, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            score = int("".join(filter(str.isdigit, text)))
            return score
        except (ValueError, pytesseract.TesseractNotFoundError):
            return self.current_score

    def _get_observation(self):
        screenshot = self.sct.grab(self.game_region)
        img = np.array(screenshot)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        edges = cv2.Canny(gray_img, 100, 200)
        resized_img = cv2.resize(edges, (84, 84), interpolation=cv2.INTER_AREA)
        return np.reshape(resized_img, (84, 84, 1))

    def _is_game_over(self):
        try:
            return pyautogui.locateOnScreen('replay_button.png', region=tuple(self.game_over_region.values()),
                                            confidence=0.9) is not None
        except (pyautogui.ImageNotFoundException, OSError):
            return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.click(self.game_over_region['left'] + 10, self.game_over_region['top'] + 10)
        pyautogui.press('space')
        self.current_score = 0
        return self._get_observation(), {}

    def step(self, action):
        # First, check if the game is already over.
        if self._is_game_over():
            # If the game has ended, do not perform any actions.
            # Just return the final state to the training algorithm.
            terminated = True
            reward = -10
            observation = self._get_observation()
            info = {}
            truncated = False
            return observation, reward, terminated, truncated, info

        # --- If the game is NOT over, proceed with the action ---

        # Click to ensure the game has focus.
        pyautogui.click(self.game_region['left'] + 50, self.game_region['top'] + 50)

        reward = 0.01  # Base survival reward

        if action == 1:  # Jump
            pyautogui.press('up')
            reward -= 0.02
        elif action == 2:  # Duck
            pyautogui.keyDown('down');
            time.sleep(0.1);
            pyautogui.keyUp('down')
            reward -= 0.02

        observation = self._get_observation()
        new_score = self._get_score()

        if new_score > self.current_score:
            reward += (new_score - self.current_score)
            self.current_score = new_score

        # Check if our action JUST caused a game over.
        terminated = self._is_game_over()
        if terminated:
            reward = -10  # Override reward with penalty for dying.
            # Click replay immediately for a faster visual loop.
            pyautogui.click(self.game_over_region['left'] + 10, self.game_over_region['top'] + 10)

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        # The render function is now empty to maximize speed.
        pass

    def close(self):
        # No windows to destroy.
        pass