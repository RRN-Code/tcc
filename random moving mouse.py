# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:29:50 2024

@author: ramos
"""

import pyautogui
import time
import keyboard
import numpy as np

# print(pyautogui.size())

# pyautogui.moveTo(100, 100, duration=1)

# w = np.random.randint(0, 1919)
# h = np.random.randint(0, 1079)

# pyautogui.moveTo(w, h, duration= 1)


while True:
    w = np.random.randint(0, 1919)
    h = np.random.randint(0, 1079)
    pyautogui.moveTo(w, h, duration= 1)

    if keyboard.is_pressed('esc'):
        break
    
    time.sleep(2)      
