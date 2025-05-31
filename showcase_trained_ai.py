import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path
import time

import gym # 確保您使用的是 gym 而非 gymnasium，如果 env.py 使用的是 gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
# 從 wrappers.py 匯入您自訂的 Wrapper
from wrappers import CutAndScaleObservation, SkipFrame # ResizeObservation 可能不再需要

# --- 設定要載入的已訓練模型路徑 ---
CHECKPOINT_PATH = Path('checkpoints\mario_net_7.chkpt'
'') # <--- 確保這是你正確的模型路徑

if not CHECKPOINT_PATH.exists():
    print(f"錯誤: 找不到指定的模型檔案: {CHECKPOINT_PATH}")
    exit()

# --- 環境初始化 (完全依照 env.py 中的 build_env() 邏輯) ---
print("正在建立環境 (依照訓練時的設定)...")
env = gym_super_mario_bros.make("SuperMarioBros-1-2-v0") # 使用的關卡
env = JoypadSpace(env, [["right"], ["right", "A"]])

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = CutAndScaleObservation(env) # 輸出 (21, 21)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=4) # 輸出 (4, 21, 21)
print("環境建立完成。")
# --- 環境初始化結束 ---


# --- AI Agent 初始化 ---
# state_dim 應匹配 FrameStack 的輸出 (4, 21, 21)
mario = Mario(state_dim=(4, 21, 21), action_dim=env.action_space.n, save_dir=None, checkpoint=CHECKPOINT_PATH)
mario.exploration_rate = 0.0 # 總是利用學到的策略
print(f"已載入模型: {CHECKPOINT_PATH}")
print(f"AI 探索率設定為: {mario.exploration_rate}")

# --- 展示設定 (與之前相同，只顯示成功過關的回合) ---
num_successful_episodes_to_show = 3
max_steps_per_episode = 3000 # 每回合最大步數，以防AI卡在某處不動
max_total_attempts = 100

print(f"將嘗試展示 {num_successful_episodes_to_show} 個 AI 成功過關的回合 (關卡 1-2)...")

successful_episodes_shown = 0
total_episodes_attempted = 0

while successful_episodes_shown < num_successful_episodes_to_show and total_episodes_attempted < max_total_attempts:
    total_episodes_attempted += 1
    
    # env.reset() 返回的可能是 (obs, info) 或只有 obs，取決於 gym 版本和 Wrapper
    reset_output = env.reset()
    if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
        state = reset_output[0]
    else:
        state = reset_output

    print(f"--- 開始第 {total_episodes_attempted} 次嘗試 (目標成功展示: {successful_episodes_shown+1}/{num_successful_episodes_to_show}) ---")
    current_episode_successful = False

    for step in range(max_steps_per_episode):
        env.render()
        action = mario.act(state)
        
        # env.step() 返回的可能是 obs, reward, done, info 或 obs, reward, terminated, truncated, info
        step_output = env.step(action)
        if len(step_output) == 4: # gym API
            next_state_obs, reward, done, info = step_output
        elif len(step_output) == 5: # gymnasium API
            next_state_obs, reward, terminated, truncated, info = step_output
            done = terminated or truncated # 組合 done 狀態
        else:
            raise ValueError(f"env.step() 返回了未預期的元組長度: {len(step_output)}")

        state = next_state_obs
        time.sleep(0.03) # 控制渲染速度

        if info.get('flag_get', False):
            current_episode_successful = True
        
        if done or info.get('flag_get', False):
            print(f"嘗試 {total_episodes_attempted} 結束於第 {step+1} 步。分數: {info.get('score', 'N/A')}, 原因: {'遊戲結束 (done)' if done and not info.get('flag_get', False) else '到達旗杆' if info.get('flag_get', False) else '未知'}")
            break
    
    if step == max_steps_per_episode - 1: # 如果是因為達到最大步數而結束
        print(f"嘗試 {total_episodes_attempted} 因達到最大步數 ({max_steps_per_episode}) 而結束。")

    if current_episode_successful:
        successful_episodes_shown += 1
        print(f"*** 成功展示第 {successful_episodes_shown} 個過關回合! ***")
    else:
        print(f"--- 本次嘗試 ({total_episodes_attempted}) 未成功過關，繼續嘗試。 ---")
    
    # 可選：如果一回合剛結束，可以稍微暫停一下再開始下一回合
    if successful_episodes_shown < num_successful_episodes_to_show: # 如果還沒展示夠，稍微等待
        time.sleep(0.5) 

if successful_episodes_shown < num_successful_episodes_to_show:
    print(f"\n在 {max_total_attempts} 次嘗試後，未能展示足夠 ({num_successful_episodes_to_show}) 的成功回合。僅成功展示 {successful_episodes_shown} 次。")

env.close()
print(f"\n--- 展示結束 --- (共嘗試 {total_episodes_attempted} 回合，成功展示了 {successful_episodes_shown} 個過關回合)")