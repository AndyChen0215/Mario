import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path
import time # 引入 time 模組來控制渲染速度

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

# from metrics import MetricLogger # 如果不記錄，可以不用
from agent import Mario # 假設 Mario agent 的 act 方法在未訓練時會隨機行動或基於初始權重行動
from wrappers import ResizeObservation, SkipFrame # ResizeObservation 應為您 wrappers.py 中的 CutAndScaleObservation 或類似功能

# --- 環境初始化 (與 main.py 相同) ---
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# 如果 render_mode='human' 在 make 中無效，則在迴圈中使用 env.render()

env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
# 注意：你的 main.py 使用 ResizeObservation，但你的 wrappers.py 提供的是 CutAndScaleObservation
# 假設 ResizeObservation 的功能與 CutAndScaleObservation 類似，或者你有名為 ResizeObservation 的 Wrapper
env = ResizeObservation(env, shape=84) # 或者使用 CutAndScaleObservation
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
# --- 環境初始化結束 ---

# --- AI Agent 初始化 ---
# 確保 checkpoint 為 None，這樣就會使用一個全新的、未經訓練的模型
# save_dir 在這裡可能不是必要的，除非你的 Mario class 強制要求
# save_dir_showcase = Path('showcase_runs') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
# save_dir_showcase.mkdir(parents=True, exist_ok=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None, checkpoint=None)

# 如果你的 Mario agent 有明確的探索率 (exploration_rate) 並且在 act 方法中使用它來進行隨機選擇：
# 確保探索率設為最大值 (例如 1.0)，使其完全隨機行動。
# 這取決於你的 agent.Mario 類的實現。
# 例如: mario.exploration_rate = 1.0

# --- 展示設定 ---
num_showcase_episodes = 3 # 你想展示幾回合
max_steps_per_episode = 1000 # 每回合最多跑多少步，防止無限迴圈 (可選)

print(f"將展示 {num_showcase_episodes} 回合未訓練的 AI...")

for e in range(num_showcase_episodes):
    state_tuple = env.reset()
    # gym.reset() 可能返回 (state, info) 或只有 state
    if isinstance(state_tuple, tuple):
        state = state_tuple[0]
    else:
        state = state_tuple

    print(f"--- 開始第 {e+1} 回合展示 ---")
    env.render() # 在每回合開始時渲染一次，確保視窗出現

    for step in range(max_steps_per_episode):
        # 1. 顯示環境 (最重要的部分)
        env.render()

        # 2. AI 根據當前狀態決定動作
        # 未訓練的 AI (mario.act) 可能會：
        #   a) 隨機選擇動作 (如果 exploration_rate = 1.0 且 act 方法支持)
        #   b) 基於隨機初始化的神經網路權重輸出一個動作 (看起來也會很隨機)
        action = mario.act(state)

        # 3. AI 執行動作
        # 根據錯誤，env.step(action) 返回 4 個值
        next_state_obs, reward, done, info = env.step(action)
        # 在舊版 gym API 中，'done' 包含了 terminated 和 truncated 的情況。
        # 'truncated' 變數在這裡不會被賦值，如果後續嚴格需要，可以設為 False。

        # 4. 更新狀態
        state = next_state_obs # 第一個返回的值是觀察值

        # 5. 控制渲染速度，否則太快看不清楚
        time.sleep(0.03) # 每幀延遲 0.03 秒，你可以調整這個值

        # 6. 檢查遊戲是否結束
        if done or info.get('flag_get', False):
            print(f"第 {e+1} 回合結束於第 {step+1} 步。原因: {'遊戲結束' if done else '到達旗杆'}")
            break
    
    if step == max_steps_per_episode -1:
        print(f"第 {e+1} 回合達到最大步數。")

env.close() # 結束後關閉環境視窗
print("--- 展示結束 ---")