# showcase_new_model.py (每次嘗試都顯示結果)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from pathlib import Path
import time

try:
    from env import build_env
    from agent import Mario
except ImportError:
    print("錯誤: 無法導入 'env' 或 'agent' 模組。")
    # ... (之前的錯誤提示) ...
    exit()

# --- 設定要載入的已訓練模型路徑 ---
CHECKPOINT_PATH = Path(r'checkpoints\mario_net_7.chkpt') # <--- 修改為您的新模型路徑

if not CHECKPOINT_PATH.exists():
    print(f"錯誤: 找不到指定的模型檔案: {CHECKPOINT_PATH}")
    exit()

# --- 環境初始化 ---
print("正在建立環境 (依照新的訓練設定)...")
env = build_env()
print("環境建立完成。")
# --- 環境初始化結束 ---

# --- AI Agent 初始化 ---
mario_agent = Mario(
    state_dim=(4, 21, 21),
    action_dim=env.action_space.n,
    save_dir=None,
    checkpoint=CHECKPOINT_PATH
)
mario_agent.exploration_rate = 0.0
print(f"已載入模型: {CHECKPOINT_PATH}")
print(f"AI 探索率設定為: {mario_agent.exploration_rate}")
print(f"環境動作空間大小: {env.action_space.n}")

# --- 展示設定 ---
num_episodes_to_show = 10             # 您想展示幾回合
max_steps_per_episode = 3500         # 每回合最大步數

print(f"將展示 {num_episodes_to_show} 回合已訓練的 AI (每次嘗試都會在控制台顯示結果)...")

for e in range(num_episodes_to_show): # 直接迴圈固定的回合數
    total_episodes_attempted = e + 1 # 用 e 來計數當前回合
    
    reset_output = env.reset()
    if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
        state = reset_output[0]
    else:
        state = reset_output

    print(f"\n--- 開始第 {total_episodes_attempted}/{num_episodes_to_show} 回合展示 ---")
    
    final_info_of_episode = {}
    done_reason = "未知" # 用於記錄回合結束原因

    for step in range(max_steps_per_episode):
        env.render()
        action = mario_agent.act(state)
        
        step_output = env.step(action)
        if len(step_output) == 4:
            next_state_obs, reward, done, info = step_output
        elif len(step_output) == 5:
            next_state_obs, reward, terminated, truncated, info = step_output
            done = terminated or truncated
        else:
            raise ValueError(f"env.step() 返回了未預期的元組長度: {len(step_output)}")

        state = next_state_obs
        final_info_of_episode = info
        time.sleep(0.03)

        if info.get('flag_get', False):
            done_reason = "🎉 成功抵達旗杆! 🎉"
            break 
        
        if done: # 因其他原因 (如死亡) 結束
            done_reason = "遊戲結束 (例如：死亡)"
            break
    
    # --- 回合結束後的處理 ---
    if step == max_steps_per_episode - 1 and not done and not final_info_of_episode.get('flag_get', False):
        done_reason = f"達到最大步數 ({max_steps_per_episode})"

    print(f"--- 第 {total_episodes_attempted} 回合結束 ---")
    print(f"   原因: {done_reason}")
    print(f"   步數: {step+1}")
    print(f"   分數: {final_info_of_episode.get('score', 'N/A')}")
    
    if total_episodes_attempted < num_episodes_to_show:
        print("   ...準備下一回合...")
        time.sleep(1.0) # 回合間的短暫停頓

env.close()
print(f"\n--- 所有 {num_episodes_to_show} 回合展示結束 ---")