# env.py
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT # <--- 匯入預定義的動作列表
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

# 從您自訂的 wrappers.py 匯入
from wrappers import CutAndScaleObservation, SkipFrame, CustomRewardMario

def build_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-2-v0") # 或者您想用的關卡
    env = JoypadSpace(env, SIMPLE_MOVEMENT) # <--- 使用 SIMPLE_MOVEMENT

    # --- Wrapper 順序 ---
    env = CustomRewardMario(env) # 在 SkipFrame 之前套用自訂獎勵
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = CutAndScaleObservation(env)
    env = TransformObservation(env, f=lambda x: x / 255.0)
    env = FrameStack(env, num_stack=4)

    return env