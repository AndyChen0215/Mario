import gym
import numpy as np

from gym.spaces import Box
from skimage import transform


class CutAndScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (21, 21)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation[120:, 128:], (21, 21))
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # 假設輸入的 observation 已經是灰階 (H, W)
        # 如果原始環境的 observation_space 不是 Box，或者 low/high 不同，需要調整
        # 但通常在 GrayScaleObservation 之後，會是 Box(low=0, high=255, shape=(H,W), dtype=uint8)
        # 這個 Wrapper 的輸出觀察空間
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        # transform.resize 預設會將輸出值縮放到 [0, 1] 浮點數 (如果 preserve_range=False)
        # 但因為後續有 x / 255. 的操作，這裡 ResizeObservation 應該輸出 uint8 [0, 255]
        # 因此使用 preserve_range=True，並確保輸入是 uint8
        # 或者，如果輸入是 [0,1] float，則 resize 後乘以 255
        
        # 假設輸入的 observation 是 uint8 類型且範圍是 [0, 255] (例如來自 GrayScaleObservation)
        resized_obs = transform.resize(
            observation,
            self.shape,
            preserve_range=True, # 保持原始值的範圍
            anti_aliasing=True   # 建議開啟以獲得更好的縮放品質
        )
        return resized_obs.astype(np.uint8)
    
class CustomRewardMario(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 追蹤狀態變化的變數
        self.last_x_pos = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.last_time = 400 # 初始時間通常是400
        self.last_life = 2   # 滿血時 life 通常是 2 (代表2個命的條)
        self.last_score = 0

        # 獎勵/懲罰的權重 (這些值需要您仔細調整！)
        self.x_pos_reward_factor = 0.05      # 向右移動的獎勵因子
        self.x_pos_penalty_factor = 0.08     # 向左移動的懲罰因子 (避免卡住來回晃)
        self.time_change_penalty_factor = -0.002 # 時間每減少1的懲罰 (鼓勵效率)
        self.death_penalty = -20.0           # 死亡的巨大懲罰 (當 life 從 >0 到 0)
        self.win_reward = 50.0               # 到達旗杆的巨大獎勵
        self.coin_reward_per_coin = 0.25     # 每收集一個金幣的獎勵
        self.powerup_reward = 6.0            # 獲得道具 (蘑菇、花) 的獎勵
        self.lose_powerup_penalty = -4.0     # 失去道具的懲罰 (受傷但未死)
        
        # 針對卡住的懲罰
        self.stuck_penalty = -0.05           # 如果卡住不動的懲罰
        self.stuck_frames_threshold = 25     # 連續多少幀 X 沒變算卡住 (考慮 skipframe 影響)
        self.frames_stuck = 0
        self.info_initialized = False        # 標記是否已從 info 初始化過一次狀態

    def _initialize_states_from_info(self, info):
        """Helper function to initialize/update states from info dict."""
        self.last_x_pos = info.get('x_pos', self.last_x_pos)
        self.last_coins = info.get('coins', self.last_coins)
        self.last_status = info.get('status', self.last_status)
        self.last_time = info.get('time', self.last_time)
        # 'life' in gym-super-mario-bros: 2 means full health (big Mario), 
        # 1 means small Mario after taking a hit (if was big) or just small Mario.
        # 0 means Mario is in the death animation or dead.
        self.last_life = info.get('life', self.last_life)
        self.last_score = info.get('score', self.last_score)
        self.frames_stuck = 0
        self.info_initialized = True

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)
        self.info_initialized = False # 重置時標記為未初始化
        # 在第一次 step 時從 info 初始化狀態，因為 reset 可能不返回完整的 info
        if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
            obs, info = reset_output
            self._initialize_states_from_info(info) # 如果 reset 返回 info，則嘗試初始化
            return obs, info
        return reset_output


    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)

        if not self.info_initialized: # 如果在 reset 時沒有 info，則在第一次 step 後初始化
            self._initialize_states_from_info(info)

        # 獲取當前狀態值
        current_x_pos = info.get('x_pos', self.last_x_pos)
        current_coins = info.get('coins', self.last_coins)
        current_status = info.get('status', self.last_status)
        current_time = info.get('time', self.last_time)
        current_life = info.get('life', self.last_life) # 0, 1, or 2
        current_score = info.get('score', self.last_score)
        flag_get = info.get('flag_get', False)

        custom_reward = 0.0

        # 1. X 軸進度獎勵/懲罰
        x_diff = current_x_pos - self.last_x_pos
        if x_diff > 0:
            custom_reward += x_diff * self.x_pos_reward_factor
            self.frames_stuck = 0
        elif x_diff < -1: # 給予一點容錯空間，避免因微小調整而懲罰
            custom_reward -= abs(x_diff) * self.x_pos_penalty_factor
            self.frames_stuck = 0
        else:
            self.frames_stuck += 1

        # 2. 時間懲罰
        if current_time < self.last_time:
            custom_reward += (current_time - self.last_time) * abs(self.time_change_penalty_factor) # time_change_penalty_factor 是負的
        
        # 3. 死亡懲罰
        # 'life' == 0 表示瑪利歐正在死亡或已死亡
        if current_life == 0 and self.last_life > 0: # 從活著到死亡
            custom_reward += self.death_penalty
            # print("Death penalty applied.") # Debug

        # 4. 過關獎勵
        if flag_get:
            custom_reward += self.win_reward
            # print("Win reward applied.") # Debug
            # 過關後通常 done 也會是 True

        # 5. 金幣獎勵
        if current_coins > self.last_coins:
            custom_reward += (current_coins - self.last_coins) * self.coin_reward_per_coin

        # 6. 道具狀態獎勵/懲罰 (基於 'status' 和 'life' 的變化)
        # 獲得道具: small -> tall/fireball OR tall -> fireball
        if ((self.last_status == 'small' and (current_status == 'tall' or current_status == 'fireball')) or \
            (self.last_status == 'tall' and current_status == 'fireball')):
            custom_reward += self.powerup_reward
            # print("Powerup reward.") # Debug
        # 失去道具 (受傷但未死): (tall/fireball -> small) AND life 減少但 > 0
        elif ((self.last_status == 'tall' or self.last_status == 'fireball') and current_status == 'small') and \
             (current_life > 0 and current_life < self.last_life):
            custom_reward += self.lose_powerup_penalty
            # print("Lose powerup penalty.") # Debug
        
        # 7. 卡住懲罰
        if self.frames_stuck >= self.stuck_frames_threshold:
            custom_reward += self.stuck_penalty
            # print(f"Stuck penalty at x={current_x_pos}") # Debug
            self.frames_stuck = 0 # 重置計數器，避免連續懲罰

        # 更新 "上一步" 的狀態
        self.last_x_pos = current_x_pos
        self.last_coins = current_coins
        self.last_status = current_status
        self.last_time = current_time
        self.last_life = current_life
        self.last_score = current_score

        return obs, custom_reward, done, info