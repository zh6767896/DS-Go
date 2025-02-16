# 文件1: go_engine.py
import numpy as np
import torch
from enum import IntEnum
from collections import deque
import math

class Player(IntEnum):
    BLACK = 1
    WHITE = -1

class EnhancedBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.ko = None
        self.history = []
        self.liberties = np.full((size, size), -1, dtype=np.int8)
        self.current_player = Player.BLACK
        self.zobrist_hash = 0
        self._init_zobrist()

    def _init_zobrist(self):
        np.random.seed(42)
        self.zobrist_table = np.random.randint(0, 2**63, size=(3, 19, 19), dtype=np.uint64)

    def copy(self):
        new_board = EnhancedBoard(self.size)
        new_board.board = self.board.copy()
        new_board.ko = self.ko
        new_board.history = self.history.copy()
        new_board.liberties = self.liberties.copy()
        new_board.current_player = self.current_player
        new_board.zobrist_hash = self.zobrist_hash
        return new_board

    def get_features(self):
        features = np.zeros((17, 19, 19), dtype=np.float32)
        # 历史状态（8步）
        for i in range(8):
            if i < len(self.history):
                move, player = self.history[-(i+1)]
                if move != 'pass':
                    y, x = move
                    features[i*2 + (0 if player == Player.BLACK else 1), y, x] = 1
        # 当前玩家
        features[16] = 1 if self.current_player == Player.BLACK else 0
        return features

    # 其他方法保持不变（同之前实现）
    # ... [保持原有游戏逻辑代码完整] ...