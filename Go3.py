# 文件3: mcts_engine.py
import numpy as np
from collections import defaultdict

class StrategicNode:
    def __init__(self, prior=1.0, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.state = None
        self.reward = 0.0

class StrategicMCTS:
    def __init__(self, model, simulations=2000):
        self.model = model
        self.simulations = simulations
        self.root = StrategicNode()
        self.transposition_table = defaultdict(StrategicNode)
        
        # 动态参数
        self.c_puct_base = 5.0
        self.dirichlet_alpha = 0.3
        self.temperature = 1.0

    def search(self, board):
        for _ in range(self.simulations):
            self._simulate(board.copy())
        return self._get_action_probs()

    def _simulate(self, board):
        path = []
        node = self.root
        while node.children:
            action, node = self._select_child(node)
            board.place_stone(action)
            path.append(node)
        
        if not board.is_game_over():
            policy, value = self.model.predict(board)
            self._expand(node, board, policy)
        else:
            value = self._final_value(board)
        
        self._backpropagate(path, value)

    def _select_child(self, node):
        total_n = math.log(sum(c.visit_count for c in node.children.values()) + 1e-8)
        best_score = -np.inf
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            exploit = child.total_value / (child.visit_count + 1e-8)
            explore = self.c_puct_base * child.prior * math.sqrt(total_n) / (1 + child.visit_count)
            score = exploit + explore + 0.2 * math.sqrt(child.reward + 1e-8)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def _expand(self, node, board, policy):
        legal_moves = board.get_legal_moves()
        policy = self._mask_policy(policy, legal_moves)
        
        # 添加Dirichlet噪声
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            idx = 361 if move == 'pass' else move[0]*19 + move[1]
            prior = 0.75 * policy[idx] + 0.25 * noise[i]
            node.children[move] = StrategicNode(prior=prior, parent=node)
            node.children[move].reward = self._calculate_reward(board, move)

    def _calculate_reward(self, board, move):
        # 计算即时奖励（吃子奖励等）
        # ... [具体实现需结合游戏规则] ...
        return 0.0

    # 其他辅助方法保持不变
    # ... [保持原有MCTS逻辑完整] ...