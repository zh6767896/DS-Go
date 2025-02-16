# go_program.py
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from enum import IntEnum
from collections import defaultdict, deque
from termcolor import colored

# ==================== 基础引擎模块 ====================
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
        self.stones_remaining = {Player.BLACK: 0, Player.WHITE: 0}
        self._init_zobrist()

    def _init_zobrist(self):
        np.random.seed(42)
        self.zobrist_table = np.random.randint(0, 2**63, size=(2, 19, 19), dtype=np.uint64)

    def copy(self):
        new_board = EnhancedBoard(self.size)
        new_board.board = self.board.copy()
        new_board.ko = self.ko
        new_board.history = self.history.copy()
        new_board.liberties = self.liberties.copy()
        new_board.current_player = self.current_player
        new_board.zobrist_hash = self.zobrist_hash
        new_board.stones_remaining = self.stones_remaining.copy()
        return new_board

    def get_features(self):
        features = np.zeros((17, 19, 19), dtype=np.float32)
        for i in range(8):
            if i < len(self.history):
                move, player = self.history[-(i+1)]
                if move != 'pass':
                    y, x = move
                    features[i*2 + (0 if player == Player.BLACK else 1), y, x] = 1
        features[16] = 1 if self.current_player == Player.BLACK else 0
        return features

    def place_stone(self, move):
        if move == 'pass':
            self.history.append(('pass', None))
            self.current_player = -self.current_player
            return
        
        y, x = move
        self.board[y, x] = self.current_player
        self._update_liberties(y, x)
        self._remove_dead_stones(y, x)
        self._update_ko(move)
        self.history.append((move, self.current_player))
        self.current_player = -self.current_player

    def _update_liberties(self, y, x):
        visited = set()
        queue = [(y, x)]
        group = []
        player = self.current_player
        
        while queue:
            cy, cx = queue.pop(0)
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            group.append((cy, cx))
            
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < 19 and 0 <= nx < 19:
                    if self.board[ny, nx] == player and (ny, nx) not in visited:
                        queue.append((ny, nx))
        
        liberties = 0
        for (cy, cx) in group:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == 0:
                    liberties += 1
        
        for (cy, cx) in group:
            self.liberties[cy, cx] = liberties

    def _remove_dead_stones(self, y, x):
        opponent = -self.current_player
        to_remove = []
        
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == opponent:
                if self.liberties[ny, nx] == 0:
                    to_remove.extend(self._find_group((ny, nx)))
        
        for (ry, rx) in set(to_remove):
            self.board[ry, rx] = 0
            self.liberties[ry, rx] = -1
            self.stones_remaining[opponent] += 1

    def _find_group(self, pos):
        y, x = pos
        player = self.board[y, x]
        visited = set()
        queue = [(y, x)]
        group = []
        
        while queue:
            cy, cx = queue.pop(0)
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            group.append((cy, cx))
            
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == player:
                    queue.append((ny, nx))
        
        return group

    def get_captures(self):
        return self.stones_remaining[self.current_player]

    def apply_move(self, move):
        self.place_stone(move)
        self.zobrist_hash ^= self._get_zobrist_update(move)
        
    def _get_zobrist_update(self, move):
        if move == 'pass':
            return 0
        y, x = move
        player_idx = 0 if self.current_player == Player.BLACK else 1
        return self.zobrist_table[player_idx, y, x]

    def is_game_over(self):
        return len(self.history) >= 2 and self.history[-1] == 'pass' and self.history[-2] == 'pass'

    def get_score(self):
        # 实现数目计算逻辑
        return 0  # 简化实现

# ==================== 神经网络模块 ====================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x)

class DynamicPolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels//2, 5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)
        return self.fusion(x).view(x.size(0), -1)

class ProNetwork(nn.Module):
    def __init__(self, blocks=40, channels=512):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(17, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = DynamicPolicyHead(channels)
        self.value_net = nn.Sequential(
            nn.Conv2d(channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_net(x)
        return policy, value

    def predict(self, board):
        with torch.no_grad():
            features = torch.from_numpy(board.get_features()).unsqueeze(0).float()
            policy, value = self.forward(features)
            return F.softmax(policy[0], dim=0).numpy(), value.item()

# ==================== MCTS引擎模块 ====================
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
            board.apply_move(action)
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
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            idx = 361 if move == 'pass' else move[0]*19 + move[1]
            prior = 0.75 * policy[idx] + 0.25 * noise[i]
            node.children[move] = StrategicNode(prior=prior, parent=node)
            node.children[move].reward = self._calculate_reward(board, move)

    def _calculate_reward(self, board, move):
        if move == 'pass':
            return 0.0
        
        temp_board = board.copy()
        original_captures = temp_board.get_captures()
        temp_board.place_stone(move)
        new_captures = temp_board.get_captures()
        
        capture_reward = (new_captures - original_captures) * 0.2
        ladder_reward = self._check_ladder(temp_board, move) * 0.3
        eye_bonus = 0.15 if self._forms_eye(temp_board, move) else 0.0
        
        return capture_reward + ladder_reward + eye_bonus

    def _check_ladder(self, board, move):
        y, x = move
        player = board.current_player
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for dy, dx in directions:
            if self._is_ladder_shape(board, (y,x), (dy,dx), player):
                return 1.0
        return 0.0

    def _is_ladder_shape(self, board, pos, direction, player):
        y, x = pos
        dy, dx = direction
        for i in range(1, 4):
            ny = y + dy*i
            nx = x + dx*i
            if not (0 <= ny < 19 and 0 <= nx < 19):
                return False
            if board.board[ny, nx] != -player:
                return False
        return True

    def _forms_eye(self, board, move):
        y, x = move
        player = board.current_player
        neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
        for ny, nx in neighbors:
            if board.board[ny, nx] != player:
                return False
        diagonals = [(y-1,x-1), (y-1,x+1), (y+1,x-1), (y+1,x+1)]
        control_count = sum(1 for dy, dx in diagonals if board.board[dy, dx] == player)
        return control_count >= 3

    def _final_value(self, board):
        score = board.get_score()
        if score > 0:
            return 1.0 if board.current_player == Player.BLACK else -1.0
        elif score < 0:
            return -1.0 if board.current_player == Player.BLACK else 1.0
        return 0.0

# ==================== 训练系统模块 ====================
class TrainingSystem:
    def __init__(self):
        self.model = ProNetwork()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.buffer = deque(maxlen=1000000)
        
    def self_play(self, num_games=1000):
        for _ in range(num_games):
            game_data = self._generate_game()
            self.buffer.extend(game_data)
            
    def train(self, epochs=100):
        dataset = GoDataset(self.buffer)
        loader = DataLoader(dataset, batch_size=2048, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for states, policies, values in loader:
                self.optimizer.zero_grad()
                pred_p, pred_v = self.model(states)
                loss = self._loss_fn(pred_p, pred_v, policies, values)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    def _loss_fn(self, pred_p, pred_v, target_p, target_v):
        policy_loss = F.kl_div(
            F.log_softmax(pred_p, dim=1),
            target_p,
            reduction='batchmean',
            log_target=False
        )
        value_loss = F.huber_loss(pred_v.squeeze(), target_v, delta=1.0)
        l2_reg = torch.tensor(0.0)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)
        return 0.8 * policy_loss + 0.2 * value_loss + 1e-4 * l2_reg

# ==================== 人机交互界面 ====================
class ProfessionalInterface:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.board = EnhancedBoard()
        self.mcts = StrategicMCTS(self.model)
        
    def start_game(self):
        print("=== 职业围棋AI对战 ===")
        print("输入格式: x,y (例: 4,4) 或 pass")
        self._select_color()
        
        while not self.board.is_game_over():
            self._print_board()
            if self.board.current_player == self.human_color:
                self._human_move()
            else:
                self._ai_move()
                
        self._show_result()
        def _select_color(self):
        while True:
            choice = input("选择执黑(B)或执白(W): ").upper()
            if choice in ('B', 'W'):
                self.human_color = Player.BLACK if choice == 'B' else Player.WHITE
                return
            print("请输入B或W")

    def _human_move(self):
        while True:
            try:
                cmd = input("你的落子: ").strip()
                if cmd.lower() == 'pass':
                    self.board.apply_move('pass')
                    return
                
                x, y = map(int, cmd.split(','))
                if 0 <= x < 19 and 0 <= y < 19:
                    if self.board.is_valid_move((y, x)):
                        self.board.apply_move((y, x))
                        return
                    print("无效落子: 违反围棋规则")
                else:
                    print("坐标超出范围(0-18)")
            except ValueError:
                print("输入格式错误，请使用 x,y 格式")

    def _ai_move(self):
        print("\nAI正在思考...", end='', flush=True)
        probs = self.mcts.search(self.board)
        best_move = max(probs, key=probs.get)
        self.board.apply_move(best_move)
        print(f"\rAI落子: {best_move if best_move != 'pass' else '停着'}")

    def _print_board(self):
        symbols = {
            0: '·',
            Player.BLACK: '●',
            Player.WHITE: '○'
        }
        print("\n   " + " ".join(f"{i:2d}" for i in range(19)))
        for y in range(19):
            line = [f"{y:2d} "]
            for x in range(19):
                stone = self.board.board[y, x]
                color = 'red' if (y, x) == self.board.ko else None
                bg = 'on_cyan' if self._is_eye((y, x)) else None
                line.append(colored(symbols[stone], color, bg))
            print(" ".join(line))

    def _is_eye(self, pos):
        y, x = pos
        if self.board.board[y, x] != 0:
            return False
        
        player = self.board.current_player
        neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
        for ny, nx in neighbors:
            if not (0 <= ny < 19 and 0 <= nx < 19):
                return False
            if self.board.board[ny, nx] != player:
                return False
        
        diagonals = [(y-1,x-1), (y-1,x+1), (y+1,x-1), (y+1,x+1)]
        control = sum(1 for dy, dx in diagonals 
                   if 0 <= dy < 19 and 0 <= dx < 19
                   and self.board.board[dy, dx] == player)
        return control >= 3

    def _show_result(self):
        self._print_board()
        score = self.board.get_score()
        print("\n=== 对局结束 ===")
        print(f"最终得分: {abs(score):.1f}目")
        if score > 0:
            winner = "黑方" if self.human_color == Player.BLACK else "白方(AI)"
        elif score < 0:
            winner = "白方(AI)" if self.human_color == Player.BLACK else "黑方"
        else:
            winner = "平局"
        print(f"胜方: {winner}")

# ==================== 辅助类与入口 ====================
class GoDataset(torch.utils.data.Dataset):
    def __init__(self, buffer):
        self.data = list(buffer)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, policies, values = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(policies).float(),
            torch.tensor(values).float()
        )

if __name__ == "__main__":
    # 训练模式
    if '--train' in sys.argv:
        trainer = TrainingSystem()
        trainer.self_play(num_games=100)
        trainer.train(epochs=50)
        torch.jit.save(torch.jit.script(trainer.model), "go_model.pt")
    # 对战模式
    else:
        if not os.path.exists("go_model.pt"):
            print("请先使用 --train 参数训练模型")
            sys.exit(1)
        interface = ProfessionalInterface("go_model.pt")
        interface.start_game()    