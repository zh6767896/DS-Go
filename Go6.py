# 补充StrategicMCTS类中的_calculate_reward方法
def _calculate_reward(self, board, move):
    """计算即时奖励，包含吃子奖励、征子奖励和眼位奖励"""
    if move == 'pass':
        return 0.0
    
    # 模拟落子
    temp_board = board.copy()
    original_captures = temp_board.get_captures()
    temp_board.place_stone(move)
    new_captures = temp_board.get_captures()
    
    # 基础吃子奖励
    capture_reward = (new_captures - original_captures) * 0.2
    
    # 征子奖励
    ladder_reward = self._check_ladder(temp_board, move) * 0.3
    
    # 眼位奖励
    eye_bonus = 0.0
    if self._forms_eye(temp_board, move):
        eye_bonus = 0.15
    
    return capture_reward + ladder_reward + eye_bonus

def _check_ladder(self, board, move):
    """检查是否形成有效征子"""
    y, x = move
    player = board.current_player
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    
    for dy, dx in directions:
        if self._is_ladder_shape(board, (y,x), (dy,dx), player):
            return 1.0
    return 0.0

def _is_ladder_shape(self, board, pos, direction, player):
    """判断征子形状"""
    y, x = pos
    dy, dx = direction
    steps = 3  # 检查三步征子
    
    for i in range(1, steps+1):
        ny = y + dy*i
        nx = x + dx*i
        if not (0 <= ny < 19 and 0 <= nx < 19):
            return False
        if board.board[ny, nx] != -player:
            return False
    return True

def _forms_eye(self, board, move):
    """判断落子是否形成眼位"""
    y, x = move
    player = board.current_player
    eye_type = self._detect_eye_shape(board, (y,x), player)
    
    if eye_type == 'real':
        return 1.0
    elif eye_type == 'false':
        return 0.5
    return 0.0

def _detect_eye_shape(self, board, pos, player):
    """详细眼位检测"""
    y, x = pos
    neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
    diagonals = [(y-1,x-1), (y-1,x+1), (y+1,x-1), (y+1,x+1)]
    
    # 必须全部是己方棋子
    for ny, nx in neighbors:
        if board.board[ny, nx] != player:
            return 'none'
    
    # 计算对角控制
    control_count = 0
    for dy, dx in diagonals:
        if 0 <= dy < 19 and 0 <= dx < 19:
            if board.board[dy, dx] == player:
                control_count += 1
    
    if control_count >= 3:  # 真眼需要至少3个对角控制
        return 'real'
    elif control_count >= 2:
        return 'false'
    return 'none'

# 补充ProfessionalInterface类中的_is_eye方法
def _is_eye(self, pos):
    """可视化用的简化眼位检测"""
    y, x = pos
    player = self.board.board[y, x]
    if player == 0:
        return False
    
    # 检查直接相邻点
    neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
    for ny, nx in neighbors:
        if not (0 <= ny < 19 and 0 <= nx < 19):
            continue
        if self.board.board[ny, nx] != player:
            return False
    
    # 检查对角线控制
    diagonal_control = 0
    diagonals = [(y-1,x-1), (y-1,x+1), (y+1,x-1), (y+1,x+1)]
    for dy, dx in diagonals:
        if 0 <= dy < 19 and 0 <= dx < 19:
            if self.board.board[dy, dx] == player:
                diagonal_control += 1
    
    return diagonal_control >= 3

# 补充TrainingSystem类中的_loss_fn方法
def _loss_fn(self, pred_p, pred_v, target_p, target_v):
    """混合损失函数"""
    # 策略损失（KL散度）
    policy_loss = F.kl_div(
        F.log_softmax(pred_p, dim=1),
        target_p,
        reduction='batchmean',
        log_target=False
    )
    
    # 价值损失（Huber损失）
    value_loss = F.huber_loss(pred_v.squeeze(), target_v, delta=1.0)
    
    # 正则化项
    l2_reg = torch.tensor(0.0)
    for param in self.model.parameters():
        l2_reg += torch.norm(param, p=2)
    
    return 0.8 * policy_loss + 0.2 * value_loss + 1e-4 * l2_reg

# 补充StrategicMCTS类中的_final_value方法
def _final_value(self, board):
    """终局胜负判断"""
    score = board.get_score()
    if score > 0:
        return 1.0 if board.current_player == Player.BLACK else -1.0
    elif score < 0:
        return -1.0 if board.current_player == Player.BLACK else 1.0
    return 0.0

# 补充EnhancedBoard类的get_captures方法
class EnhancedBoard:
    # ... 其他方法保持原样 ...
    
    def get_captures(self):
        """获取当前玩家提子总数"""
        return self.stones_remaining[self.current_player]
    
    def apply_move(self, move):
        """应用落子并更新状态"""
        self.place_stone(move)
        self.zobrist_hash ^= self._get_zobrist_update(move)
        
    def _get_zobrist_update(self, move):
        """更新Zobrist哈希"""
        if move == 'pass':
            return 0
        y, x = move
        player_idx = 0 if self.current_player == Player.BLACK else 1
        return self.zobrist_table[player_idx, y, x]

# 补充TrainingSystem类的_generate_game方法
def _generate_game(self, temperature=1.0):
    """生成自对弈数据"""
    game_data = []
    board = EnhancedBoard()
    mcts = StrategicMCTS(self.model, simulations=1000)
    
    while not board.is_game_over():
        # 运行MCTS搜索
        probs = mcts.search(board)
        
        # 保存训练数据
        features = board.get_features()
        value = self.model.predict(board)[1]
        game_data.append((features, probs, value))
        
        # 选择动作
        action = np.random.choice(list(probs.keys()), p=list(probs.values()))
        board.apply_move(action)
        
        # 更新MCTS根节点
        mcts.update_root(action)
    
    # 处理最终结果
    final_score = board.get_score()
    return [(d[0], d[1], final_score * (-1)**i) for i, d in enumerate(game_data)]

# 补充EnhancedBoard类的完整实现
class EnhancedBoard:
    # 初始化...
    
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
        """使用BFS更新气数"""
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
        
        # 计算整个群的气
        liberties = 0
        for (cy, cx) in group:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == 0:
                    liberties += 1
        
        # 更新气数
        for (cy, cx) in group:
            self.liberties[cy, cx] = liberties
    
    def _remove_dead_stones(self, y, x):
        """移除无气棋子"""
        opponent = -self.current_player
        to_remove = []
        
        # 检查周围对手棋子
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == opponent:
                if self.liberties[ny, nx] == 0:
                    to_remove.extend(self._find_group((ny, nx)))
        
        # 移除棋子并更新计数
        for (ry, rx) in set(to_remove):
            self.board[ry, rx] = 0
            self.liberties[ry, rx] = -1
            self.stones_remaining[opponent] += 1
    
    def _find_group(self, pos):
        """找到连通块"""
        # ... BFS实现同前 ...
    
    def get_score(self):
        """精确数目计算"""
        # ... 完整实现同前 ...