# 文件4: human_interface.py
import sys
from termcolor import colored

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
        choice = input("选择执黑(B)或执白(W): ").upper()
        self.human_color = Player.BLACK if choice == 'B' else Player.WHITE

    def _human_move(self):
        while True:
            try:
                cmd = input("你的落子: ").strip()
                if cmd == 'pass':
                    self.board.place_stone('pass')
                    return
                
                x, y = map(int, cmd.split(','))
                if self.board.is_valid_move((y, x)):
                    self.board.place_stone((y, x))
                    return
                print("无效落子!")
            except:
                print("输入格式错误")

    def _ai_move(self):
        print("\nAI正在思考...")
        probs = self.mcts.search(self.board)
        best_move = max(probs.items(), key=lambda x: x[1])[0]
        self.board.place_stone(best_move)
        print(f"AI落子: {best_move}")

    def _print_board(self):
        # 高级棋盘显示
        symbols = {0: '·', 1: '●', -1: '○'}
        print("\n   "+" ".join(f"{i:2d}" for i in range(19)))
        for y in range(19):
            line = [f"{y:2d} "]
            for x in range(19):
                stone = self.board.board[y, x]
                color = 'red' if (y, x) == self.board.ko else None
                bg = 'on_cyan' if self._is_eye((y, x)) else None
                line.append(colored(symbols[stone], color, bg))
            print(" ".join(line))

    def _is_eye(self, pos):
        # 判断是否为眼
        # ... [实现眼位检测逻辑] ...
        return False

if __name__ == "__main__":
    interface = ProfessionalInterface("go_model.pt")
    interface.start_game()