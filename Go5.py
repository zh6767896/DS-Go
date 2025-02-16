# 文件5: train.py
import torch.optim as optim
from torch.utils.data import DataLoader

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

    # 其他辅助方法省略...