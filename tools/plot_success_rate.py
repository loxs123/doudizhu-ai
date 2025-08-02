import re
import matplotlib.pyplot as plt

# Step 1: 读取日志文件
with open('logs/train.log', 'r') as f:
    lines = f.readlines()

# Step 2: 提取成功率
success_rates = []
epochs = []
pattern = r"Agent SUCCESS Rate ([0-9]*\.[0-9]+)"
epoch = 0

for line in lines:
    match = re.search(pattern, line)
    if match:
        rate = float(match.group(1))
        success_rates.append(rate)
        epochs.append(epoch)
        epoch += 1

# Step 3: 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, success_rates, marker='o')
plt.xlabel("Training Epoch")
plt.ylabel("Success Rate")
plt.title("Success Rate over Training Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/success_rate.png", dpi=300)
plt.show()
