# 强化学习斗地主

## 动机

探究Transformer序列建模方法在斗地主上的性能

## 方法总览

### 出牌表示向量

![image-20250713020528965](images/vector.jpg)

### 斗地主出牌模型（douformer）

![image-20250713021030769](images/model.jpg)

## 测试结果

### 运行命令
```bash
python src/train.py
```

| 超参数        | 值         |
|---------------|------------|
| 学习率        | 1e-4       |
| 批量大小      | 512        |
| buffer_size  | 2048 * 3   |
| 随机采样概率 | 0.03 |
| ppo_step | 4 |

### 训练结果

1. 使用两个随机出牌的agent作为douformer的对手，douformer（地主）赢牌率随训练轮次的变化如下：
![image-20250713021030769](images/success_rate.png)

日志文件为：`logs/fix-bug-token-value.log`

[^1]: 先前代码的代码实现有误，上面是修正bug(去掉开局过牌的两次补位)并加入ε探索

### 对比模型
![image-20250713021030769](images/baselines.jpg)

### 下一步计划

借鉴大语言模型训练的思想，先让他学习baseline中的agent，然后再用强化学习激发性能。