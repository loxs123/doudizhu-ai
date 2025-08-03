# 强化学习斗地主

## 方法总览

### 牌型表示(v1)

![image-20250713020528965](images/image-v1.png)

### 出牌身份表示(v2)

![image-20250713020604702](images/image-v2.png)

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
| buffer_size  | 2048       |

### 选择最优动作执行+同时采样地主出牌和农民出牌训练结果

使用两个随机出牌的agent作为douformer的对手，douformer（地主）赢牌率随训练轮次的变化如下：

![image-20250713021030769](images/success_rate_all_data.png)

日志文件为：`logs/all-data.log`

如果地主也随机出牌，赢牌率约为0.34

### 选择最优动作执行+只采样地主出牌训练结果

同样使用两个随机出牌的agent作为douformer的对手

![image-20250713021030769](images/success_rate_only_dizhu.png)

日志文件为：`logs/only-dizhu-data.log`