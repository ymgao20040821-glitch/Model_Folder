# Multi-modal CNN-BiLSTM-Attention + Physics-informed Loss

这是一个用于锂离子电池 SOH 预测并可换算 RUL 的 PyTorch 科研基线。模型对 Voltage、Current、Temperature、Capacity 分别编码，再进行特征融合、BiLSTM 时序学习和注意力汇总。训练损失同时约束数据拟合、SOH 单调下降、容量衰减连续性和长期退化趋势。

## 模型数据流

```text
Voltage -------- CNN --┐
Current -------- CNN --┤
Temperature ---- CNN --┼-- Fusion -- BiLSTM -- Attention -- SOH
Capacity -------- LSTM -┘
                                              + Physics loss
```

输入张量为 `(batch, window, 4)`，四列固定为电压、电流、温度和容量。当前 NASA 解析器把一个放电循环内的曲线聚合为均值，因此它是“循环级多模态基线”。若研究重点是曲线形态，可进一步加入循环内采样点维度或 dQ/dV、EIS 分支。

## 目录

```text
Battery_RUL_Prediction/
├── dataset/raw_data/          # 放置 NASA .mat 文件
├── dataset/processed_data/    # 预留缓存目录
├── losses/physics_loss.py
├── models/cnn_branch.py
├── models/bilstm_attention.py
├── models/hybrid_model.py
├── utils/preprocess.py
├── utils/visualization.py
├── train.py
└── test.py
```

## 安装与快速验证

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python train.py --synthetic --epochs 5 --window 30
python test.py --synthetic
```

合成模式只用于检查代码和研究流程是否可运行，不代表真实电池性能。

## NASA 数据

将 NASA PCoE 锂离子电池数据中的多块 `.mat` 文件放入 `dataset/raw_data/`。为了按电池划分训练、验证、测试集，至少需要三块电池。常用文件包括 `B0005.mat`、`B0006.mat`、`B0007.mat` 和 `B0018.mat`。

```powershell
python train.py --data-dir dataset/raw_data --epochs 100 --window 50
python test.py --data-dir dataset/raw_data
```

程序只用训练电池拟合标准化参数，并以整块电池划分数据，降低滑窗重叠导致的数据泄漏。最佳检查点由验证损失选择，默认输出到 `outputs/`。

## 物理约束

总损失为：

```text
L_total = L_MSE + lambda * (L_monotonic + 0.5 * L_continuity + L_trend)
```

- `L_monotonic` 惩罚预测 SOH 随循环次数上升。
- `L_continuity` 约束预测的一阶有限差分接近真实衰减速度。
- `L_trend` 惩罚同一电池在一个批次内的平均正斜率。

`--physics-weight` 控制约束强度。建议通过验证集调参，并报告无物理损失的消融结果。这里的约束属于弱物理先验，不等同于由电化学方程残差构造的严格 PINN；加入温度相关 Arrhenius 项、等效电路参数或 dQ/dV 峰位演化后，才能形成更强的材料机理约束。

## 输出

训练生成 `best_model.pt`、`scaler.json`、`history.json`、`test_metrics.json` 和 `loss_curve.png`。独立测试还生成真实/预测 SOH、RUL 曲线、注意力热力图及 `predictions.npz`。评价指标包括 MAE、RMSE 和 R²。

## 实验注意事项

NASA 不同电池的工况和循环长度有限。正式论文实验应进行 leave-one-battery-out 交叉验证，固定随机种子，报告多次运行均值与标准差，并将普通 MSE、单调约束、连续性约束和新增材料特征逐项消融。注意力权重能显示模型关注的位置，但不能单独证明电化学因果关系。
