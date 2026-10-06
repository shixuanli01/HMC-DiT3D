# 下一步：重训 HMC 条件 VAE（2026-10-05）

接着 `PROGRESS_2026-10-04.md` 往下走。这份文档是给租卡重训 VAE 用的操作手册，
每一步都有能直接跑的命令。环境、路径相关的注意事项在最后一节。

## 1. 为什么是 VAE

10 月 4 日的 Chair 扫描把三种条件来源放在同一个 DiT、同一套 val 协议下比了 10 个 seed：

| 条件来源 | COV-CD | COV-EMD | 说明 |
|---|---|---|---|
| trainbank（真实 train 条件） | 56.6 | 56.7 | 任何 VAE 的上限 |
| vae（N(0,I) 先验） | 53.9 | 53.6 | 当前方案 |
| gmm（train 后验均值上的 GMM 先验） | 54.1 | 53.8 | 先验空洞已补上，COV 没变 |

GMM 把 |z| 从 8 拉回到 2，和后验均值分布一致，但 COV 一点没涨。
所以损失的 3 个 COV 点不在先验，而在"条件经过 encode→decode 之后变差了"，也就是解码器。

DiT 这边已经排除过了：三个类别都用 bs512 重训到 10000 epoch，每 1000 epoch 在正式协议下评估，
Chair 的 epoch7000/8000 和原版在 seed 噪声之内持平，Airplane/Car 没有任何 epoch 超过原版。
所以本轮 **DiT 固定用 `checkpoints/dit/{cat}.pt`，只动 VAE**。

## 2. 目标和判定标准

- 协议：val 协议（`scripts/diag/val_eval_mmd.py`），N=664，3 次排列取平均，exact CUDA EMD。
- 参考值（TopoDiT-3D，同协议）：

| 类别 | 1-NNA-CD | 1-NNA-EMD | COV-CD | COV-EMD |
|---|---|---|---|---|
| Chair | 48.87 | 46.53 | 56.67 | 57.12 |
| Airplane | 57.15 | 55.83 | 57.05 | 54.11 |
| Car | 57.95 | 59.09 | 55.58 | 51.98 |

- 先做 Chair：基线数据最全（vae 30 个 seed、trainbank 30 个、gmm 20 个），变化最容易看出来。
  Chair 达标再把同样的配方套到 Car（当前 COV-CD 46.7，差 9 个点，是最大短板）。
- 判定分两层：
  1. **解码器修好了**：`trainrecon` 模式（train 条件 encode→decode 后喂 DiT）的 COV 追平 trainbank，
     也就是 Chair 的 COV-CD / COV-EMD 到 56 以上。
  2. **先验也能用**：`gmm` 模式 10 个 seed 的均值追平 trainbank，1-NNA 保持在 48 到 52。

一个必须说清楚的事实：trainbank 本身 10 个 seed 平均只有 56.6 / 56.7，
**COV-EMD 的平均值低于 57.12 的参考值**，只有约 1/3 的 seed 能过线。
所以 VAE 重训的上限就是"追平 trainbank"，超过参考值仍然要靠挑 seed，
或者靠第 6 节的办法把上限本身抬高。

## 3. 第 0 步：先验证假设（半天以内）

用现有的 VAE 跑 `trainrecon`，预期 COV 掉到 54 左右。如果掉到 54，解码器假设成立，继续第 4 步；
如果没掉（还是 56 左右），那损失在先验和解码器的交互上，应该改做第 6 节。

```bash
cd hmc_dit3d
export PYTHONPATH=$EMD_BUILD:src CUDA_VISIBLE_DEVICES=0
O=results/vae_retrain/chair; mkdir -p $O
CFG=results/retrain_epoch_sweep_n664_batch8/runtime_configs/chair_orig_seed0.yaml
BANK=results/cov_diag/banks/chair_train.pt   # 没有就按第 7 节重建

for SEED in 1 2 3; do
  python scripts/diag/diag_gen.py --config $CFG --checkpoint ../checkpoints/dit/chair.pt \
    --vae ../checkpoints/vae/chair.pt --mode trainrecon --bank $BANK --seed $SEED \
    --out $O/orig_trainrecon_seed$SEED.pt
done
python scripts/diag/val_eval_mmd.py --perms 3 --out $O/val_orig_trainrecon.json $O/orig_trainrecon_seed*.pt
```

`trainrecon` 抽的 train 条件和 `trainbank` 同 seed 完全相同（同一个 `seed+1` 的排列），
所以两者的差值就是解码器单独造成的损失。对照组 trainbank seed 1–3 的结果已经在
`PROGRESS_2026-10-04.md` 里（orig 行）。

## 4. 第 1 步：训练 VAE 变体

### 4.1 训练

当前基线：beta 0.05，z 64，hidden 2048/1024/512，2000 epoch。
KL 项是按 latent 维度求和再对 batch 取平均，所以 z 翻倍时 KL 也差不多翻倍，beta 要一起看。

第一轮建议 6 个变体，每个在 H100 上只要几分钟到几十分钟（bank 约 4600 条，bs 256）：

| tag | beta | z | hidden | epochs | 目的 |
|---|---|---|---|---|---|
| b0.02_z64 | 0.02 | 64 | 2048,1024,512 | 2000 | 轻微放松 KL |
| b0.01_z64 | 0.01 | 64 | 2048,1024,512 | 2000 | |
| b0.005_z64 | 0.005 | 64 | 2048,1024,512 | 2000 | 之前试过，但当时只配了 N(0,I) 先验 |
| b0.01_z128 | 0.01 | 128 | 2048,1024,512 | 2000 | 更大 latent |
| b0.005_z128 | 0.005 | 128 | 2048,1024,512 | 3000 | 之前试过，同上 |
| b0.01_z64_big | 0.01 | 64 | 4096,2048,1024 | 3000 | 更宽解码器 |

```bash
cd hmc_dit3d
V=results/vae_retrain/chair/vae; mkdir -p $V
train_vae () {  # tag beta z hidden epochs
  python -m hmc_dit3d.train.train_condition_vae --bank $BANK \
    --output $V/chair_$1.pt --beta $2 --latent-dim $3 --hidden-dims $4 --epochs $5 \
    --device cuda > $V/train_$1.log 2>&1
}
train_vae b0.02_z64     0.02  64  2048,1024,512  2000
train_vae b0.01_z64     0.01  64  2048,1024,512  2000
train_vae b0.005_z64    0.005 64  2048,1024,512  2000
train_vae b0.01_z128    0.01  128 2048,1024,512  2000
train_vae b0.005_z128   0.005 128 2048,1024,512  3000
train_vae b0.01_z64_big 0.01  64  4096,2048,1024 3000
```

训练日志最后会打印 prior 描述子的均值/方差和每个尺度的 perplexity，先看一眼：
`loss_sequences` 应该随 beta 降低而明显下降，否则解码器容量不够，直接跳到 big 变体。

### 4.2 筛选：先 trainrecon，再 gmm

trainrecon 一个 VAE 只要跑 1 到 3 个 seed 就能看出解码器好坏，比直接跑 gmm 10 个 seed 便宜得多。

```bash
for TAG in b0.02_z64 b0.01_z64 b0.005_z64 b0.01_z128 b0.005_z128 b0.01_z64_big; do
  for SEED in 1 2 3; do
    python scripts/diag/diag_gen.py --config $CFG --checkpoint ../checkpoints/dit/chair.pt \
      --vae $V/chair_$TAG.pt --mode trainrecon --bank $BANK --seed $SEED \
      --out $O/${TAG}_trainrecon_seed$SEED.pt
  done
done
python scripts/diag/val_eval_mmd.py --perms 3 --out $O/val_trainrecon.json $O/*_trainrecon_seed*.pt
```

淘汰规则：trainrecon 3 个 seed 平均 COV-CD < 55.5 的变体不再往下跑。

通过的变体跑 gmm 先验，先 5 个 seed，和 vae（N(0,I)）对照：

```bash
for TAG in <通过的 tag>; do
  for SEED in 1 2 3 4 5; do
    for MODE in gmm vae; do
      python scripts/diag/diag_gen.py --config $CFG --checkpoint ../checkpoints/dit/chair.pt \
        --vae $V/chair_$TAG.pt --mode $MODE --bank $BANK --seed $SEED --gmm-k 32 \
        --out $O/${TAG}_${MODE}_seed$SEED.pt
    done
  done
done
python scripts/diag/val_eval_mmd.py --perms 3 --out $O/val_prior.json $O/*_gmm_seed*.pt $O/*_vae_seed*.pt
```

多卡的话直接套 `scripts/diag/chair_seed_sweep.sh` 的写法，一张卡两个任务。

### 4.3 确认

5 个 seed 里 gmm 平均 COV 到 56 以上的变体，补到 10 个 seed（seed 6–10），
和 PROGRESS 里 trainbank 的 10 seed 表并排比较。1-NNA 必须还在 48 到 52 之间，
COV 涨但 1-NNA 偏离 50 说明是在过拟合 reference。

### 4.4 要记录的东西

每个变体一行，写进本文档末尾或新的 PROGRESS 文件：
VAE 的 beta / z / hidden / epochs / seed、bank 路径和条数、VAE 文件的 SHA-256、
DiT 检查点 SHA-256（`CHECKPOINTS.md` 里有）、模式（trainrecon / gmm / vae）、
gmm 的 K 和温度、threshold 0.1、seed 列表、val 协议的 N 和排列数。
不要覆盖已有的 json 和 pt，`val_eval_mmd.py` 遇到同名 payload 会直接跳过。

## 5. 第 2 步：Car

Chair 选定配方后，Car 用相同的 beta / z / hidden 重训一次，走同样的 trainrecon → gmm 流程。
Car 的 trainbank 基线（seed 0）是 58.38 / 50.47 / 51.33 / 56.82，COV-CD 距参考 55.58 还有 4 个点，
解码器修好也未必够，要有心理准备。Airplane 的 vae 模式已经两项 COV 都超过参考，暂时不动。

## 6. 如果 VAE 重训不够：抬高上限

trainbank 是上限这件事意味着，光靠还原真实条件最多到 56.6 / 56.7。
要稳定超过参考值，得让条件比真实 train 条件更"散"但又不离开流形。三个候选，按实现难度排：

1. **条件空间插值**：trainbank 里随机取两条 kNN 邻居，在后验均值 μ 上线性插值后解码，
   或直接在原始条件空间插值。改 `diag_gen.py` 加一个模式就能试。
2. **guidance 和条件 dropout**：`--guidance` 小于 1（比如 0.7、0.85）会削弱条件、增加多样性，
   代价是 1-NNA 可能变差。这个不用训练，可以和第 3 步并行跑。
3. **条件空间的小 flow / 扩散模型**：代替 VAE 直接在 4672 维条件空间学一个生成模型，
   彻底绕开解码器。工作量最大，前两个都不行再做。

## 7. 新机器上的准备

- 数据：`ShapeNetCore.v2.PC15k` 放到仓库根目录，或者改 `val_eval_mmd.py` 里的 `ROOT`。
- exact EMD：需要 TopoDiT 的 CUDA approxmatch 扩展（PyTorchEMD）。编译后把目录赋给 `$EMD_BUILD`，
  并把 `val_eval_mmd.py` 第 9 行写死的 `/fsx/weicyang/shix/emd_build` 改掉。
  没有这个扩展就只能用内置 Sinkhorn EMD，数字和参考值不可比。
- 写死的集群路径：`scripts/diag/val_eval_mmd.py`、`chair_seed_sweep.sh`、`rescore_missing.sh`
  里的 `/fsx/weicyang/shix/...` 都要改。
- 重建 train bank（只用 train split，这是 AGENTS.md 的第一条规则）：

```bash
cd hmc_dit3d
python -m hmc_dit3d.train.build_bank \
  --config configs/train_chair_h100_formal_bottleneck.yaml --split train \
  --output results/cov_diag/banks/chair_train.pt
```

- `runtime_configs/chair_orig_seed0.yaml` 是 `sweep_retrain_epochs.py` 生成的，不在 git 里。
  没有的话用 `configs/train_chair_h100_formal_bottleneck.yaml` 代替，
  `diag_gen.py` 只从里面读数据路径、HMC 设置和 `train.seed`。
- 先跑 `python -m pytest`，再跑第 3 节的 orig trainrecon 做冒烟，确认数字和 PROGRESS 对得上再开扫描。

## 8. 代码改动

- `scripts/diag/diag_gen.py` 新增 `--mode trainrecon`：抽样方式和 `trainbank` 完全一致，
  但把条件过一遍 VAE 的 encode（取 μ）→ `decode_conditions`，threshold 用 `--threshold`。
  用来单独量化解码器的损失。
