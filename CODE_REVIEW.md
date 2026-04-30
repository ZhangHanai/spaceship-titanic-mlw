# 代码评估（Spaceship Titanic SVM Notebook）

## 总体评价

你的 Notebook 结构清晰、步骤完整，已经覆盖了一个较规范的机器学习建模流程：
数据读取 → 预处理 → 划分验证集 → baseline → 超参搜索 → 最终提交文件生成。整体可读性和工程化意识都不错。

**综合评分：8/10（学习/比赛场景表现优秀）**

---

## 亮点

1. **流程完整且分阶段明确**
   - 先训练 baseline，再做 Optuna 调参，再训练 final model，这个节奏非常好。

2. **预处理思路合理**
   - 对消费类长尾特征使用 `log1p`，并对 SVM 输入进行 `StandardScaler`，这是符合算法特性的。

3. **避免数据泄漏（部分做得不错）**
   - 类别缺失值、年龄缺失值都使用训练集统计量填充后再应用到测试集，方向正确。

4. **验证策略意识不错**
   - 有 `train_test_split(..., stratify=y)` 与 `StratifiedKFold`，对分类任务是合理选择。

5. **产出导向明确**
   - 直接生成 Kaggle submission 文件，实用性强。

---

## 主要问题与风险

1. **存在“验证集重复使用”风险（轻度过拟合验证集）**
   - 你先在 `X_train` 上通过 CV 调参，再在固定 `X_valid` 上反复评估“最佳模型”。
   - 如果后续根据 `X_valid` 结果继续人工迭代，会逐步对验证集过拟合。
   - 建议：
     - 使用嵌套交叉验证（更严谨，代价更高），或
     - 保留一个完全不参与调参的最终 holdout。

2. **特征工程还有明显提升空间**
   - 当前丢弃了 `Cabin`、`Name`，但这两个字段在该比赛通常可提取有效信息：
     - `Cabin` 可拆分 deck/side/num；
     - `Name` 可提取 family group（同姓/同 group）。
   - `PassengerId` 也可拆分出 group size，这在 Spaceship Titanic 中通常有效。

3. **搜索空间相对保守**
   - 仅搜索 `C` 与 `gamma`，且 trial 数仅 20。对 SVM 来说通常偏少。
   - 建议扩大试验数（如 80~200），并考虑加入：
     - `class_weight`（处理潜在类别不平衡），
     - `kernel` 备选（线性核可能更快），
     - `probability=False/True`（若后续要阈值调优）。

4. **Notebook 中运行时安装依赖不利于复现**
   - `%pip install ipywidgets tqdm` 以及 cell 内 `try/except` 安装 optuna，虽然方便，但在 CI 或离线环境不稳定。
   - 建议将依赖写入 `requirements.txt`，Notebook 只负责 import。

5. **指标体系略单一**
   - 只看 `accuracy` 对该题通常够用，但仍建议记录：
     - CV mean ± std，
     - 多次随机种子下稳定性，
     - 若有需要可加 balanced accuracy / F1。

---

## 可执行的下一步优化（按优先级）

1. **先补强特征**：Cabin 拆分、PassengerId group 特征、family/group 统计特征。
2. **把预处理和模型参数集中配置**：减少硬编码，提升复现实验效率。
3. **增大 Optuna trial 数并启用 pruning**：同等时间下通常能得到更稳结果。
4. **引入结果记录表**：保存每次实验配置 + 分数，避免“感觉调参”。
5. **做简单模型对比**：SVM vs XGBoost/LightGBM/RandomForest（同一 CV 协议下比较）。

---

## 一句话结论

这份代码已经达到“可提交且有一定竞争力”的水平；如果你补齐高价值特征并提升验证/调参严谨度，分数通常还能再上一个台阶。
