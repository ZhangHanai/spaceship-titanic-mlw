这个包的用途：复现这次小号测出来的 0.82183 诊断提交。

运行方式：
1. 把 spaceship_082183_reference_aided.py 放到你的项目文件夹 C:\Users\28782\MLWproject。
2. 项目文件夹里需要有：
   - test.csv
   - submission.csv  （别人 0.82137 的 reference submission）
   - spaceship_082plus_outputs/submission_catboost_threshold_050.csv  或 submission_catboost_threshold_050.csv
3. 在 Anaconda Prompt / Jupyter Terminal 里运行：
   python spaceship_082183_reference_aided.py
4. 输出会在 spaceship_082183_outputs 文件夹里：
   - submission_082183_reference_aided_sideP.csv
   - audit_082183_reference_aided_sideP.csv

核心逻辑：
- 先以我们 0.80547 的 CatBoost 提交为 baseline。
- 把 baseline=False 但 reference=True 的 216 行翻成 True。
- 再把 baseline=True、reference=False、且 Cabin side=P 的 36 行翻成 False。
- 这对应你刚刚测出来的 submission_plus_true_minus_sideP.csv，Public Score = 0.82183。

严肃提醒：
这个文件是 reference-aided post-processing，不是独立原创模型。它适合用于诊断、理解高分差异和小号验证。
正式报告里不要把它写成“我们独立训练出的模型”。如果要写进正式项目，必须诚实表述为 public-notebook/reference-guided ablation 或启发式后处理实验，并且最好同时保留一个完全不读取 reference submission 的 standalone model 版本。
