## 5 结论

### 5.1 项目核心研究成果总结

本项目以《人工智能导论》课程实践教学为导向，以贪吃蛇游戏为实验载体，系统实现了 DQN、Double DQN、Dueling DQN 和 PPO 四种主流深度强化学习算法，在统一环境下完成了对比训练与结果分析，并配套开发了训练阶段演进可视化回放工具。经过多轮参数调试与优化，主要取得了以下成果：

在算法实现方面，成功搭建了从仿真环境到训练流程的完整框架。贪吃蛇环境从零自主构建，采用 12×12 三通道图像状态表示与组合奖励函数。DQN 系列三种变体共享经验回放与目标网络机制，通过 --algo 参数在标准 DQN、Double DQN 和 Dueling DQN 之间切换；PPO 基于 Actor-Critic 架构独立实现，包含裁剪目标函数、GAE 优势估计与熵正则化三项核心机制。

在训练优化方面，经历多轮迭代解决了训练中的关键瓶颈。修复了探索率双重衰减 Bug，避免了智能体过早丧失探索能力；三轮调整奖励函数使食物净收益显著为正，确保了探索-进食的正向循环；修复 CNN 空间分辨率并将卷积层从 3 层精简至 1 层，兼顾了感知能力与轻量化需求；通过调整学习率、批大小、目标网络更新频率和探索率衰减方式，进一步提升了训练稳定性与收敛速度。

在实验结果方面，Dueling DQN、Double DQN 和标准 DQN 在 10000 回合训练后均能达到 300 以上的平均奖励，其中 Dueling DQN 表现最优，其价值-优势分解结构在动作价值相近的场景中提供了更稳定的估计。PPO 在本场景下未实现有效收敛，分析表明 on-policy 训练模式的样本利用效率较低，且小网格离散动作空间更适配值函数方法。

在可视化工具方面，采用录制-回放分离的设计思路解决了训练速度与可视化需求的矛盾。训练时轻量录制逐帧数据不影响训练效率，训练后通过独立程序实现多算法并排回放、训练曲线渐进延伸、Q 值热力图显示等功能，为观察智能体从随机试探到策略收敛的完整学习过程提供了直观的分析手段。

### 5.2 项目研究局限性剖析

本项目虽然取得了上述成果，但仍然存在以下局限：

第一，PPO 算法未能收敛。尽管完整实现了 PPO 的训练流程，但奖励曲线始终在负值区间波动。可能的原因包括超参数配置未达到最优区间、网络结构设计或训练轮次不足以支撑策略梯度方法的有效收敛等，需要对 PPO 的训练策略进行更加深入的调试。

第二，轻量化网络的表达能力有限。当前模型在达到 300+ 的平均奖励后性能提升趋于停滞，卷积层仅为 1 层的轻量化设计限制了网络对复杂空间模式的学习能力，难以支撑蛇体较长、活动空间狭窄等极限场景下的高阶决策需求。

第三，训练流程的自动化与系统化不足。超参数调试主要依赖人工逐轮尝试，缺乏自动化的超参数搜索机制，调优效率较低。同时实验记录的规范化程度也有待提高。

### 5.3 后续优化与学习展望

针对上述局限性，后续可以从以下几个方面进行改进和深入：

在算法扩展方面，尝试实现更多样化的算法变体，如引入 Prioritized Experience Replay 优化样本利用效率、实现 SAC（Soft Actor-Critic）算法并观察其在离散动作空间下的表现，或整合多个改进机制构建 Rainbow DQN。

在训练自动化方面，引入自动化超参数搜索工具（如 Optuna），对学习率、批大小、探索率衰减参数等进行系统性的网格搜索或贝叶斯优化，提升调优效率并寻找更优的参数组合。

在可视化功能方面，进一步扩展可视化工具的功能，如增加实时训练监控模式（在训练过程中以低频率轮询读取日志和模型检查点，实现准实时进度展示）、支持更多维度的数据呈现（如损失曲线、Q 值分布变化、探索率变化曲线等）。

在工程规范方面，增强实验管理的自动化程度，实现训练中断自动恢复、实验参数自动归档、多轮对比实验的标准化报告生成等功能，建立更加规范和高效的实验管理体系。

## 6 参考文献

[1] Mnih V, Kavukcuoglu K, Silver D, et al. Playing Atari with Deep Reinforcement Learning[J]. arXiv preprint arXiv:1312.5602, 2013.

[2] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.

[3] Van Hasselt H, Guez A, Silver D. Deep Reinforcement Learning with Double Q-Learning[C]. AAAI, 2016.

[4] Wang Z, Schaul T, Hessel M, et al. Dueling Network Architectures for Deep Reinforcement Learning[C]. ICML, 2016.

[5] Schaul T, Quan J, Antonoglou I, et al. Prioritized Experience Replay[J]. arXiv preprint arXiv:1511.05952, 2015.

[6] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.

[7] Schulman J, Moritz P, Levine S, et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation[J]. arXiv preprint arXiv:1506.02438, 2015.

[8] Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements in Deep Reinforcement Learning[C]. AAAI, 2018.

[9] Bellemare M G, Naddaf Y, Veness J, et al. The Arcade Learning Environment: An Evaluation Platform for General Agents[J]. Journal of Artificial Intelligence Research, 2013, 47: 253-279.

[10] 刘全, 翟建伟, 章宗长, 等. 深度强化学习综述[J]. 计算机学报, 2018, 41(1): 1-27.
