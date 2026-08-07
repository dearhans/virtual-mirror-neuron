"""perturbation/graph.py — 因果图与 do-干预（呼应七层框架的「因果层」与 do-演算）。

把「给一个刺激/扰动」定义成图上的 do-干预，使相关→可证伪预测的升级有显式结构。
本模块与 model/twin.py 解耦：图只描述「哪些节点可干预、干预如何改写特征」，
模型消费改写后的特征。
"""
from __future__ import annotations

from typing import Dict, List


class CausalGraph:
    """轻量因果图。

    nodes: 变量名列表
    edges: 有向边 (src, dst) 列表（用于可读性与后续 d-分离分析）
    intervenable: 支持 do() 的节点集合
    """

    def __init__(
        self,
        nodes: List[str],
        edges: List[tuple],
        intervenable: List[str],
    ):
        self.nodes = nodes
        self.edges = edges
        self.intervenable = set(intervenable)

    def do(self, node: str, value, X: "np.ndarray", col_map: Dict[str, int]):
        """对节点 node 施加 do-干预：把 X 中对应列固定为 value，返回新特征矩阵。"""
        if node not in self.intervenable:
            raise ValueError(f"节点 {node} 不在可干预集合 {self.intervenable} 中。")
        if node not in col_map:
            raise ValueError(f"节点 {node} 未在 col_map 中映射列。")
        X_new = X.copy()
        X_new[:, col_map[node]] = value
        return X_new

    def summary(self) -> str:
        lines = ["CausalGraph:", f"  nodes: {self.nodes}", f"  edges: {self.edges}",
                 f"  intervenable: {sorted(self.intervenable)}"]
        return "\n".join(lines)


def build_mirror_neuron_graph(Dp: int = 8) -> CausalGraph:
    """构建虚拟镜像神经元的默认因果图。

    列布局：[p(0..Dp-1), act(Dp..Dp+1), g(Dp+2)]
      - perturbation(p): 可干预（do-扰动强度/模式）
      - action(act):     可干预（do-设定 自我/他者/模仿）
      - neuromodulator(g): 可干预（do-设定调质状态）
      - agent:           混杂因子，不可干预（须边缘化/排除于特征）
      - response(y):     观测结果
    """
    nodes = ["perturbation", "action", "neuromodulator", "agent", "response"]
    edges = [
        ("perturbation", "response"),
        ("action", "response"),
        ("neuromodulator", "response"),
        ("agent", "response"),  # 混杂：仅训练主体有偏置
    ]
    intervenable = ["perturbation", "action", "neuromodulator"]
    return CausalGraph(nodes, edges, intervenable)


def col_map(Dp: int = 8) -> Dict[str, int]:
    """节点名 → X 列索引。"""
    return {
        "perturbation": 0,          # p 占 0..Dp-1（do-干预作用于整段）
        "action": Dp,               # act one-hot 起点
        "neuromodulator": Dp + 2,   # g 单列
    }
