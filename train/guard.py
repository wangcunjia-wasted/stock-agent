"""
guard.py —— 项目统一错误处理守卫

为什么需要它（都是本项目踩过的坑）：
  1. OpenHowNet 的 HowNetDict.get() 静默失败，义原覆盖 0% 却不报任何错
  2. panel 构建 668 只股票全部失败，只留一屏 warning，最后才在 concat 处
     炸出一句没人看得懂的 "No objects to concatenate"
  3. BaoStock 请求无超时，无限挂起

守卫的三层设计：
  A. 异常层级    QuantError 根异常 + 按环节分类的子类。顶层 except 只拦
                 "自己认识的自定义错误"，意外异常（KeyError/断言失败等）
                 原样穿透——守卫绝不吞掉没见过的错。
  B. 失败账本    FailureTracker：批量任务逐条记录成败与原因，结束统一清算。
                 失败率超阈值立刻抛 ThresholdError，并附上原因分布——
                 把"668 只全灭只剩 warning"升级为开局即炸、原因一目了然。
  C. 步骤装饰器  @step：给流水线每个阶段计时、写日志；失败时 traceback
                 落到 data/logs/，控制台只留一句人话，事后可以翻旧账。

使用范式（每个可独立运行的脚本）：

    from guard import QuantError

    def main():
        ...  # 业务逻辑，内部该炸就炸

    if __name__ == "__main__":
        try:
            main()
        except QuantError as e:        # 只拦认识的自定义错误
            print(f"\\n✗ 任务失败：{e}")
            sys.exit(1)
"""
import functools
import logging
import os
import time
import traceback
from collections import Counter
from datetime import datetime

# ============================================================
# A. 异常层级
# ============================================================

class QuantError(Exception):
    """项目内所有自定义异常的根。
    存在的意义：顶层 except QuantError 只拦守卫主动抛的错，
    语法错误、KeyError 之类的意外仍原样抛出，不会被守卫吞掉。"""


class DataError(QuantError):
    """数据获取/文件缺失/格式问题（fetch_*.py、数据加载环节）"""


class FactorError(QuantError):
    """因子计算环节的问题"""


class LabelError(QuantError):
    """标签构造/对齐闸失败"""


class PanelError(QuantError):
    """横截面大表构造问题（成分过滤、合并等）"""


class ThresholdError(QuantError):
    """批量任务失败率超阈值（FailureTracker 清算时抛出）"""


# ============================================================
# 日志：控制台照常 print（给人实时看），日志文件记细节（给事后查）
# ============================================================

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs")
_logger = None


def get_logger(name: str = "quant") -> logging.Logger:
    """全局唯一 logger，写 data/logs/<启动时间>.log。重复调用不重复挂 handler。"""
    global _logger
    if _logger is not None:
        return _logger
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{datetime.now():%Y%m%d_%H%M%S}.log")
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(fh)
    lg.info(f"=== 日志启动：{log_file} ===")
    _logger = lg
    return lg


def step(desc: str):
    """流水线步骤装饰器：计时、写日志、失败落 traceback 后原样抛出（不吞错）。

    用法：
        @step("逐股因子计算")
        def compute_all(files): ...
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            lg = get_logger()
            t0 = time.time()
            lg.info(f"▶ {desc} ...")
            try:
                result = fn(*args, **kwargs)
            except QuantError as e:
                lg.error(f"✗ {desc} 失败：{e}\n{traceback.format_exc()}")
                raise
            except Exception as e:                      # 意外异常：留痕后原样抛
                lg.error(f"✗ {desc} 意外异常 {type(e).__name__}：{e}\n"
                         f"{traceback.format_exc()}")
                raise
            lg.info(f"✓ {desc} 完成，耗时 {time.time()-t0:.1f}s")
            return result
        return wrapper
    return deco


# ============================================================
# B. 失败账本：批量任务的统一清算
# ============================================================

class FailureTracker:
    """批量任务（逐股/逐文件）的失败账本。

    防的就是"全灭只剩 warning"：逐条记录谁失败、为什么，任务结束时
    check() 统一清算——失败率超阈值或成功数不足即抛 ThresholdError，
    并附上原因分布，一眼看出是不是所有失败都是同一个错。
    """

    def __init__(self, task: str, max_fail_rate: float = 0.2, min_success: int = 1):
        """
        task          任务名（报错时用）
        max_fail_rate 允许的最大失败率。批量任务容忍个别坏数据，
                      但超过 20% 通常说明是系统性问题而非数据问题
        min_success   最少成功条数（防 total=0 的假成功）
        """
        self.task = task
        self.max_fail_rate = max_fail_rate
        self.min_success = min_success
        self.ok: list = []
        self.failed: list = []            # [(item, exc_type, reason_str), ...]

    # ---- 记账 ----
    def record_ok(self, item):
        self.ok.append(item)

    def record_fail(self, item, exc: Exception):
        self.failed.append((item, type(exc).__name__, str(exc)))

    # ---- 统计 ----
    @property
    def total(self) -> int:
        return len(self.ok) + len(self.failed)

    @property
    def fail_rate(self) -> float:
        return len(self.failed) / self.total if self.total else 1.0

    def reason_summary(self) -> Counter:
        """原因分布：失败理由 → 次数。全灭时立刻看出是不是同一个错。"""
        return Counter((t, r) for _, t, r in self.failed)

    # ---- 清算 ----
    def check(self):
        lg = get_logger()
        lg.info(f"[{self.task}] 清算：成功 {len(self.ok)} / 失败 {len(self.failed)}"
                f" / 失败率 {self.fail_rate:.1%}")
        if self.failed:
            lg.error(f"[{self.task}] 失败原因分布："
                     + "; ".join(f"{t}: {r[:80]} ×{n}"
                                 for (t, r), n in self.reason_summary().most_common(5)))

        if self.total == 0:
            raise ThresholdError(f"[{self.task}] 一条记录都没有，任务没有真正执行")
        if len(self.ok) < self.min_success:
            raise ThresholdError(
                f"[{self.task}] 成功数 {len(self.ok)} < 最低要求 {self.min_success}，"
                f"失败原因分布：{dict(self.reason_summary())}")
        if self.fail_rate > self.max_fail_rate:
            raise ThresholdError(
                f"[{self.task}] 失败率 {self.fail_rate:.1%}"
                f"（{len(self.failed)}/{self.total}）超过阈值 {self.max_fail_rate:.0%}，"
                f"疑似系统性问题。原因分布：{dict(self.reason_summary())}")
