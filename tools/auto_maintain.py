"""
启动时自动维护（run.py / app.py 共用，入口零改动：只加一行调用）。

职责（全部后台线程，绝不阻塞启动、绝不抛异常打断主流程）：
1. 日更数据：交易日首次启动时跑 train/refresh_panel.py 增量补数；
2. 月更模型：当月首次启动时跑 build_dataset_panel.py + export_champion.py
   全量重训并导出 champion 模型（h5/price21）。

并发安全：
- 锁文件 train/data/.refreshing：数据刷新期间 forecast_tools.get_forecast
  会等待其消失（防止读到一半新一半旧的横截面）；
- 模型重训在锁外进行（它只动 parquet 和 artifacts，不动 panel/*.csv）；
  export_champion 落盘改为「临时文件 + os.replace」原子替换，即使极端
  时序下读到新旧交叠，LightGBM 特征校验也会报错被 get_forecast 捕获，
  不会静默出错。

失败策略：任何一步失败只记 marker + 打日志，30 分钟冷却后允许重试，
当天已成功则不再跑。
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, date

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_TOOLS_DIR)
TRAIN_DIR = os.path.join(ROOT, "train")
DATA_DIR = os.path.join(TRAIN_DIR, "data")
MARKER_PATH = os.path.join(DATA_DIR, ".auto_maintain.json")
LOCK_PATH = os.path.join(DATA_DIR, ".refreshing")

PANEL_COOLDOWN_MIN = 30      # 失败后重试冷却（分钟）
SUB_TIMEOUT = {"refresh": 600, "build": 900, "export": 600}

_done = threading.Event()    # 本进程内维护只跑一次


def _load_marker() -> dict:
    try:
        with open(MARKER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_marker(m: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = MARKER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MARKER_PATH)


def _run(script: str, timeout: int):
    """在子进程里跑 train/ 脚本（环境隔离，BaoStock 登录互不干扰）。"""
    print(f"[auto-maintain] 运行 {os.path.basename(script)} ...")
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, script],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        ok = r.returncode == 0
        tail = (r.stdout or "")[-800:]
        if not ok:
            tail += "\n" + (r.stderr or "")[-800:]
        print(f"[auto-maintain] {os.path.basename(script)} "
              f"{'完成' if ok else '失败'}（{time.time()-t0:.0f}s）")
        if not ok:
            print(tail)
        return ok
    except subprocess.TimeoutExpired:
        print(f"[auto-maintain] {os.path.basename(script)} 超时（{timeout}s），已终止")
        return False
    except Exception as e:
        print(f"[auto-maintain] {os.path.basename(script)} 异常：{e}")
        return False


def panel_due(marker: dict) -> bool:
    """今天没成功刷过，且距上次尝试超过冷却期。"""
    if marker.get("panel_ok") == str(date.today()):
        return False
    last_try = marker.get("panel_attempt")
    if last_try:
        try:
            dt = datetime.fromisoformat(last_try)
            if (datetime.now() - dt).total_seconds() < PANEL_COOLDOWN_MIN * 60:
                return False
        except Exception:
            pass
    return True


def model_due(marker: dict) -> bool:
    """当月没重训过，或模型文件缺失（失败则下次启动自动重试）。"""
    if marker.get("model_month") != date.today().strftime("%Y-%m"):
        return True
    return not os.path.exists(os.path.join(TRAIN_DIR, "artifacts", "lgbm_panel.txt"))


def _maintain_once():
    marker = _load_marker()

    # ---- 1. 日更数据（持锁，forecast 会等待）----
    if panel_due(marker):
        marker["panel_attempt"] = datetime.now().isoformat(timespec="seconds")
        _save_marker(marker)
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            # 独占创建锁文件；残留锁（上次进程崩溃）超过 30 分钟视为死锁，强删
            if os.path.exists(LOCK_PATH):
                if time.time() - os.path.getmtime(LOCK_PATH) > 1800:
                    os.remove(LOCK_PATH)
                else:
                    print("[auto-maintain] 检测到别的进程正在刷新，跳过本轮")
                    return
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return
        try:
            ok = _run(os.path.join(TRAIN_DIR, "refresh_panel.py"),
                      SUB_TIMEOUT["refresh"])
            if ok:
                marker = _load_marker()
                marker["panel_ok"] = str(date.today())
                _save_marker(marker)
        finally:
            try:
                os.remove(LOCK_PATH)
            except OSError:
                pass

    # ---- 2. 月更模型（锁外：不碰 panel/*.csv，与打分无锁竞争）----
    marker = _load_marker()
    if model_due(marker):
        ok = _run(os.path.join(TRAIN_DIR, "build_dataset_panel.py"),
                  SUB_TIMEOUT["build"])
        if ok:
            ok = _run(os.path.join(TRAIN_DIR, "export_champion.py"),
                      SUB_TIMEOUT["export"])
        if ok:
            marker = _load_marker()
            marker["model_month"] = date.today().strftime("%Y-%m")
            _save_marker(marker)


def _worker():
    try:
        _maintain_once()
    except Exception as e:                       # 兜底：绝不让后台线程炸主程序
        print(f"[auto-maintain] 后台维护异常（已忽略）：{e}")
    finally:
        _done.set()


def start_maintenance():
    """启动后台维护线程（幂等：同进程多次调用只生效一次）。"""
    if getattr(start_maintenance, "_started", False):
        return
    start_maintenance._started = True
    t = threading.Thread(target=_worker, name="auto-maintain", daemon=True)
    t.start()
    print("[auto-maintain] 后台维护线程已启动（日更数据 / 月更模型，详见 tools/auto_maintain.py）")


def wait_refresh_lock(timeout: int = 600) -> bool:
    """供 forecast_tools 调用：等待数据刷新结束。True=无锁或已释放。"""
    if not os.path.exists(LOCK_PATH):
        return True
    t0 = time.time()
    while os.path.exists(LOCK_PATH) and time.time() - t0 < timeout:
        time.sleep(2)
    return not os.path.exists(LOCK_PATH)


if __name__ == "__main__":
    # 自检：同步跑一轮（可单独执行观察输出）
    _maintain_once()
    print(">>> auto_maintain 自检完成")
