from __future__ import annotations

import html
import json
import math
import re
import socket
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from result.io import write_json


DEFAULT_VISDOM_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "server": "http://localhost",
    "port": 8097,
    "base_url": "/",
    "env_prefix": "fatigue_assembly",
    "connection_timeout_seconds": 2.0,
    "fail_fast": False,
    "update_every": 1,
    "rolling_window_updates": 20,
    "diagnostic_every_validations": 5,
    "representative_instance_index": 0,
    "save_environment": True,
}


# Stable internal keys remain English in CSV/JSON logs.  This mapping is only
# for the human-facing Visdom panel, so that neither experiment schemas nor
# replay compatibility depend on the display language.
DISPLAY_LABELS: dict[str, str] = {
    "feasibility_shaping": "Feasibility shaping",
    "worker_matching_deficit_event_count": "Matching deficit events",
    "resource_admission_masked_action_ratio": "Admission mask ratio",
    "minimum_worker_alternatives": "Minimum worker alternatives",
    "matching_preserving_worker_action_count": "Matching-preserving actions",
    "candidate_recovery_advance_count": "Candidate recovery advances",
    "mean_worker_matching_deficit_event_count": "Mean matching deficit events",
    "mean_resource_admission_masked_action_ratio": "Mean admission mask ratio",
    "mean_minimum_worker_alternatives": "Mean minimum worker alternatives",
    "reward_mean": "平均回报",
    "reward_std": "回报标准差",
    "reward_rolling_mean": "回报滚动均值",
    "quality_score": "质量得分",
    "completed_order_ratio": "订单完成率",
    "completed_operation_ratio": "工序完成率",
    "terminated_rate": "正常终止比例",
    "truncated_rate": "截断比例",
    "flow": "流经时间",
    "cost": "重构成本",
    "variance": "工人负荷方差",
    "completion_progress": "完成进度奖励",
    "completion_bonus": "完成奖励",
    "quality": "质量奖励",
    "truncation": "截断惩罚",
    "unfinished": "未完成惩罚",
    "flow_time_objective": "流经时间目标",
    "reconfiguration_cost": "重构成本",
    "worker_load_variance": "工人负荷方差",
    "policy_loss": "策略损失",
    "value_loss": "价值损失",
    "loss": "总损失",
    "entropy": "策略熵",
    "approx_kl": "近似 KL 散度",
    "clip_fraction": "裁剪比例",
    "ratio_mean": "概率比均值",
    "learning_rate": "学习率",
    "gradient_norm": "平均梯度范数",
    "gradient_norm_max": "最大梯度范数",
    "gradient_clipped_fraction": "梯度裁剪比例",
    "pre_update_explained_variance": "更新前解释方差",
    "return_mean": "回报均值",
    "return_std": "回报标准差",
    "advantage_mean": "原始优势均值",
    "advantage_std": "原始优势标准差",
    "value_prediction_mean": "价值预测均值",
    "value_prediction_std": "价值预测标准差",
    "maximum_worker_fatigue": "最大工人疲劳",
    "mean_peak_worker_fatigue": "平均峰值疲劳",
    "fatigue_masked_action_ratio": "疲劳屏蔽动作比例",
    "safe_fatigue_limit": "疲劳安全阈值",
    "completed_reconfigurations": "完成重构次数",
    "worker_switch_ratio": "拆装工人切换比例",
    "worker_competition_event_count": "工人竞争事件数",
    "machine_waiting_for_worker_time": "机器等待工人时间",
    "transitions_per_second": "每秒状态转移数",
    "transition_count": "状态转移数",
    "sampling_wall_time_seconds": "采样耗时（秒）",
    "policy_inference_time_seconds": "策略推理耗时（秒）",
    "ppo_update_time_seconds": "PPO 更新耗时（秒）",
    "generation_time_seconds": "实例生成耗时（秒）",
    "environment_step_time_seconds": "环境推进耗时（秒）",
    "greedy_completion_rate": "贪心完成率",
    "sampled_completion_rate": "采样完成率",
    "greedy_truncated_count": "贪心截断数",
    "sampled_truncated_count": "采样截断数",
    "greedy_mean_unfinished_orders": "贪心平均未完成订单数",
    "sampled_mean_unfinished_orders": "采样平均未完成订单数",
    "schedule_violation_count": "调度违规数",
    "mean_makespan": "当前平均完工期",
    "mean_total_flow_time": "当前平均总流经时间",
    "mean_flow_time_objective": "当前平均惩罚后流经目标",
    "best_makespan": "最佳平均完工期",
    "best_total_flow_time": "最佳平均总流经时间",
    "best_flow_time_objective": "最佳平均惩罚后流经目标",
    "mean_reconfiguration_cost": "当前平均重构成本",
    "mean_worker_load_variance": "当前平均负荷方差",
    "best_reconfiguration_cost": "最佳平均重构成本",
    "best_worker_load_variance": "最佳平均负荷方差",
    "flow_gap": "流经时间差距",
    "sampled_flow_gap": "采样流经时间差距",
    "makespan_gap": "完工期差距",
    "reconfiguration_cost_gap": "重构成本差距",
    "worker_load_variance_gap": "负荷方差差距",
    "greedy": "贪心策略",
    "sampled": "采样策略",
    "mean_maximum_worker_fatigue": "平均最大工人疲劳",
    "mean_mean_peak_worker_fatigue": "平均峰值疲劳",
    "mean_safe_fatigue_limit": "平均疲劳安全阈值",
    "mean_fatigue_masked_action_ratio": "平均疲劳屏蔽动作比例",
    "mean_worker_competition_event_count": "平均工人竞争事件数",
    "mean_machine_waiting_for_worker_time": "平均机器等待工人时间",
    "mean_completed_reconfigurations": "平均完成重构次数",
    "mean_worker_switch_ratio": "平均拆装工人切换比例",
}

PRESSURE_PROFILE_LABELS: dict[str, str] = {
    "easy": "低压力场景",
    "balanced": "均衡压力场景",
    "bottleneck": "瓶颈压力场景",
    "machine_bottleneck": "机器瓶颈场景",
    "worker_bottleneck": "工人瓶颈场景",
    "mixed_bottleneck": "混合瓶颈场景",
}

PHASE_STATE_LABELS: dict[str, str] = {
    "enabled": "两阶段训练已启用",
    "phase": "当前阶段",
    "completion_target": "完成率目标",
    "consecutive_validations_required": "要求连续验证次数",
    "consecutive_validation_successes": "连续验证成功次数",
    "quality_completion_floor": "质量阶段完成率下限",
    "phase_transition_episode": "阶段切换回合",
    "accepted_quality_updates": "已接受质量更新数",
    "rejected_quality_updates": "已拒绝/回滚质量更新数",
    "formal_training_status": "正式训练状态",
}

STATE_VALUE_LABELS: dict[str, str] = {
    "feasibility": "可行性阶段",
    "quality": "质量优化阶段",
    "legacy": "传统加权模式",
    "feasibility_not_reached": "尚未达到可行性阈值",
    "quality_constrained": "质量约束训练",
    "legacy_weighted_sum": "传统加权求和",
    "transition": "切换至质量阶段",
    "accepted": "候选模型已接受",
    "rejected": "候选模型已拒绝/回滚",
}


def _display_label(name: Any) -> str:
    """Return a Chinese label while retaining unmapped IDs for diagnosis."""
    key = str(name)
    if key in DISPLAY_LABELS:
        return DISPLAY_LABELS[key]
    if key in PRESSURE_PROFILE_LABELS:
        return PRESSURE_PROFILE_LABELS[key]
    return f"指标（{key}）"


def _localized_state_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, str):
        return STATE_VALUE_LABELS.get(value, value)
    return value


def _localized_phase_state(
    phase_state: Mapping[str, Any],
    *,
    completed_episodes: int,
    total_episodes: int,
) -> dict[str, Any]:
    localized = {
        "已完成训练回合": int(completed_episodes),
        "训练总回合数": int(total_episodes),
    }
    for key, value in phase_state.items():
        localized[PHASE_STATE_LABELS.get(key, _display_label(key))] = (
            _localized_state_value(value)
        )
    return localized


def _localized_event_message(message: str) -> str:
    """Translate known training events without changing the training API."""
    text = str(message)
    direct = {
        "dashboard connected": "仪表盘已连接",
        "dashboard is in offline replay mode": "仪表盘处于离线回放模式",
    }
    if text in direct:
        return direct[text]
    if match := re.fullmatch(r"episode (\d+): validation event=(.+)", text):
        return f"回合 {match.group(1)}：验证事件为{_localized_state_value(match.group(2))}"
    if match := re.fullmatch(r"episode (\d+): new best checkpoint", text):
        return f"回合 {match.group(1)}：产生新的最佳检查点"
    if match := re.fullmatch(
        r"representative diagnostic failed at episode (\d+): (.+)",
        text,
    ):
        return f"回合 {match.group(1)}：代表性诊断失败：{match.group(2)}"
    if match := re.fullmatch(
        r"training completed with status=(.+)",
        text,
    ):
        return f"训练完成，状态：{_localized_state_value(match.group(1))}"
    return text


def resolve_visdom_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the nested configuration and the legacy enabled flag."""
    settings = dict(DEFAULT_VISDOM_SETTINGS)
    logging_config = config.get("logging", {})
    if not isinstance(logging_config, Mapping):
        raise TypeError("logging configuration must be a mapping")
    nested = logging_config.get("visdom")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("logging.visdom must be a mapping")
        settings.update(nested)
    elif "visdom_enabled" in logging_config:
        settings["enabled"] = bool(logging_config["visdom_enabled"])
    if int(settings["port"]) < 1:
        raise ValueError("logging.visdom.port must be positive")
    if int(settings["update_every"]) < 1:
        raise ValueError("logging.visdom.update_every must be positive")
    if int(settings["rolling_window_updates"]) < 1:
        raise ValueError(
            "logging.visdom.rolling_window_updates must be positive"
        )
    if int(settings["diagnostic_every_validations"]) < 1:
        raise ValueError(
            "logging.visdom.diagnostic_every_validations must be positive"
        )
    if int(settings["representative_instance_index"]) < 0:
        raise ValueError(
            "logging.visdom.representative_instance_index cannot be negative"
        )
    return settings


def override_visdom_enabled(
    config: dict[str, Any],
    enabled: bool | None,
) -> None:
    """Apply a CLI override while preserving the resolved configuration."""
    if enabled is None:
        return
    logging_config = config.setdefault("logging", {})
    nested = logging_config.setdefault("visdom", {})
    nested["enabled"] = bool(enabled)


def _safe_environment_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "run"


def visdom_environment_name(
    config: Mapping[str, Any],
    run_directory: str | Path,
) -> str:
    settings = resolve_visdom_settings(config)
    prefix = _safe_environment_component(str(settings["env_prefix"]))
    run_name = _safe_environment_component(Path(run_directory).name)
    seed = int(config["seed"])
    return f"{prefix}_{run_name}_seed{seed}"


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    return float(np.mean(values)) if values else None


def _std(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    return float(np.std(values)) if values else None


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if field in row]
    return float(np.mean(values)) if values else None


class DisabledTrainingDashboard:
    enabled = False
    connected = False
    environment = None
    validation_count = 0

    def log_update(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_validation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_capture_diagnostic(self, **kwargs: Any) -> bool:
        return False

    def log_diagnostic(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class TrainingDashboard:
    """Main-process Visdom writer with live/offline event logging."""

    enabled = True

    def __init__(
        self,
        *,
        config: dict[str, Any],
        run_directory: str | Path,
        total_episodes: int,
        visdom_class: type | None = None,
    ) -> None:
        self.config = config
        self.run_directory = Path(run_directory)
        self.total_episodes = int(total_episodes)
        self.settings = resolve_visdom_settings(config)
        self.environment = visdom_environment_name(
            config,
            self.run_directory,
        )
        self.event_log = self.run_directory / "visdom_events.log"
        self.fail_fast = bool(self.settings["fail_fast"])
        self.validation_count = 0
        self._event_messages: list[str] = []
        self._known_windows: set[str] = set()
        self._reward_history: deque[float] = deque(
            maxlen=int(self.settings["rolling_window_updates"])
        )
        self._injected_visdom_class = visdom_class is not None
        self._visdom_class = visdom_class or self._load_visdom_class()
        self.connected = self._probe_connection()
        if not self.connected and self.fail_fast:
            raise ConnectionError(
                "Visdom server is unavailable at "
                f"{self.settings['server']}:{self.settings['port']}"
            )
        if not self.connected:
            warnings.warn(
                "Visdom server is unavailable; recording offline events to "
                f"{self.event_log}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.client = self._new_client(offline=not self.connected)
        self._write_metadata()
        self.log_event(
            "dashboard connected"
            if self.connected
            else "dashboard is in offline replay mode"
        )

    @staticmethod
    def _load_visdom_class() -> type:
        try:
            from visdom import Visdom
        except ImportError as error:
            raise RuntimeError(
                "Visdom logging is enabled but visdom is not installed; "
                "install requirements.txt or run `pip install visdom==0.2.4`"
            ) from error
        return Visdom

    def _client_arguments(self) -> dict[str, Any]:
        return {
            "server": str(self.settings["server"]),
            "port": int(self.settings["port"]),
            "base_url": str(self.settings["base_url"]),
            "env": self.environment,
            "use_incoming_socket": False,
        }

    def _probe_connection(self) -> bool:
        if not self._injected_visdom_class:
            server = str(self.settings["server"])
            parsed = urlparse(
                server if "://" in server else f"http://{server}"
            )
            host = parsed.hostname or "localhost"
            try:
                with socket.create_connection(
                    (host, int(self.settings["port"])),
                    timeout=max(
                        0.05,
                        float(
                            self.settings[
                                "connection_timeout_seconds"
                            ]
                        ),
                    ),
                ):
                    return True
            except OSError:
                return False
        probe = self._visdom_class(
            **self._client_arguments(),
            raise_exceptions=False,
        )
        try:
            return bool(
                probe.check_connection(
                    timeout_seconds=float(
                        self.settings["connection_timeout_seconds"]
                    )
                )
            )
        except Exception:
            return False

    def _new_client(self, *, offline: bool) -> Any:
        return self._visdom_class(
            **self._client_arguments(),
            raise_exceptions=True,
            log_to_filename=str(self.event_log),
            offline=offline,
        )

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except Exception as error:
            if self.fail_fast:
                raise
            if self.connected:
                warnings.warn(
                    "Visdom connection failed during training; subsequent "
                    f"events will be written offline ({error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.connected = False
                self.client = self._new_client(offline=True)
                return getattr(self.client, method)(*args, **kwargs)
            warnings.warn(
                f"Visdom offline event logging failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _write_metadata(self) -> None:
        ppo = self.config["ppo"]
        network = self.config["network"]
        reward = self.config["reward"]
        training = self.config["training"]
        quality_weights = reward.get("quality_weights")
        localized_weights = (
            {
                _display_label(key): value
                for key, value in quality_weights.items()
            }
            if isinstance(quality_weights, Mapping)
            else quality_weights
        )
        summary = {
            "运行目录": str(self.run_directory),
            "Visdom 环境": self.environment,
            "随机种子": self.config["seed"],
            "计算设备": self.config["device"],
            "编码器类型": network.get("encoder_type", "typed_mlp"),
            "隐藏层维度": network["hidden_dim"],
            "消息传递层数": network.get("message_passing_layers"),
            "学习率": ppo["learning_rate"],
            "折扣因子（gamma）": ppo["gamma"],
            "GAE 系数（lambda）": ppo["gae_lambda"],
            "裁剪阈值（epsilon）": ppo["clip_epsilon"],
            "PPO 更新轮数": ppo["epochs"],
            "小批量大小": ppo["batch_size"],
            "熵系数": ppo["entropy_coefficient"],
            "价值损失系数": ppo["value_coefficient"],
            "最大梯度范数": ppo["max_grad_norm"],
            "奖励模式": _localized_state_value(
                reward.get("mode", "legacy_weighted_sum")
            ),
            "质量权重": localized_weights,
            "质量预算": reward.get("quality_budget"),
            "并行环境数": training.get("parallel_envs"),
            "验证集划分": training.get("validation_split"),
            "验证间隔（回合）": training.get(
                "validation_interval_episodes"
            ),
        }
        metadata_html = (
            "<h3>运行信息</h3><pre>"
            + html.escape(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )
            + "</pre>"
            + "<h3>只读调参指引</h3>"
            + "<ul>"
            + "<li>近似 KL / 裁剪比例 → 学习率、裁剪阈值、训练轮数</li>"
            + "<li>策略熵 → 熵系数</li>"
            + "<li>价值损失 / 解释方差 → 价值网络和价值损失系数</li>"
            + "<li>梯度裁剪 → 学习率、批量大小、奖励尺度</li>"
            + "<li>疲劳 / 重构 → 课程分布、质量权重、策略结构</li>"
            + "</ul>"
        )
        self._invoke(
            "text",
            metadata_html,
            win="00_run_metadata",
            opts={"title": "00 运行信息与调参指引"},
        )

    def _line(
        self,
        *,
        win: str,
        title: str,
        x: int | float,
        series: Mapping[str, Any],
        xlabel: str = "已完成训练回合数",
        ylabel: str | None = None,
    ) -> None:
        finite_series = [
            (name, value)
            for name, raw in series.items()
            if (value := _finite_number(raw)) is not None
        ]
        if not finite_series:
            return
        exists = win in self._known_windows
        for index, (name, value) in enumerate(finite_series):
            update = "append" if exists or index > 0 else None
            options = None
            if update is None:
                options = {
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel or "指标值",
                    "showlegend": True,
                }
            self._invoke(
                "line",
                X=np.asarray([float(x)], dtype=np.float64),
                Y=np.asarray([value], dtype=np.float64),
                win=win,
                name=_display_label(name),
                update=update,
                opts=options,
            )
        self._known_windows.add(win)

    def _status_text(
        self,
        phase_state: Mapping[str, Any],
        *,
        completed_episodes: int,
    ) -> None:
        status = _localized_phase_state(
            phase_state,
            completed_episodes=completed_episodes,
            total_episodes=self.total_episodes,
        )
        self._invoke(
            "text",
            "<h3>当前训练状态</h3><pre>"
            + html.escape(json.dumps(status, ensure_ascii=False, indent=2))
            + "</pre>",
            win="01_training_state",
            opts={"title": "01 当前训练状态"},
        )

    def log_update(
        self,
        update_row: dict[str, Any],
        episode_rows: list[dict[str, Any]],
        phase_state: Mapping[str, Any],
    ) -> None:
        update_id = int(update_row["update_id"])
        if update_id % int(self.settings["update_every"]) != 0:
            return
        completed_episodes = int(update_row["episode_end"]) + 1
        reward_mean = _mean(episode_rows, "reward")
        if reward_mean is not None:
            self._reward_history.append(reward_mean)
        rolling_reward = (
            float(np.mean(self._reward_history))
            if self._reward_history
            else None
        )
        self._status_text(
            phase_state,
            completed_episodes=completed_episodes,
        )
        self._line(
            win="10_training_effect",
            title="10 训练回报与质量",
            x=completed_episodes,
            series={
                "reward_mean": reward_mean,
                "reward_std": _std(episode_rows, "reward"),
                "reward_rolling_mean": rolling_reward,
                "quality_score": _mean(episode_rows, "quality_score"),
            },
        )
        self._line(
            win="11_training_completion",
            title="11 训练完成情况",
            x=completed_episodes,
            series={
                "completed_order_ratio": _mean(
                    episode_rows, "completed_order_ratio"
                ),
                "completed_operation_ratio": _mean(
                    episode_rows, "completed_operation_ratio"
                ),
                "terminated_rate": _rate(episode_rows, "terminated"),
                "truncated_rate": _rate(episode_rows, "truncated"),
            },
        )
        self._line(
            win="12_reward_components",
            title="12 奖励分量",
            x=completed_episodes,
            series={
                name: _mean(episode_rows, f"reward_{name}")
                for name in (
                    "flow",
                    "cost",
                    "variance",
                    "completion_progress",
                    "completion_bonus",
                    "quality",
                    "truncation",
                    "unfinished",
                    "feasibility_shaping",
                )
            },
        )
        self._line(
            win="13_training_objectives",
            title="13 训练目标值",
            x=completed_episodes,
            series={
                "flow_time_objective": _mean(
                    episode_rows, "flow_time_objective"
                ),
                "reconfiguration_cost": _mean(
                    episode_rows, "reconfiguration_cost"
                ),
                "worker_load_variance": _mean(
                    episode_rows, "worker_load_variance"
                ),
            },
        )
        self._line(
            win="20_ppo_losses",
            title="20 PPO 损失",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "policy_loss",
                    "value_loss",
                    "loss",
                    "entropy",
                )
            },
        )
        self._line(
            win="21_ppo_policy_health",
            title="21 PPO 策略更新健康度",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "approx_kl",
                    "clip_fraction",
                    "ratio_mean",
                    "learning_rate",
                )
            },
        )
        self._line(
            win="22_ppo_gradient_health",
            title="22 PPO 梯度与价值网络健康度",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "gradient_norm",
                    "gradient_norm_max",
                    "gradient_clipped_fraction",
                    "pre_update_explained_variance",
                )
            },
        )
        self._line(
            win="23_ppo_value_scales",
            title="23 PPO 回报、优势与价值尺度",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "return_mean",
                    "return_std",
                    "advantage_mean",
                    "advantage_std",
                    "value_prediction_mean",
                    "value_prediction_std",
                )
            },
        )
        self._line(
            win="30_fatigue_training",
            title="30 训练疲劳指标",
            x=completed_episodes,
            series={
                "maximum_worker_fatigue": _mean(
                    episode_rows, "maximum_worker_fatigue"
                ),
                "mean_peak_worker_fatigue": _mean(
                    episode_rows, "mean_peak_worker_fatigue"
                ),
                "fatigue_masked_action_ratio": _mean(
                    episode_rows, "fatigue_masked_action_ratio"
                ),
                "safe_fatigue_limit": _mean(
                    episode_rows, "safe_fatigue_limit"
                ),
            },
        )
        self._line(
            win="31_reconfiguration_training",
            title="31 训练重构与工人压力",
            x=completed_episodes,
            series={
                "completed_reconfigurations": _mean(
                    episode_rows, "completed_reconfigurations"
                ),
                "worker_switch_ratio": _mean(
                    episode_rows, "worker_switch_ratio"
                ),
                "worker_competition_event_count": _mean(
                    episode_rows, "worker_competition_event_count"
                ),
                "worker_matching_deficit_event_count": _mean(
                    episode_rows,
                    "worker_matching_deficit_event_count",
                ),
                "resource_admission_masked_action_ratio": _mean(
                    episode_rows,
                    "resource_admission_masked_action_ratio",
                ),
                "minimum_worker_alternatives": _mean(
                    episode_rows,
                    "minimum_worker_alternatives",
                ),
                "matching_preserving_worker_action_count": _mean(
                    episode_rows,
                    "matching_preserving_worker_action_count",
                ),
                "candidate_recovery_advance_count": _mean(
                    episode_rows,
                    "candidate_recovery_advance_count",
                ),
                "machine_waiting_for_worker_time": _mean(
                    episode_rows, "machine_waiting_for_worker_time"
                ),
            },
        )
        self._line(
            win="40_training_throughput",
            title="40 训练吞吐量",
            x=completed_episodes,
            series={
                "transitions_per_second": update_row.get(
                    "transitions_per_second"
                ),
                "transition_count": update_row.get("transition_count"),
            },
        )
        self._line(
            win="41_training_times",
            title="41 训练耗时分解",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "sampling_wall_time_seconds",
                    "policy_inference_time_seconds",
                    "ppo_update_time_seconds",
                    "generation_time_seconds",
                    "environment_step_time_seconds",
                )
            },
        )
        profiles = sorted(
            {
                str(row["pressure_type"])
                for row in episode_rows
                if row.get("pressure_type") is not None
            }
        )
        self._line(
            win="50_pressure_completion",
            title="50 各压力场景的完成率",
            x=completed_episodes,
            series={
                profile: _mean(
                    [
                        row
                        for row in episode_rows
                        if str(row.get("pressure_type")) == profile
                    ],
                    "completed_order_ratio",
                )
                for profile in profiles
            },
        )
        self._line(
            win="51_pressure_quality",
            title="51 各压力场景的质量得分",
            x=completed_episodes,
            series={
                profile: _mean(
                    [
                        row
                        for row in episode_rows
                        if str(row.get("pressure_type")) == profile
                    ],
                    "quality_score",
                )
                for profile in profiles
            },
        )

    def log_validation(
        self,
        validation_row: dict[str, Any],
        *,
        best_validation: Mapping[str, Any] | None,
        phase_state: Mapping[str, Any],
    ) -> None:
        self.validation_count += 1
        completed_episodes = int(validation_row["episode"])
        self._status_text(
            phase_state,
            completed_episodes=completed_episodes,
        )
        self._line(
            win="60_validation_feasibility",
            title="60 验证集可行性",
            x=completed_episodes,
            series={
                "greedy_completion_rate": validation_row.get(
                    "completion_rate"
                ),
                "sampled_completion_rate": validation_row.get(
                    "sampled_completion_rate"
                ),
                "greedy_truncated_count": validation_row.get(
                    "truncated_count"
                ),
                "sampled_truncated_count": validation_row.get(
                    "sampled_truncated_count"
                ),
                "greedy_mean_unfinished_orders": validation_row.get(
                    "mean_unfinished_orders"
                ),
                "sampled_mean_unfinished_orders": validation_row.get(
                    "sampled_mean_unfinished_orders"
                ),
                "schedule_violation_count": validation_row.get(
                    "schedule_violation_count"
                ),
            },
        )
        self._line(
            win="61_validation_flow",
            title="61 验证集流经时间目标",
            x=completed_episodes,
            series={
                "mean_makespan": validation_row.get("mean_makespan"),
                "mean_total_flow_time": validation_row.get(
                    "mean_total_flow_time"
                ),
                "mean_flow_time_objective": validation_row.get(
                    "mean_flow_time_objective"
                ),
                "best_makespan": (
                    best_validation.get("mean_makespan")
                    if best_validation
                    else None
                ),
                "best_total_flow_time": (
                    best_validation.get("mean_total_flow_time")
                    if best_validation
                    else None
                ),
                "best_flow_time_objective": (
                    best_validation.get("mean_flow_time_objective")
                    if best_validation
                    else None
                ),
            },
        )
        self._line(
            win="62_validation_quality_components",
            title="62 验证集质量分量",
            x=completed_episodes,
            series={
                "mean_reconfiguration_cost": validation_row.get(
                    "mean_reconfiguration_cost"
                ),
                "mean_worker_load_variance": validation_row.get(
                    "mean_worker_load_variance"
                ),
                "best_reconfiguration_cost": (
                    best_validation.get("mean_reconfiguration_cost")
                    if best_validation
                    else None
                ),
                "best_worker_load_variance": (
                    best_validation.get("mean_worker_load_variance")
                    if best_validation
                    else None
                ),
            },
        )
        self._line(
            win="63_validation_gaps",
            title="63 验证集相对启发式差距（%）",
            x=completed_episodes,
            series={
                "flow_gap": validation_row.get(
                    "mean_relative_heuristic_gap_percent"
                ),
                "sampled_flow_gap": validation_row.get(
                    "sampled_mean_relative_heuristic_gap_percent"
                ),
                "makespan_gap": validation_row.get(
                    "mean_makespan_heuristic_gap_percent"
                ),
                "reconfiguration_cost_gap": validation_row.get(
                    "mean_reconfiguration_cost_heuristic_gap_percent"
                ),
                "worker_load_variance_gap": validation_row.get(
                    "mean_worker_load_variance_heuristic_gap_percent"
                ),
            },
            ylabel="相对启发式差距（%）",
        )
        self._line(
            win="66_validation_proxy_return",
            title="66 验证集可行性代理回报",
            x=completed_episodes,
            series={
                "greedy": validation_row.get(
                    "mean_feasibility_proxy_return"
                ),
                "sampled": validation_row.get(
                    "sampled_mean_feasibility_proxy_return"
                ),
            },
        )
        self._line(
            win="64_validation_fatigue",
            title="64 验证集疲劳指标",
            x=completed_episodes,
            series={
                "mean_maximum_worker_fatigue": validation_row.get(
                    "mean_maximum_worker_fatigue"
                ),
                "mean_mean_peak_worker_fatigue": validation_row.get(
                    "mean_mean_peak_worker_fatigue"
                ),
                "mean_safe_fatigue_limit": validation_row.get(
                    "mean_safe_fatigue_limit"
                ),
                "mean_fatigue_masked_action_ratio": validation_row.get(
                    "mean_fatigue_masked_action_ratio"
                ),
            },
        )
        self._line(
            win="65_validation_reconfiguration",
            title="65 验证集重构与工人压力",
            x=completed_episodes,
            series={
                "mean_worker_competition_event_count": validation_row.get(
                    "mean_worker_competition_event_count"
                ),
                "mean_worker_matching_deficit_event_count": (
                    validation_row.get(
                        "mean_worker_matching_deficit_event_count"
                    )
                ),
                "mean_resource_admission_masked_action_ratio": (
                    validation_row.get(
                        "mean_resource_admission_masked_action_ratio"
                    )
                ),
                "mean_minimum_worker_alternatives": validation_row.get(
                    "mean_minimum_worker_alternatives"
                ),
                "mean_machine_waiting_for_worker_time": validation_row.get(
                    "mean_machine_waiting_for_worker_time"
                ),
                "mean_completed_reconfigurations": validation_row.get(
                    "mean_completed_reconfigurations"
                ),
                "mean_worker_switch_ratio": validation_row.get(
                    "mean_worker_switch_ratio"
                ),
            },
        )

    def log_event(self, message: str) -> None:
        self._event_messages.append(_localized_event_message(str(message)))
        rendered = "<br>".join(
            html.escape(value) for value in self._event_messages[-200:]
        )
        self._invoke(
            "text",
            rendered,
            win="02_training_events",
            opts={"title": "02 训练事件"},
        )

    def should_capture_diagnostic(
        self,
        *,
        validation_event: str,
        is_new_best: bool,
    ) -> bool:
        interval = int(
            self.settings["diagnostic_every_validations"]
        )
        return (
            self.validation_count % interval == 0
            or validation_event in {"transition", "rejected"}
            or bool(is_new_best)
        )

    def log_diagnostic(
        self,
        trace: dict[str, Any],
        *,
        completed_episodes: int,
    ) -> None:
        output_directory = self.run_directory / "diagnostics"
        output_directory.mkdir(parents=True, exist_ok=True)
        output = (
            output_directory
            / f"validation_{int(completed_episodes):07d}.json"
        )
        write_json(output, trace)
        svg = build_schedule_gantt_svg(trace)
        self._invoke(
            "svg",
            svg,
            win="70_representative_schedule",
            opts={"title": "70 代表性实例调度甘特图"},
        )
        fatigue_trace = trace.get("fatigue_trace", [])
        worker_names = list(trace.get("worker_ids", []))
        if fatigue_trace and worker_names:
            times = np.asarray(
                [row["time"] for row in fatigue_trace],
                dtype=np.float64,
            )
            values = np.asarray(
                [
                    [row["workers"][name] for name in worker_names]
                    for row in fatigue_trace
                ],
                dtype=np.float64,
            )
            limit = float(trace["safe_fatigue_limit"])
            values = np.column_stack(
                [values, np.full(len(times), limit, dtype=np.float64)]
            )
            names = [*worker_names, "疲劳安全阈值"]
            self._invoke(
                "line",
                X=np.tile(times[:, None], (1, len(names))),
                Y=values,
                win="71_representative_fatigue_trace",
                opts={
                    "title": "71 代表性实例的工人疲劳曲线",
                    "xlabel": "调度时间（分钟）",
                    "ylabel": "疲劳值",
                    "legend": names,
                    "showlegend": True,
                },
            )
            self._known_windows.add(
                "71_representative_fatigue_trace"
            )
        peaks = trace.get("worker_peak_fatigue", {})
        if peaks:
            names = list(peaks)
            values = [float(peaks[name]) for name in names]
            if len(values) == 1:
                # Visdom 0.2.4 squeezes a single bar to a scalar.
                names.append("")
                values.append(0.0)
            self._invoke(
                "bar",
                X=np.asarray(values, dtype=np.float64),
                win="72_representative_peak_fatigue",
                opts={
                    "title": "72 代表性实例的工人峰值疲劳",
                    "rownames": names,
                    "ylabel": "峰值疲劳",
                },
            )
        self.log_event(
            "代表性诊断已保存（回合 "
            f"{completed_episodes}）：{output.name}"
        )

    def close(self) -> None:
        if bool(self.settings["save_environment"]):
            self._invoke("save", [self.environment])


def create_training_dashboard(
    *,
    config: dict[str, Any],
    run_directory: str | Path,
    total_episodes: int,
    visdom_class: type | None = None,
) -> TrainingDashboard | DisabledTrainingDashboard:
    settings = resolve_visdom_settings(config)
    if not bool(settings["enabled"]):
        return DisabledTrainingDashboard()
    return TrainingDashboard(
        config=config,
        run_directory=run_directory,
        total_episodes=total_episodes,
        visdom_class=visdom_class,
    )


def build_schedule_gantt_svg(trace: Mapping[str, Any]) -> str:
    """Render machine and worker use without adding a plotting dependency."""
    schedule = list(trace.get("schedule", []))
    reconfigurations = list(trace.get("reconfigurations", []))
    machine_ids = sorted(
        {
            str(row["machine_id"])
            for row in [*schedule, *reconfigurations]
        }
    )
    worker_ids = sorted(
        {str(row["worker_id"]) for row in reconfigurations}
        | {str(value) for value in trace.get("worker_ids", [])}
    )
    rows = [
        *(("machine", value) for value in machine_ids),
        *(("worker", value) for value in worker_ids),
    ]
    width = 1200
    left = 150
    right = 30
    top = 54
    row_height = 30
    axis_height = 52
    height = top + max(1, len(rows)) * row_height + axis_height
    plot_width = width - left - right
    maximum_time = max(
        [
            _finite_number(row.get("end")) or 0.0
            for row in [*schedule, *reconfigurations]
        ]
        or [1.0]
    )
    maximum_time = max(maximum_time, 1.0)
    row_index = {value: index for index, value in enumerate(rows)}
    module_palette = (
        "#4C78A8",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#EDC948",
    )
    modules = sorted(
        {str(row.get("required_module", "")) for row in schedule}
    )
    module_color = {
        module: module_palette[index % len(module_palette)]
        for index, module in enumerate(modules)
    }

    def x_position(value: Any) -> float:
        number = _finite_number(value) or 0.0
        return left + plot_width * number / maximum_time

    def rect(
        *,
        row_key: tuple[str, str],
        start: Any,
        end: Any,
        color: str,
        label: str,
        opacity: float = 1.0,
    ) -> str:
        y = top + row_index[row_key] * row_height + 5
        x = x_position(start)
        rectangle_width = max(1.0, x_position(end) - x)
        safe_label = html.escape(label)
        return (
            f'<rect x="{x:.2f}" y="{y}" width="{rectangle_width:.2f}" '
            f'height="20" rx="2" fill="{color}" opacity="{opacity}">'
            f"<title>{safe_label}</title></rect>"
            f'<text x="{x + 3:.2f}" y="{y + 14}" font-size="10" '
            f'fill="white">{safe_label}</text>'
        )

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18" font-weight="bold">'
        + html.escape(
            f"代表性调度：{trace.get('instance_id', '')}"
        )
        + "</text>",
    ]
    resource_labels = {"machine": "机器", "worker": "工人"}
    for index, (kind, identifier) in enumerate(rows):
        y = top + index * row_height
        fill = "#F7F7F7" if index % 2 == 0 else "#FFFFFF"
        content.append(
            f'<rect x="0" y="{y}" width="{width}" '
            f'height="{row_height}" fill="{fill}"/>'
        )
        content.append(
            f'<text x="8" y="{y + 20}" font-size="12">'
            f"{resource_labels[kind]}：{html.escape(identifier)}</text>"
        )
    for row in schedule:
        module = str(row.get("required_module", ""))
        content.append(
            rect(
                row_key=("machine", str(row["machine_id"])),
                start=row["start"],
                end=row["end"],
                color=module_color.get(module, "#4C78A8"),
                label=f"工序 {row['operation_id']}（模块 {module}）",
            )
        )
    stage_colors = {"DIS": "#B279A2", "INS": "#FF9DA7"}
    stage_labels = {"DIS": "拆卸（DIS）", "INS": "安装（INS）"}
    for row in reconfigurations:
        stage = str(row["stage"])
        color = stage_colors.get(stage, "#9D755D")
        label = f"{stage_labels.get(stage, stage)}：工序 {row['operation_id']}"
        content.append(
            rect(
                row_key=("machine", str(row["machine_id"])),
                start=row["start"],
                end=row["end"],
                color=color,
                label=label,
                opacity=0.9,
            )
        )
        content.append(
            rect(
                row_key=("worker", str(row["worker_id"])),
                start=row["start"],
                end=row["end"],
                color=color,
                label=label,
                opacity=0.9,
            )
        )
    axis_y = top + max(1, len(rows)) * row_height
    content.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" '
        f'y2="{axis_y}" stroke="#333"/>'
    )
    for index in range(6):
        value = maximum_time * index / 5
        x = x_position(value)
        content.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" '
            f'y2="{axis_y + 6}" stroke="#333"/>'
        )
        content.append(
            f'<text x="{x:.2f}" y="{axis_y + 22}" font-size="11" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    content.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{axis_y + 42}" '
        'font-size="12" text-anchor="middle">时间（分钟）</text>'
    )
    legend_x = 540
    legend_entries = [
        *((f"模块 {label}", color) for label, color in module_color.items()),
        *((stage_labels.get(label, label), color) for label, color in stage_colors.items()),
    ]
    for index, (label, color) in enumerate(legend_entries):
        x = legend_x + index * 105
        content.append(
            f'<rect x="{x}" y="15" width="14" height="14" fill="{color}"/>'
            f'<text x="{x + 18}" y="27" font-size="11">'
            f"{html.escape(label)}</text>"
        )
    content.append("</svg>")
    return "".join(content)


__all__ = [
    "DisabledTrainingDashboard",
    "TrainingDashboard",
    "build_schedule_gantt_svg",
    "create_training_dashboard",
    "override_visdom_enabled",
    "resolve_visdom_settings",
    "visdom_environment_name",
]
