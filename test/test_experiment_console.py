"""Legacy console layout and execution-event progress with real counters."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import time

import pytest

from configs.experiment_suite import load_suite, arm_config
from agent.comparators.training_budget import training_budget
from result.console_reporter import ExperimentConsoleReporter, TrainingConsoleReporter, report_training_failure


def reporter(arm='full', *, writer=None, steps=None):
    config = arm_config(load_suite(), arm, 11)
    output = []
    r = ExperimentConsoleReporter(config, training_budget(config, steps=steps), load_suite()['arms'][arm]['method'],
                                  20, 27, 13, started_at=time.perf_counter(), writer=writer or output.append)
    return r, output


def test_full_uses_established_header_epoch_metrics_and_done():
    r, lines = reporter()
    r.start_run()
    r.epoch_update(dict(epoch=1, update_id=1, episodes_started=20, training_steps=1234, collect_seconds=12.,
                        update_seconds=2., learning_rate=1e-4,
                        losses=dict(policy_loss=-.02, value_loss=.3, entropy=1.4, approx_kl=.006)),
                   [dict(reward_sum=.6, metrics=dict(task_succeeded=True, operation_progress=1., preference_quality_score=.4)),
                    dict(reward_sum=-.2, metrics=dict(task_succeeded=False, operation_progress=.8, preference_quality_score=1.))])
    r.done(selector_state={'best_episode': 100, 'best_completion_rate': .9,
                          'best_preference_balanced_quality_score': .3}, run_directory='artifacts',
           best_checkpoint='best_checkpoint.pt', last_checkpoint='last_checkpoint.pt', elapsed_seconds=9.)
    text = '\n'.join(lines)
    assert '[RUN] full | method=ppo | seed=11 | envs=20' in text
    assert 'episodes=2000 | epochs=100 | batch=20' in text
    assert 'val=27 × repeats=3 × prefs=13 | envs=20 | schedule=every 5 epochs + final' in text
    assert '[train] ep 20/2000 | upd 1 | complete 50.0% | reward 0.200 | progress 0.900 | quality 0.700' in text
    assert 'p_loss -0.0200 | v_loss 0.300 | ent 1.40 | kl 0.0060' in text
    assert 'epoch 1/100 | steps=1234 | samples=20' in text
    assert '[DONE] status=best_selected' in text and 'checkpoint=best_checkpoint.pt' in text


def test_ddqn_reports_greedy_budget_and_td_loss():
    r, lines = reporter('rule_ddqn')
    r.start_run()
    r.epoch_update(dict(epoch=2, update_id=2, episodes_started=40, training_steps=456, collect_seconds=1.,
                        update_seconds=0., losses={'td_loss': .04}, exploration_epsilon=.8), [])
    text = '\n'.join(lines)
    assert 'val=27 × repeats=1 × prefs=13' in text and 'formal aggregation=greedy_single_v1' in text
    assert '[train] ep 40/2000' in text and 'td_loss 0.0400 | epsilon 0.800' in text
    assert 'p_loss' not in text


def test_step_budget_header_reports_its_actual_schedule():
    r, lines = reporter(steps=1000)
    r.start_run()
    assert 'steps=1000 | epochs=variable' in '\n'.join(lines)
    assert 'schedule=step fifths + final' in '\n'.join(lines)


def test_validation_keeps_raw_and_delivered_counts_and_safety():
    r, lines = reporter()
    result = {'summary': {'safe': False, 'quality': .25}, 'rows': [{'success': True}, {'success': False}],
              'deliveries': [{'raw': [{'success': True, 'quality': .2, 'metrics': {'operation_progress': 1.}},
                                      {'success': False, 'quality': 1., 'metrics': {'operation_progress': .5}},
                                      {'success': True, 'quality': .4, 'metrics': {'operation_progress': 1.}}]}]}
    original = deepcopy(result)
    row = r.validation_row(result, 100)
    row['checkpoint_event'] = 'best_retained'
    r.validation(row, selector_state={'best_episode': 80})
    assert result == original
    assert 'complete 50.0% | quality 0.250000' in '\n'.join(lines)
    assert 'safety=FAIL | event=best_retained | best_ep=80' in '\n'.join(lines)
    assert 'requests=3 | deliveries=2 | raw complete=66.7%' in '\n'.join(lines)


@pytest.mark.parametrize('failing', [False, True])
def test_phase_emits_once_at_start_and_finish_with_latest_counts(failing, monkeypatch):
    r, lines = reporter()
    clock = [100.]
    monkeypatch.setattr('result.console_reporter.time', SimpleNamespace(perf_counter=lambda: clock[0]))
    def create_thread(*args, **kwargs):
        raise AssertionError('phase events run in the execution thread')
    monkeypatch.setattr(threading, 'Thread', create_thread)
    def work():
        with r.phase('val', epoch=5, total=1053, unit='rollouts') as update:
            update(dict(completed=20, active=20, environment_steps=456))
            clock[0] += 60.
            update(dict(completed=40, active=20, environment_steps=900))
            assert len(lines) == 1 and '[val] start' in lines[0]
            if failing:
                raise RuntimeError('worker failed')
            update(dict(completed=1053, active=0, environment_steps=7890))
    if failing:
        with pytest.raises(RuntimeError, match='worker failed'):
            work()
    else:
        work()
    assert len(lines) == 2
    assert ('failed' if failing else 'done') in lines[-1]
    assert 'elapsed=1m 00s' in lines[-1]
    if not failing:
        assert 'rollouts 1053/1053 | active=0 | steps=7890' in lines[-1]
    else:
        assert 'rollouts 40/1053 | active=20 | steps=900' in lines[-1]


def test_existing_failure_reporter_still_emits_exit_code_and_trailing_line():
    output = []
    report_training_failure(RuntimeError('failed'), run_directory='artifacts', exit_code=1, writer=output.append)
    assert '[FAILED] RuntimeError' in output and 'exit_code=1' in output and output[-1] == ''


def test_checkpoint_event_names_the_files_saved():
    r, lines = reporter()
    r.checkpoint_saved(5, 'best_initialized', last_checkpoint='last.pt', best_checkpoint='best.pt')
    r.checkpoint_saved(10, 'best_retained', last_checkpoint='last.pt')
    assert lines == ['[checkpoint] epoch 5 | event=best_initialized | last=last.pt | best=best.pt',
                     '[checkpoint] epoch 10 | event=best_retained | last=last.pt']
