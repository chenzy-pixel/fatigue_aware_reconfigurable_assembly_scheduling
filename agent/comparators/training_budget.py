"""Explicit stopping and validation contracts for learning requests."""

PROTOCOL = "episode_training_v1"


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def training_budget(config, *, steps=None, episodes=None):
    if steps is not None and episodes is not None:
        raise ValueError("steps and episodes are mutually exclusive")
    training = config['training']
    if steps is None and episodes is None:
        if training.get('budget_unit') != 'episodes':
            raise ValueError("formal training requires budget_unit=episodes")
        episodes = training['episodes']
    unit = 'steps' if steps is not None else 'episodes'
    return {
        'protocol': PROTOCOL,
        'budget_unit': unit,
        'budget_amount': positive_int(steps if unit == 'steps' else episodes, unit),
        'episodes_per_update': positive_int(training['episodes_per_update'], 'episodes_per_update'),
        'validation_interval_epochs': positive_int(training['validation_interval_epochs'], 'validation_interval_epochs'),
        'validation_schedule': 'epoch_interval_and_final' if unit == 'episodes' else 'step_fifths_and_final',
        'ddqn_exploration_schedule': 'episode_batch_linear_half_v1' if unit == 'episodes' else 'step_linear_half_v1',
    }
