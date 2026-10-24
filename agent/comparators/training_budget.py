"""Episode budget and validation schedule shared by all learning methods."""
PROTOCOL = "episode_training_v1"

def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def training_budget(config, *, episodes=None):
    training = config['training']
    if training.get('budget_unit') != 'episodes':
        raise ValueError('training requires budget_unit=episodes')
    return {'protocol': PROTOCOL, 'budget_unit': 'episodes',
            'budget_amount': positive_int(training['episodes'] if episodes is None else episodes, 'episodes'),
            'episodes_per_update': positive_int(training['episodes_per_update'], 'episodes_per_update'),
            'validation_interval_epochs': positive_int(training['validation_interval_epochs'], 'validation_interval_epochs'),
            'validation_schedule': 'epoch_interval_and_final',
            'ddqn_exploration_schedule': 'episode_batch_linear_half_v1'}
