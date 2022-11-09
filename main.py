"""
NNI hyperparameter optimization example.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]}
}

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py' #call the prepared torch model
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

# Run it!
experiment.run(port=8078, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()
