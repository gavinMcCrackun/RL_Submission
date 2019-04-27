import os
import time
import click
from project.deep_rl.configs.manager import ConfigManager
# need this because it executes the gym environment registration
import project.mdp.line_env


@click.command()
@click.argument('config_name')
@click.option('-m', '--model-name', default=None)
def run(config_name, model_name=None):
    cfg = ConfigManager.load(config_name)

    if model_name is None:
        model_name = '-'.join([
            cfg.env_name.lower(),
            cfg.policy_name.replace('_', '-'),
            os.path.splitext(os.path.basename(config_name))[0] if config_name else 'default',
            str(int(time.time()))
        ])

    model_name = model_name.lower()

    # setup output dir for episodes
    cfg.env.render_dir = os.path.join(os.getcwd(), 'figs', model_name)
    os.makedirs(os.path.join(os.getcwd(), 'figs'), exist_ok=True)
    os.makedirs(cfg.env.render_dir, exist_ok=False)

    # run RL
    cfg.start_training(model_name)


if __name__ == '__main__':
    run()
