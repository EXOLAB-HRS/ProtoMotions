"""Regression tests for the steering root-velocity double buffer.

An in-episode command change must not seed the double buffer: doing so makes the
measured root velocity exactly zero for that control step, which shows up as a
spurious near-minimum heading reward and, in evaluation, as a latched steering
failure for the whole rollout.
"""

import torch

from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)


class _RootState:
    def __init__(self, root_pos):
        self.root_pos = root_pos


class _Simulator:
    def __init__(self, root_pos):
        self.root_pos = root_pos

    def get_root_state(self):
        return _RootState(self.root_pos)


class _Env:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.device = "cpu"
        self.progress_buf = torch.zeros(num_envs, dtype=torch.int64)
        self.simulator = _Simulator(torch.zeros(num_envs, 3))


def _make_control(num_envs=1, dwell=2):
    env = _Env(num_envs)
    config = SteeringControlConfig(
        heading_change_steps_min=dwell,
        heading_change_steps_max=dwell + 1,
    )
    control = SteeringControl(config, env)
    control.reset(torch.arange(num_envs))
    return control, env


def _advance(control, env, displacement):
    """Advance one control step by moving the root and stepping the component."""
    env.simulator.root_pos = env.simulator.root_pos + displacement
    control.step()
    env.progress_buf += 1


def test_command_change_preserves_measured_root_velocity():
    control, env = _make_control(dwell=2)
    step_displacement = torch.tensor([[0.0333, 0.0, 0.0]])

    # Run past the dwell so that step() resamples the command.
    changed = False
    for _ in range(4):
        before = control._heading_change_steps.clone()
        _advance(control, env, step_displacement)
        changed = changed or bool((control._heading_change_steps != before).any())
        delta = env.simulator.root_pos - control._prev_root_pos
        torch.testing.assert_close(delta, step_displacement)

    assert changed, "the command never changed; the dwell setup is wrong"


def test_environment_reset_seeds_the_double_buffer():
    control, env = _make_control()
    _advance(control, env, torch.tensor([[0.0333, 0.0, 0.0]]))

    # A real reset teleports the humanoid; stale history would report a huge speed.
    env.simulator.root_pos = torch.tensor([[60.0, 60.0, 0.9]])
    control.reset(torch.arange(env.num_envs))

    delta = env.simulator.root_pos - control._prev_root_pos
    torch.testing.assert_close(delta, torch.zeros(1, 3))
