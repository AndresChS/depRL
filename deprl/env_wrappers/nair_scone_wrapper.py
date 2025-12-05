import numpy as np #type: ignore

from .wrappers import ExceptionWrapper


class CustomSconeException(Exception):
    """
    Custom exception class for Scone. The SconePy Interface doesn't define a scone exception
    at the moment and I have never encountered a simulator-based exception until now.
    Will update when that is the case.
    """

    pass


class NairSconeWrapper(ExceptionWrapper):
    """Wrapper for Nair envs implemented in SconeRL Hyfydy, compatible with
    gymnasium (returns 5-tuple from step, 2-tuple from reset).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = CustomSconeException

    def render(self, *args, **kwargs):
        pass

    def muscle_lengths(self):
        length = self.unwrapped.model.muscle_fiber_length_array()
        return length

    def muscle_forces(self):
        force = self.unwrapped.model.muscle_force_array()
        return force

    def muscle_velocities(self):
        velocity = self.unwrapped.model.muscle_fiber_velocity_array()
        return velocity

    def muscle_activity(self):
        return self.unwrapped.model.muscle_activation_array()

    def write_now(self):
        if self.unwrapped.store_next:
            self.unwrapped.model.write_results(
                self.unwrapped.output_dir, f"{self.unwrapped.episode:05d}_{self.unwrapped.total_reward:.3f}"
            )
        self.unwrapped.episode += 1
        self.unwrapped.store_next = False

    def _inner_step(self, action):
        """
        takes an action and advances environment by 1 step.
        Changed to allow for correct sto saving.
        
        if not self.unwrapped.has_reset:
            raise Exception("You have to call reset() once before step()")

        if self.unwrapped.clip_actions:
            action = np.clip(action, 0, 0.5)
        else:
            action = np.clip(action, 0, 1.0)

        if self.unwrapped.use_delayed_actuators:
            self.unwrapped.model.set_delayed_actuator_inputs(action)
        else:
            self.unwrapped.model.set_actuator_inputs(action)

        self.unwrapped.model.advance_simulation_to(self.time + self.step_size)
        reward = self.unwrapped._get_rew()
        obs = self.unwrapped._get_obs()
        done = self.unwrapped._get_done()
        self.unwrapped.time += self.step_size
        self.unwrapped.total_reward += reward
        truncated = (
            self.unwrapped.time / self.step_size
        ) < self._max_episode_steps
        return obs, reward, done, truncated, {}
        """
        return self.unwrapped.step(action)
    
    def reset(self, *args, **kwargs):
        #print(f'[Info] Resetting environment...')
        result = super().reset(*args, **kwargs)
        #print(f'[Info] Reset done. Initial COM height: {self.unwrapped.model.com_pos().y:.3f}')
        # Ensure gymnasium format (obs, info)
        if isinstance(result, tuple):
            return result  # Already (obs, info)
        return result, {}  # Convert old format to gymnasium

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        if hasattr(self.unwrapped, 'seed'):
            return self.unwrapped.seed(seed)
        # Fallback for environments without seed method
        np.random.seed(seed)
        return [seed]

    @property
    def model(self):
        """Access to the underlying SCONE model."""
        return self.unwrapped.model

    @property
    def name(self):
        """Environment name."""
        if hasattr(self.unwrapped, 'name'):
            return self.unwrapped.name
        if hasattr(self.unwrapped, 'spec') and self.unwrapped.spec is not None:
            return self.unwrapped.spec.id
        return 'nair_gait_h0918-v0'

    @property
    def _max_episode_steps(self):
        return 1000

    @property
    def results_dir(self):
        return self.unwrapped.results_dir
