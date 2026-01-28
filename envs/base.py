"""
Abstract Base Classes for Reinforcement Learning Environments
==============================================================

This module defines the interface contracts that all environments must satisfy.

Class Hierarchy:
    BaseEnvironment (ABC)
        └── TabularEnvironment (ABC)
                └── GridWorld, FrozenLake, etc.

    BaseEnvironment
        └── ContinuousEnvironment (future)
                └── RocketLander, PointMass, etc.

Usage in algorithms:
    from environments.base import BaseEnvironment, TabularEnvironment
    
    def q_learning(env: BaseEnvironment, ...):
        # Works with ANY environment (model-free)
        state = env.reset()
        next_state, reward, done, _, _ = env.step(action)
    
    def value_iteration(env: TabularEnvironment, ...):
        # Requires model access (model-based)
        for s in env.states:
            for a in env.actions:
                transitions = env.P(s, a)

"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union
)
import numpy as np


# Type variables for generic state and action types
StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')

# Common type aliases
StepResult = Tuple[Any, float, bool, bool, dict]  # (next_state, reward, terminated, truncated, info)


class BaseEnvironment(ABC, Generic[StateType, ActionType]):
    """
    Abstract base class for all reinforcement learning environments.
    
    This defines the minimal interface that ANY environment must implement,
    whether tabular, continuous, or hybrid. It follows the Gymnasium convention
    for the step() interface.
    
    Type Parameters
    ---------------
    StateType : type
        The type of state observations (int for tabular, np.ndarray for continuous)
    ActionType : type
        The type of actions (int for discrete, np.ndarray for continuous)
    
    Abstract Methods (must be implemented)
    --------------------------------------
    reset() -> state
        Reset environment to initial state
    step(action) -> (next_state, reward, terminated, truncated, info)
        Execute one environment step
    
    Abstract Properties (must be implemented)
    -----------------------------------------
    n_actions : int or float('inf')
        Number of actions (inf for continuous)
    action_space : Any
        Description of valid actions
    
    Optional Methods (have default implementations)
    -----------------------------------------------
    render() -> Any
        Visualize the environment
    close() -> None
        Clean up resources
    seed(seed) -> None
        Set random seed
    
    Example
    -------
    >>> class MyEnv(BaseEnvironment[int, int]):
    ...     def reset(self): return 0
    ...     def step(self, a): return 0, 0.0, False, False, {}
    ...     @property
    ...     def n_actions(self): return 4
    ...     @property
    ...     def action_space(self): return [0, 1, 2, 3]
    """
    
    # ==================== Core Interface (MUST implement) ====================
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> StateType:
        """
        Reset the environment to an initial state.
        
        Parameters
        ----------
        seed : int or None
            Optional random seed for reproducibility
        
        Returns
        -------
        state : StateType
            The initial state observation
        
        Notes
        -----
        - Should reset any internal episode state (step counter, etc.)
        - If seed is provided, should reinitialize randomness
        """
        pass
    
    @abstractmethod
    def step(self, action: ActionType) -> StepResult:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : ActionType
            The action to execute
        
        Returns
        -------
        next_state : StateType
            The resulting state observation
        reward : float
            The reward received for this transition
        terminated : bool
            True if the episode ended due to environment rules
            (e.g., reached goal, fell into trap, game over)
        truncated : bool
            True if the episode ended due to external limits
            (e.g., max steps reached, time limit)
        info : dict
            Additional diagnostic information (not used for learning)
        
        Raises
        ------
        RuntimeError
            If called after episode termination without reset()
        
        Notes
        -----
        The distinction between terminated and truncated matters for
        bootstrapping: we should bootstrap from truncated states but
        not from terminated states.
        """
        pass
    
    @property
    @abstractmethod
    def n_actions(self) -> Union[int, float]:
        """
        Number of possible actions.
        
        Returns
        -------
        n : int or float
            Number of discrete actions, or float('inf') for continuous
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Description of the action space.
        
        Returns
        -------
        space : Any
            For discrete: list of valid actions
            For continuous: dict with 'low', 'high', 'shape' keys
        """
        pass
    
    # ==================== Optional Interface (CAN override) ====================
    
    @property
    def n_states(self) -> Union[int, float]:
        """
        Number of possible states.
        
        Returns
        -------
        n : int or float
            Number of discrete states, or float('inf') for continuous
        
        Notes
        -----
        Default returns inf. Tabular environments should override this.
        """
        return float('inf')
    
    @property
    def observation_space(self) -> Any:
        """
        Description of the observation/state space.
        
        Returns
        -------
        space : Any
            For discrete: list of valid states or int count
            For continuous: dict with 'low', 'high', 'shape' keys
        """
        return None
    
    @property
    def gamma(self) -> float:
        """
        Recommended discount factor for this environment.
        
        Returns
        -------
        gamma : float
            Discount factor in [0, 1)
        """
        return 0.99
    
    def render(self, **kwargs) -> Any:
        """
        Render the environment for visualization.
        
        Parameters
        ----------
        **kwargs : dict
            Environment-specific rendering options
        
        Returns
        -------
        output : Any
            Rendered output (figure, string, etc.)
        """
        raise NotImplementedError("This environment does not support rendering")
    
    def close(self) -> None:
        """
        Clean up any resources (display windows, file handles, etc.).
        """
        pass
    
    def seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Parameters
        ----------
        seed : int
            Random seed
        
        Notes
        -----
        Prefer using reset(seed=...) instead for full reproducibility.
        """
        pass
    
    # ==================== Utility Methods ====================
    
    def sample_action(self) -> ActionType:
        """
        Sample a random action from the action space.
        
        Returns
        -------
        action : ActionType
            A randomly sampled valid action
        """
        if isinstance(self.action_space, (list, range)):
            return np.random.choice(self.action_space)
        raise NotImplementedError("Override sample_action() for continuous spaces")
    
    def is_valid_action(self, action: ActionType) -> bool:
        """
        Check if an action is valid.
        
        Parameters
        ----------
        action : ActionType
            Action to validate
        
        Returns
        -------
        valid : bool
            True if action is in the action space
        """
        if isinstance(self.action_space, (list, range)):
            return action in self.action_space
        return True  # Assume valid for continuous; override if needed


class TabularEnvironment(BaseEnvironment[int, int]):
    """
    Abstract base class for tabular (finite state-action) environments.
    
    Extends BaseEnvironment with the model-based interface required for
    dynamic programming algorithms (Value Iteration, Policy Iteration).
    
    In a tabular environment:
    - States are integers from 0 to n_states - 1
    - Actions are integers from 0 to n_actions - 1
    - The transition model P(s'|s,a) and reward R(s,a,s') are accessible
    
    Abstract Methods (in addition to BaseEnvironment)
    --------------------------------------------------
    states : List[int]
        List of all valid (non-wall) state indices
    actions : List[int]
        List of all action indices
    P(s, a) -> Dict[s', prob]
        Transition probability distribution
    R(s, a, s') -> float
        Reward function
    is_terminal(s) -> bool
        Check if state is terminal
    
    Example
    -------
    >>> class MyTabularEnv(TabularEnvironment):
    ...     # Implement all abstract methods
    ...     pass
    >>> 
    >>> # Now usable with DP algorithms:
    >>> for s in env.states:
    ...     for a in env.actions:
    ...         for s_next, prob in env.P(s, a).items():
    ...             r = env.R(s, a, s_next)
    """
    
    # ==================== Model-Based Interface (MUST implement) ====================
    
    @property
    @abstractmethod
    def states(self) -> List[int]:
        """
        List of all valid state indices.
        
        Returns
        -------
        states : List[int]
            All state indices that the agent can occupy
            (excludes walls or invalid states)
        
        Notes
        -----
        For a grid with walls, this returns only traversable cells.
        """
        pass
    
    @property
    @abstractmethod
    def actions(self) -> List[int]:
        """
        List of all action indices.
        
        Returns
        -------
        actions : List[int]
            All valid action indices [0, 1, ..., n_actions-1]
        """
        pass
    
    @abstractmethod
    def P(self, state: int, action: int) -> Dict[int, float]:
        """
        Transition probability distribution P(s' | s, a).
        
        Parameters
        ----------
        state : int
            Current state index
        action : int
            Action to take
        
        Returns
        -------
        transitions : Dict[int, float]
            Dictionary mapping next_state -> probability
            Probabilities must sum to 1.0
        
        Notes
        -----
        - Terminal states should return {state: 1.0} (absorbing)
        - For deterministic dynamics, returns {next_state: 1.0}
        """
        pass
    
    @abstractmethod
    def R(self, state: int, action: int, next_state: int) -> float:
        """
        Reward function R(s, a, s').
        
        Parameters
        ----------
        state : int
            Current state index
        action : int
            Action taken
        next_state : int
            Resulting state index
        
        Returns
        -------
        reward : float
            Immediate reward for this transition
        
        Notes
        -----
        Some environments use R(s,a) instead of R(s,a,s').
        In that case, the next_state parameter can be ignored.
        """
        pass
    
    @abstractmethod
    def is_terminal(self, state: int) -> bool:
        """
        Check if a state is terminal.
        
        Parameters
        ----------
        state : int
            State index to check
        
        Returns
        -------
        terminal : bool
            True if state is terminal (goal, trap, game over)
        
        Notes
        -----
        Terminal states end the episode. The value of a terminal
        state under any policy is 0 (no future rewards).
        """
        pass
    
    # ==================== Derived Properties (CAN override for efficiency) ====================
    
    @property
    def terminal_states(self) -> Set[int]:
        """
        Set of all terminal state indices.
        
        Returns
        -------
        terminals : Set[int]
            All states where is_terminal(s) is True
        """
        return {s for s in self.states if self.is_terminal(s)}
    
    @property
    def non_terminal_states(self) -> List[int]:
        """
        List of all non-terminal state indices.
        
        Returns
        -------
        non_terminals : List[int]
            All states where is_terminal(s) is False
        """
        return [s for s in self.states if not self.is_terminal(s)]
    
    @property
    def n_states(self) -> int:
        """Number of valid states."""
        return len(self.states)
    
    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return len(self.actions)
    
    @property
    def action_space(self) -> List[int]:
        """Action space (same as actions for tabular)."""
        return self.actions
    
    # ==================== Convenience Methods ====================
    
    def get_full_transition_model(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        Get the complete MDP transition model.
        
        Returns
        -------
        model : Dict
            Nested dictionary: model[s][a] = [(prob, next_s, reward, done), ...]
            
        Notes
        -----
        This format matches what many DP implementations expect.
        Caching is recommended for large environments.
        """
        model = {}
        for s in self.states:
            model[s] = {}
            for a in self.actions:
                transitions = []
                for next_s, prob in self.P(s, a).items():
                    reward = self.R(s, a, next_s)
                    done = self.is_terminal(next_s)
                    transitions.append((prob, next_s, reward, done))
                model[s][a] = transitions
        return model
    
    def expected_reward(self, state: int, action: int) -> float:
        """
        Expected immediate reward E[R | s, a].
        
        Parameters
        ----------
        state : int
            Current state
        action : int
            Action to take
        
        Returns
        -------
        expected_r : float
            Expected reward over transition distribution
        """
        expected_r = 0.0
        for next_s, prob in self.P(state, action).items():
            expected_r += prob * self.R(state, action, next_s)
        return expected_r
    
    def validate_model(self) -> bool:
        """
        Validate that the MDP model is well-formed.
        
        Returns
        -------
        valid : bool
            True if all transition probabilities sum to 1
        
        Raises
        ------
        ValueError
            If any transition distribution doesn't sum to 1
        """
        for s in self.states:
            for a in self.actions:
                probs = self.P(s, a)
                total = sum(probs.values())
                if not np.isclose(total, 1.0):
                    raise ValueError(
                        f"P(·|s={s}, a={a}) sums to {total}, not 1.0"
                    )
        return True

# For type hints in algorithm implementations
TabularEnv = TabularEnvironment
Env = BaseEnvironment


def is_tabular(env: BaseEnvironment) -> bool:
    """Check if an environment is tabular (has model access)."""
    return isinstance(env, TabularEnvironment)


def is_continuous(env: BaseEnvironment) -> bool:
    """Check if an environment has continuous state/action spaces."""
    return env.n_states == float('inf') or env.n_actions == float('inf')