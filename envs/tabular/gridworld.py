"""
GridWorld Environment for Reinforcement Learning
=================================================

A configurable 2D grid environment implementing the MDP framework.
Supports both model-based (Dynamic Programming) and model-free (Q-learning, SARSA) methods.

Features:
- Configurable size, obstacles, traps, and goals
- Optional stochastic transitions (slip probability)
- Full MDP model exposure for DP methods
- Gymnasium-style step() interface for model-free methods
- Matplotlib visualization with value function overlays

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Dict, List, Optional, Set
from collections import deque
from enum import IntEnum

from envs.base import TabularEnvironment



class CellType(IntEnum):
    """Cell types in the grid."""
    EMPTY = 0
    WALL = 1
    TRAP = 2
    GOAL = 3
    START = 4  # Only for visualization, behaves like EMPTY


class Action(IntEnum):
    """Available actions (4-connected movement)."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Direction vectors for each action: (delta_row, delta_col)
ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

# Perpendicular actions for stochastic transitions
PERPENDICULAR = {
    Action.UP: [Action.LEFT, Action.RIGHT],
    Action.DOWN: [Action.LEFT, Action.RIGHT],
    Action.LEFT: [Action.UP, Action.DOWN],
    Action.RIGHT: [Action.UP, Action.DOWN],
}


class GridWorld(TabularEnvironment):
    """
    A 2D GridWorld environment for reinforcement learning.
    
    The environment is a rectangular grid where an agent navigates from
    a start position to a goal while avoiding walls and traps.
    
    State Space: Integer indices from 0 to (height * width - 1)
                 State s corresponds to position (s // width, s % width)
    
    Action Space: {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
    
    Rewards:
        - Reaching GOAL: +100 (terminal)
        - Falling into TRAP: -100 (terminal)
        - Each step: -1 (encourages efficiency)
        - Hitting a wall: -1 (stays in place)
    
    Parameters
    ----------
    height : int
        Number of rows in the grid (default: 10)
    width : int
        Number of columns in the grid (default: 10)
    wall_density : float
        Fraction of cells that are walls (default: 0.2)
    num_traps : int
        Number of trap cells (default: 3)
    slip_prob : float
        Probability of slipping perpendicular to intended direction (default: 0.0)
    seed : int or None
        Random seed for reproducibility (default: None)
    
    Attributes
    ----------
    grid : np.ndarray
        2D array of CellType values
    start_state : int
        Initial state index
    goal_state : int
        Goal state index
    trap_states : set
        Set of trap state indices
    
    Examples
    --------
    >>> env = GridWorld(height=5, width=5, seed=42)
    >>> state = env.reset()
    >>> next_state, reward, done, truncated, info = env.step(Action.RIGHT)
    >>> env.render()
    """
    
    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        wall_density: float = 0.2,
        num_traps: int = 3,
        slip_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        # Store configuration
        self.height = height
        self.width = width
        self.wall_density = wall_density
        self.num_traps = num_traps
        self.slip_prob = slip_prob
        self.seed = seed
        
        # Initialize random generator
        self.rng = np.random.default_rng(seed)
        
        # Generate the grid
        self._generate_grid()
        
        # Current agent position (state index)
        self.current_state = self.start_state
        
        # Episode tracking
        self._terminated = False
        self._step_count = 0
        self._max_steps = height * width  # Prevent infinite episodes
        
        # Cache transition model for DP methods
        self._transition_cache: Optional[Dict] = None
    
    # ==================== Grid Generation ====================
    
    def _generate_grid(self) -> None:
        """Generate a random grid with walls, traps, start, and goal."""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Initialize empty grid
            self.grid = np.zeros((self.height, self.width), dtype=np.int32)
            
            # Place start and goal with minimum distance constraint
            self.start_pos, self.goal_pos = self._place_start_and_goal()
            self.start_state = self._pos_to_state(self.start_pos)
            self.goal_state = self._pos_to_state(self.goal_pos)
            
            # Mark goal in grid
            self.grid[self.goal_pos] = CellType.GOAL
            
            # Place walls randomly
            self._place_walls()
            
            # Place traps
            self.trap_states = self._place_traps()
            
            # Mark start (for visualization only)
            self.grid[self.start_pos] = CellType.START
            
            # Verify path exists from start to goal
            if self._path_exists(self.start_pos, self.goal_pos):
                return
        
        raise RuntimeError(
            f"Failed to generate valid grid after {max_attempts} attempts. "
            "Try reducing wall_density or num_traps."
        )
    
    def _place_start_and_goal(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Place start and goal positions with minimum distance constraint."""
        min_distance = self.height // 2  # Must be > half the height apart
        
        for _ in range(1000):
            start_row = self.rng.integers(0, self.height)
            start_col = self.rng.integers(0, self.width)
            goal_row = self.rng.integers(0, self.height)
            goal_col = self.rng.integers(0, self.width)
            
            # Manhattan distance
            distance = abs(start_row - goal_row) + abs(start_col - goal_col)
            
            if distance > min_distance:
                return (start_row, start_col), (goal_row, goal_col)
        
        # Fallback: opposite corners
        return (0, 0), (self.height - 1, self.width - 1)
    
    def _place_walls(self) -> None:
        """Place wall cells randomly, avoiding start and goal."""
        num_walls = int(self.height * self.width * self.wall_density)
        
        available_cells = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if (r, c) != self.start_pos and (r, c) != self.goal_pos
        ]
        
        self.rng.shuffle(available_cells)
        
        for i in range(min(num_walls, len(available_cells))):
            r, c = available_cells[i]
            self.grid[r, c] = CellType.WALL
    
    def _place_traps(self) -> Set[int]:
        """Place trap cells, avoiding start, goal, and walls."""
        trap_states = set()
        
        available_cells = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if self.grid[r, c] == CellType.EMPTY
        ]
        
        self.rng.shuffle(available_cells)
        
        for i in range(min(self.num_traps, len(available_cells))):
            r, c = available_cells[i]
            self.grid[r, c] = CellType.TRAP
            trap_states.add(self._pos_to_state((r, c)))
        
        return trap_states
    
    def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Check if a path exists using BFS.""" 
        if start == goal:
            return True
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            r, c = queue.popleft()
            
            for action in Action:
                dr, dc = ACTION_DELTAS[action]
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < self.height and 0 <= nc < self.width and
                    (nr, nc) not in visited and
                    self.grid[nr, nc] != CellType.WALL):
                    
                    if (nr, nc) == goal:
                        return True
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    # ==================== State/Position Conversion ====================
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) position to state index."""
        return pos[0] * self.width + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        return (state // self.width, state % self.width)
    
    # ==================== Gymnasium-Style Interface ====================
    
    def reset(self, seed: Optional[int] = None) -> int:
        """
        Reset the environment to initial state.
        
        Parameters
        ----------
        seed : int or None
            Optional new seed (regenerates grid if provided)
        
        Returns
        -------
        state : int
            Initial state index
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self._generate_grid()
            self._transition_cache = None
        
        self.current_state = self.start_state
        self._terminated = False
        self._step_count = 0
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        
        Returns
        -------
        next_state : int
            New state index
        reward : float
            Reward received
        terminated : bool
            Whether episode ended (goal/trap reached)
        truncated : bool
            Whether episode was truncated (max steps)
        info : dict
            Additional information
        """
        if self._terminated:
            raise RuntimeError("Episode has terminated. Call reset() first.")
        
        action = Action(action)
        
        # Determine actual movement (with possible slip)
        actual_action = self._get_actual_action(action)
        
        # Compute next state
        current_pos = self._state_to_pos(self.current_state)
        next_pos = self._get_next_position(current_pos, actual_action)
        next_state = self._pos_to_state(next_pos)
        
        # Compute reward
        reward = self._get_reward(self.current_state, action, next_state)
        
        # Check termination
        cell_type = self.grid[next_pos]
        terminated = cell_type in (CellType.GOAL, CellType.TRAP)
        
        # Update state
        self.current_state = next_state
        self._step_count += 1
        
        # Check truncation
        truncated = self._step_count >= self._max_steps
        
        if terminated or truncated:
            self._terminated = True
        
        info = {
            'position': next_pos,
            'cell_type': int(cell_type),
            'intended_action': int(action),
            'actual_action': int(actual_action),
            'slipped': actual_action != action,
            'step_count': self._step_count,
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_actual_action(self, intended_action: Action) -> Action:
        """Determine actual action after possible slip."""
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            perpendicular = PERPENDICULAR[intended_action]
            return self.rng.choice(perpendicular)
        return intended_action
    
    def _get_next_position(
        self, 
        pos: Tuple[int, int], 
        action: Action
    ) -> Tuple[int, int]:
        """Get next position after taking action (handles walls and boundaries)."""
        dr, dc = ACTION_DELTAS[action]
        new_r, new_c = pos[0] + dr, pos[1] + dc
        
        # Check boundaries
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            return pos  # Stay in place
        
        # Check walls
        if self.grid[new_r, new_c] == CellType.WALL:
            return pos  # Stay in place
        
        return (new_r, new_c)
    
    def _get_reward(self, state: int, action: Action, next_state: int) -> float:
        """Compute reward for a transition."""
        next_pos = self._state_to_pos(next_state)
        cell_type = self.grid[next_pos]
        
        if cell_type == CellType.GOAL:
            return 100.0
        elif cell_type == CellType.TRAP:
            return -100.0
        else:
            return -1.0  # Step cost
    
    # ==================== Model-Based Interface (for DP) ====================
    
    @property
    def states(self) -> List[int]:
        """Return list of all non-wall state indices."""
        return [
            self._pos_to_state((r, c))
            for r in range(self.height)
            for c in range(self.width)
            if self.grid[r, c] != CellType.WALL
        ]
    
    @property
    def actions(self) -> List[int]:
        return [int(a) for a in Action]  # [0,1,2,3]

    
    @property
    def terminal_states(self) -> Set[int]:
        """Return set of terminal state indices."""
        return self.trap_states | {self.goal_state}
    
    @property
    def n_states(self) -> int:
        """
        Number of states in the environment.

        Notes
        -----
        GridWorld uses global state indexing: state = row * width + col,
        so valid state ids are in [0, height*width - 1] even though `states`
        excludes walls.
        """
        return self.height * self.width

    
    def is_terminal(self, state: int) -> bool:
        """Check if state is terminal."""
        return state in self.terminal_states
    
    def P(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s' | s, a).
        
        Parameters
        ----------
        state : int
            Current state index
        action : int
            Action to take
        
        Returns
        -------
        transitions : dict
            Dictionary mapping next_state -> probability
        """
        if self.is_terminal(state):
            return {state: 1.0}  # Terminal states are absorbing
        
        action = Action(action)
        pos = self._state_to_pos(state)
        
        transitions = {}
        
        if self.slip_prob == 0:
            # Deterministic case
            next_pos = self._get_next_position(pos, action)
            next_state = self._pos_to_state(next_pos)
            transitions[next_state] = 1.0
        else:
            # Stochastic case
            # Intended direction
            intended_pos = self._get_next_position(pos, action)
            intended_state = self._pos_to_state(intended_pos)
            
            prob_intended = 1.0 - self.slip_prob
            transitions[intended_state] = transitions.get(intended_state, 0) + prob_intended
            
            # Perpendicular slips
            for perp_action in PERPENDICULAR[action]:
                perp_pos = self._get_next_position(pos, perp_action)
                perp_state = self._pos_to_state(perp_pos)
                prob_perp = self.slip_prob / 2
                transitions[perp_state] = transitions.get(perp_state, 0) + prob_perp
        
        return transitions
    
    def R(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward R(s, a, s').
        
        Parameters
        ----------
        state : int
            Current state index
        action : int
            Action taken
        next_state : int
            Next state index
        
        Returns
        -------
        reward : float
            Reward value
        """
        return self._get_reward(state, Action(action), next_state)
    
    def get_full_transition_model(self) -> Dict:
        """
        Get the complete transition model for all state-action pairs.
        
        Returns
        -------
        model : dict
            Nested dict: model[s][a] = [(prob, next_s, reward, done), ...]
        """
        if self._transition_cache is not None:
            return self._transition_cache
        
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
        
        self._transition_cache = model
        return model
    
    # ==================== Visualization ====================
    
    def render(
        self,
        values: Optional[np.ndarray] = None,
        policy: Optional[np.ndarray] = None,
        title: str = "GridWorld",
        show_state_indices: bool = False,
        figsize: Tuple[int, int] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Render the grid with optional value function and policy overlays.
        
        Parameters
        ----------
        values : np.ndarray or None
            Value function V(s) for each state (shape: n_states,)
        policy : np.ndarray or None
            Policy π(s) giving action for each state (shape: n_states,)
        title : str
            Plot title
        show_state_indices : bool
            Whether to show state index numbers
        figsize : tuple or None
            Figure size (width, height)
        ax : plt.Axes or None
            Existing axes to draw on
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (max(8, self.width), max(6, self.height))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create base grid visualization
        grid_visual = np.zeros((self.height, self.width, 3))
        
        # Color scheme
        colors = {
            CellType.EMPTY: [1.0, 1.0, 1.0],      # White
            CellType.WALL: [0.2, 0.2, 0.2],       # Dark gray
            CellType.TRAP: [0.9, 0.3, 0.3],       # Red
            CellType.GOAL: [0.3, 0.9, 0.3],       # Green
            CellType.START: [0.3, 0.5, 0.9],      # Blue
        }
        
        for r in range(self.height):
            for c in range(self.width):
                cell_type = self.grid[r, c]
                grid_visual[r, c] = colors[cell_type]
        
        # If values provided, overlay as heatmap on empty cells
        if values is not None:
            # Normalize values for color mapping
            valid_values = []
            for s in self.states:
                if not self.is_terminal(s):
                    pos = self._state_to_pos(s)
                    if self.grid[pos] not in (CellType.WALL,):
                        valid_values.append(values[s])
            
            if valid_values:
                v_min, v_max = min(valid_values), max(valid_values)
                v_range = v_max - v_min if v_max > v_min else 1
                
                for s in self.states:
                    if not self.is_terminal(s):
                        pos = self._state_to_pos(s)
                        if self.grid[pos] == CellType.EMPTY:
                            # Blue (low) to Yellow (high)
                            normalized = (values[s] - v_min) / v_range
                            grid_visual[pos] = [
                                0.2 + 0.8 * normalized,  # R
                                0.2 + 0.6 * normalized,  # G
                                0.8 - 0.6 * normalized   # B
                            ]
        
        # Draw grid
        ax.imshow(grid_visual, origin='upper', aspect='equal')
        
        # Draw grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # Draw agent position
        agent_pos = self._state_to_pos(self.current_state)
        circle = plt.Circle(
            (agent_pos[1], agent_pos[0]), 0.3,
            color='orange', ec='black', linewidth=2, zorder=10
        )
        ax.add_patch(circle)
        
        # Draw policy arrows if provided
        if policy is not None:
            arrow_dx = {Action.UP: 0, Action.DOWN: 0, Action.LEFT: -0.3, Action.RIGHT: 0.3}
            arrow_dy = {Action.UP: -0.3, Action.DOWN: 0.3, Action.LEFT: 0, Action.RIGHT: 0}
            
            for s in self.states:
                if not self.is_terminal(s):
                    pos = self._state_to_pos(s)
                    if self.grid[pos] != CellType.WALL:
                        a = Action(policy[s])
                        ax.arrow(
                            pos[1], pos[0],
                            arrow_dx[a], arrow_dy[a],
                            head_width=0.15, head_length=0.1,
                            fc='black', ec='black', zorder=5
                        )
        
        # Show state indices if requested
        if show_state_indices:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid[r, c] != CellType.WALL:
                        s = self._pos_to_state((r, c))
                        ax.text(c, r, str(s), ha='center', va='center',
                               fontsize=6, color='gray', alpha=0.7)
        
        # Show values if provided
        if values is not None:
            for s in self.states:
                pos = self._state_to_pos(s)
                if self.grid[pos] != CellType.WALL:
                    ax.text(pos[1], pos[0] + 0.3, f'{values[s]:.1f}',
                           ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Legend
        legend_patches = [
            mpatches.Patch(color=colors[CellType.START], label='Start'),
            mpatches.Patch(color=colors[CellType.GOAL], label='Goal (+100)'),
            mpatches.Patch(color=colors[CellType.TRAP], label='Trap (-100)'),
            mpatches.Patch(color=colors[CellType.WALL], label='Wall'),
            mpatches.Patch(color='orange', label='Agent'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
    
    def render_ascii(self) -> str:
        """
        Render the grid as ASCII art.
        
        Returns
        -------
        ascii_grid : str
            ASCII representation of the grid
        """
        symbols = {
            CellType.EMPTY: '.',
            CellType.WALL: '#',
            CellType.TRAP: 'X',
            CellType.GOAL: 'G',
            CellType.START: 'S',
        }
        
        agent_pos = self._state_to_pos(self.current_state)
        
        lines = []
        lines.append('┌' + '─' * (self.width * 2 + 1) + '┐')
        
        for r in range(self.height):
            row_str = '│ '
            for c in range(self.width):
                if (r, c) == agent_pos:
                    row_str += 'A '
                else:
                    row_str += symbols[self.grid[r, c]] + ' '
            row_str += '│'
            lines.append(row_str)
        
        lines.append('└' + '─' * (self.width * 2 + 1) + '┘')
        lines.append(f'Agent at state {self.current_state} (row={agent_pos[0]}, col={agent_pos[1]})')
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return (
            f"GridWorld(height={self.height}, width={self.width}, "
            f"wall_density={self.wall_density}, num_traps={self.num_traps}, "
            f"slip_prob={self.slip_prob}, seed={self.seed})"
        )


# ==================== Demo and Testing ====================

def demo():
    """Demonstrate the GridWorld environment."""
    print("=" * 60)
    print("GridWorld Environment Demo")
    print("=" * 60)
    
    # Create environment
    env = GridWorld(
        height=25,
        width=25,
        wall_density=0.15,
        num_traps=30,
        slip_prob=0.2,
        seed=42
    )
    
    print(f"\nEnvironment: {env}")
    print(f"Number of states: {env.n_states}")
    print(f"Number of non-wall states: {len(env.states)}")
    print(f"Start state: {env.start_state} at position {env._state_to_pos(env.start_state)}")
    print(f"Goal state: {env.goal_state} at position {env._state_to_pos(env.goal_state)}")
    print(f"Trap states: {env.trap_states}")
    print(f"Terminal states: {env.terminal_states}")
    
    # ASCII visualization
    print("\n" + env.render_ascii())
    
    # Run a few random steps
    print("\n--- Running random episode ---")
    state = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = env.rng.integers(0, 4)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"Step {step+1}: Action={action_names[action]}, "
              f"State {state}->{next_state}, Reward={reward:.0f}"
              + (" [SLIPPED]" if info['slipped'] else ""))
        
        state = next_state
        
        if terminated:
            print(f"Episode terminated! Total reward: {total_reward:.0f}")
            break
        if truncated:
            print(f"Episode truncated! Total reward: {total_reward:.0f}")
            break
    
    # Show transition model example
    print("\n--- Transition Model Example ---")
    test_state = env.start_state
    print(f"From state {test_state}:")
    for a in env.actions:
        trans = env.P(test_state, a)
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"  Action {action_names[a]}: {trans}")
    
    # Matplotlib visualization
    print("\n--- Creating visualization ---")
    env.reset()
    
    # Create dummy values for visualization demo
    dummy_values = np.zeros(env.n_states)
    for s in env.states:
        pos = env._state_to_pos(s)
        # Distance-based dummy values
        goal_pos = env._state_to_pos(env.goal_state)
        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
        dummy_values[s] = -dist
    
    dummy_values[env.goal_state] = 100
    for trap in env.trap_states:
        dummy_values[trap] = -100
    
    fig = env.render(
        values=dummy_values,
        title="GridWorld with Distance-Based Values (Demo)",
        show_state_indices=False
    )
    
    # Save figure
    fig.savefig('./outputs/gridworld/gridworld_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to ./outputs/gridworld/gridworld_demo.png")
    
    plt.close(fig)
    
    return env


if __name__ == "__main__":
    demo()