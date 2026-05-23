"""
Evolutionary RL using EvoTorch CMA-ES.
Each week, evaluates population of PPO policies by their paper-trade Sharpe ratio,
then evolves the population to find better policies.
"""

import logging
import os
import copy
import json
from datetime import datetime, timedelta

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

logger = logging.getLogger(__name__)

POPULATION_DIR = "models/population"
BEST_MODEL_PATH = "models/best_policy"
FITNESS_LOG_PATH = "logs/fitness_history.json"


def get_policy_params(model: PPO) -> np.ndarray:
    """Flatten all policy network parameters into a 1D numpy array."""
    params = []
    for param in model.policy.parameters():
        params.append(param.data.cpu().numpy().flatten())
    return np.concatenate(params)


def set_policy_params(model: PPO, params: np.ndarray) -> PPO:
    """Load a flat parameter vector back into a PPO model's policy."""
    idx = 0
    for param in model.policy.parameters():
        size = param.numel()
        param.data = torch.tensor(
            params[idx: idx + size].reshape(param.shape), dtype=param.dtype
        )
        idx += size
    return model


def load_or_create_population(base_model_path: str, pop_size: int = 20) -> list[np.ndarray]:
    """Load existing population or create by perturbing the base model."""
    os.makedirs(POPULATION_DIR, exist_ok=True)
    pop_file = os.path.join(POPULATION_DIR, "population.npy")

    if os.path.exists(pop_file):
        logger.info("Loading existing population from %s", pop_file)
        population = list(np.load(pop_file, allow_pickle=True))
        if len(population) == pop_size:
            return population

    logger.info("Creating new population of %d from base model %s", pop_size, base_model_path)
    base_model = PPO.load(base_model_path)
    base_params = get_policy_params(base_model)
    param_dim = len(base_params)

    population = [base_params.copy()]
    for _ in range(pop_size - 1):
        noise = np.random.randn(param_dim) * 0.02
        population.append(base_params + noise)

    return population


def evaluate_policy_on_data(
    policy_params: np.ndarray,
    df,
    base_model_path: str = "models/baseline_ppo",
) -> float:
    """Run policy on env for full df, return Sharpe ratio as fitness."""
    from src.rl.env import XAUUSDTradingEnv

    model = PPO.load(base_model_path)
    model = set_policy_params(model, policy_params)

    env = XAUUSDTradingEnv(df)
    obs, _ = env.reset()
    portfolio_values = [env.initial_cash]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    # Compute Sharpe from daily returns
    values = np.array(portfolio_values)
    if len(values) < 2:
        return -999.0

    returns = np.diff(values) / values[:-1]
    if returns.std() < 1e-8:
        return 0.0

    # Annualize: hourly data → ~6000 hours/year
    sharpe = (returns.mean() / returns.std()) * np.sqrt(6000)
    return float(np.clip(sharpe, -10.0, 10.0))


def run_evolution(
    population: list[np.ndarray],
    fitness_scores: list[float],
    stdev: float = 0.02,
    elite_frac: float = 0.2,
    eval_df=None,
    base_model_path: str = "models/baseline_ppo",
) -> list[np.ndarray]:
    """
    CMA-ES evolution step that actually evaluates each candidate on the env.
    Falls back to simple GA if EvoTorch isn't installed or eval_df missing.
    """
    if eval_df is None:
        logger.warning("No eval_df provided to run_evolution → using simple GA")
        return _simple_ga_evolution(population, fitness_scores, stdev, elite_frac)
    try:
        return _evotorch_evolution(population, fitness_scores, stdev, eval_df, base_model_path)
    except ImportError:
        logger.warning("EvoTorch not available, using simple GA")
        return _simple_ga_evolution(population, fitness_scores, stdev, elite_frac)


def _evotorch_evolution(
    population: list[np.ndarray],
    fitness_scores: list[float],
    stdev: float,
    eval_df,
    base_model_path: str,
) -> list[np.ndarray]:
    """EvoTorch CMA-ES — each candidate's fitness is the Sharpe ratio on eval_df."""
    from evotorch import Problem
    from evotorch.algorithms import CMAES

    pop_size = len(population)
    param_dim = len(population[0])
    best_idx = int(np.argmax(fitness_scores))
    seed_params = population[best_idx]

    class LivePolicyProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="max",
                solution_length=param_dim,
                dtype=torch.float32,
                initial_bounds=(-2.0, 2.0),
            )

        def _evaluate(self, solution):
            params = solution.values.detach().cpu().numpy()
            fitness = evaluate_policy_on_data(params, eval_df, base_model_path)
            solution.set_evals(float(fitness))

    searcher = CMAES(
        LivePolicyProblem(),
        stdev_init=stdev,
        popsize=pop_size,
        center_init=torch.tensor(seed_params, dtype=torch.float32),
    )
    searcher.run(1)

    new_population = [sol.values.detach().cpu().numpy().copy() for sol in searcher.population]
    logger.info("EvoTorch CMA-ES: %d candidates evaluated (seed fitness: %.3f)",
                len(new_population), fitness_scores[best_idx])
    return new_population


def _simple_ga_evolution(
    population: list[np.ndarray],
    fitness_scores: list[float],
    stdev: float,
    elite_frac: float,
) -> list[np.ndarray]:
    """Fallback: simple genetic algorithm with elitism + mutation."""
    pop_size = len(population)
    n_elite = max(1, int(pop_size * elite_frac))

    ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
    elites = [p.copy() for _, p in ranked[:n_elite]]
    elite_mean = np.mean(elites, axis=0)

    new_population = list(elites)  # keep elites
    while len(new_population) < pop_size:
        parent = elites[np.random.randint(len(elites))].copy()
        noise = np.random.randn(len(parent)) * stdev
        child = parent + noise
        new_population.append(child)

    logger.info("GA evolution: kept %d elites, best fitness: %.3f", n_elite, fitness_scores[0])
    return new_population[:pop_size]


class EvolutionManager:
    """Manages the weekly evolutionary RL cycle."""

    def __init__(self, base_model_path: str = "models/baseline_ppo", pop_size: int = 20):
        self.base_model_path = base_model_path
        self.pop_size = pop_size
        self.population: list[np.ndarray] = []
        self.fitness_history: list[dict] = []

    def load_state(self):
        self.population = load_or_create_population(self.base_model_path, self.pop_size)
        if os.path.exists(FITNESS_LOG_PATH):
            with open(FITNESS_LOG_PATH) as f:
                self.fitness_history = json.load(f)

    def save_state(self):
        os.makedirs(POPULATION_DIR, exist_ok=True)
        np.save(os.path.join(POPULATION_DIR, "population.npy"), np.array(self.population, dtype=object))
        with open(FITNESS_LOG_PATH, "w") as f:
            json.dump(self.fitness_history, f, indent=2)

    def run_weekly_cycle(self, eval_df) -> dict:
        """
        Full weekly evolution cycle:
        1. Evaluate all policies on recent paper trade data
        2. Evolve population
        3. Deploy best policy
        """
        if not self.population:
            self.load_state()

        logger.info("Evaluating %d policies on %d bars...", len(self.population), len(eval_df))
        fitness_scores = []
        for i, params in enumerate(self.population):
            score = evaluate_policy_on_data(params, eval_df, self.base_model_path)
            fitness_scores.append(score)
            logger.info("  Policy %02d: Sharpe = %.3f", i, score)

        best_idx = int(np.argmax(fitness_scores))
        best_fitness = fitness_scores[best_idx]
        logger.info("Best policy: #%d, Sharpe = %.3f", best_idx, best_fitness)

        # Deploy best policy
        best_model = PPO.load(self.base_model_path)
        best_model = set_policy_params(best_model, self.population[best_idx])
        best_model.save(BEST_MODEL_PATH)
        logger.info("Deployed best policy to %s", BEST_MODEL_PATH)

        # Evolve — pass eval_df so CMA-ES re-evaluates each candidate properly
        self.population = run_evolution(
            self.population, fitness_scores,
            eval_df=eval_df, base_model_path=self.base_model_path,
        )

        # Log fitness
        cycle_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "best_sharpe": best_fitness,
            "mean_sharpe": float(np.mean(fitness_scores)),
            "worst_sharpe": float(np.min(fitness_scores)),
            "generation": len(self.fitness_history) + 1,
        }
        self.fitness_history.append(cycle_log)
        self.save_state()

        return cycle_log
