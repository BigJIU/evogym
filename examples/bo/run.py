from distutils.command.config import config
import os
from re import X
import shutil
import random
import numpy as np

from GPyOpt.core.task.space import Design_space
from GPyOpt.models import GPModel
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.core.evaluators import ThompsonBatch
from .optimizer import Objective, Optimization

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

import evogym.envs
from evogym import is_connected, has_actuator, get_full_connectivity
sys.path.append('./utils')
from algo_utils import TerminationCondition
from ppo import run_ppo

def get_robot_from_genome(genome, config):
    '''
    genome is a 1d vector
    robot is a 2d matrix
    '''
    structure_shape = config['structure_shape']
    robot = genome.reshape(structure_shape)
    return robot

def eval_genome_cost(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    if not (is_connected(robot) and has_actuator(robot)):
        return 10
    else:
        connectivity = get_full_connectivity(robot)
        save_path_generation = os.path.join(config['save_path'], f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        save_path_controller = os.path.join(save_path_generation, 'controller')
        np.savez(save_path_structure, robot, connectivity)
        fitness = run_ppo(
            structure=(robot, connectivity),
            termination_condition=TerminationCondition(config['train_iters']),
            saving_convention=(save_path_controller, genome_id),
        )
        cost = -fitness
        return cost

def eval_genome_constraint(genomes, config):
    all_violation = []
    for genome in genomes:
        robot = get_robot_from_genome(genome, config)
        violation = not (is_connected(robot) and has_actuator(robot))
        all_violation.append(violation)
    return np.array(all_violation)

def run_bo(
        experiment_name,
        structure_shape,
        pop_size,
        max_evaluations,
        train_iters,
        num_cores,
    ):

    save_path = os.path.join(root_dir, 'saved_data', experiment_name)

    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n' \
            f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n' \
            f'MAX_EVALUATIONS: {max_evaluations}\n' \
            f'TRAIN_ITERS: {train_iters}\n')

    config = {
        'structure_shape': structure_shape,
        'train_iters': train_iters,
        'save_path': save_path,
    }
    
    def constraint_func(genome): 
        return eval_genome_constraint(genome, config)

    def before_evaluate(generation):
        save_path = config['save_path']
        save_path_structure = os.path.join(save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def after_evaluate(generation, population_cost):
        save_path = config['save_path']
        save_path_ranking = os.path.join(save_path, f'generation_{generation}', 'output.txt')
        genome_fitness_list = -population_cost
        genome_id_list = np.argsort(population_cost)
        genome_fitness_list = np.array(genome_fitness_list)[genome_id_list]
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome_fitness in zip(genome_id_list, genome_fitness_list):
                out += f'{genome_id}\t\t{genome_fitness}\n'
            f.write(out)

    space = Design_space(
        space=[{'name': 'x', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4), 'dimensionality': np.prod(structure_shape)}], 
        constraints=[{'name': 'const', 'constraint': constraint_func}]
    )

    objective = Objective(eval_genome_cost, config, num_cores=num_cores)

    model = GPModel()

    acquisition = AcquisitionEI(
        model, 
        space, 
        optimizer=AcquisitionOptimizer(space)
    )

    evaluator = ThompsonBatch(acquisition, batch_size=pop_size)
    X_init = initial_design('random', space, pop_size)

    bo = Optimization(model, space, objective, acquisition, evaluator, X_init, de_duplication=True)
    bo.run_optimization(
        max_iter=np.ceil(max_evaluations / pop_size) - 1,
        verbosity=True,
        before_evaluate=before_evaluate,
        after_evaluate=after_evaluate
    )
    best_robot, best_fitness = bo.x_opt, -bo.fx_opt
    return best_robot, best_fitness
