from string import Template
import os

from gridsearcher import GridSearcher

ROOT_PROJECT = '' # path to the project folder
ROOT_DATASETS = '' # path to datasets
ROOT_RUNS = '' # path to save artifacts

def main(dataset, model, optimizer, gpus, params):
    gs = GridSearcher(script=os.path.join(ROOT_PROJECT, 'main.py'))
    gs.add_param('micro_adam_num_grads', '10')
    gs.add_param('micro_adam_density', '0.01')
    gs.add_param('lr_warmup_steps', '0')
    gs.add_param('lr_sched', 'const')

    gs.add_param('dataset_path', os.path.join(ROOT_DATASETS, dataset))
    gs.add_param('dataset_name', dataset)
    gs.add_param('model_name', model)
    gs.add_param('optimizer', optimizer)

    gs.add_param('wandb_enable', '1')
    gs.add_param('wandb_project', 'dp-micro-adam')
    gs.add_param('wandb_group', Template('${dataset_name}_${optimizer}_${model_name}_E=${epochs}_B=${batch_size}_WD=${weight_decay}_C=${dp_c}_S=${dp_sigma}_EPS=${dp_eps}_DELTA=${dp_delta}'))
    gs.add_param('wandb_job_type', Template('lr=${lr}'))
    gs.add_param('wandb_name', Template('seed=${seed}'))


    gs.run(
        launch_blocking=False,
        scheduling=dict(
            distributed_training=False,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params
        ),
        param_name_for_exp_root_folder='output_folder',
        exp_folder=Template(os.path.join(ROOT_RUNS, '${wandb_project}', '${wandb_group}', '${wandb_job_type}', 'seed=${seed}')))

main(
    dataset='cifar10',
    model='wrn-16-4',
    optimizer='dp-micro-adam',
    gpus=[0, 1, 2, 3, 4, 5, 6, 7],
    params={
        'seed': ['42'],
        'model_pretrained': ['0'],
        'epochs': ['100'], 
        'batch_size': [
            '4096',
        ],
        'micro_batch_size' : [
            '128',
        ],
        'aug_mult': [ 
            '0'
        ],
        'weight_decay': ['0'],
        'lr': [
            '0.001'
        ],
        'dp_c': [
            '1.0'
        ],
        'dp_sigma': [
            '4.0'
        ],
        'numeric_aprox':[
            '0'
        ],
        'dp_eps':[
           '8.0',
        ],
        'dp_delta':[
           '1e-5'
        ],
        'ema': ['0'],
    }
)
