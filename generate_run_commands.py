from itertools import product
import pickle
import numpy as np

# Put these in for args like --dr where there is no value associated with the key.
PRESENT = 'present'
ABSENT = 'absent'

# This is to grid search. If you don't want to grid search, manually write in the param_args list.
starting_id = 2668

sweep_params_1 = [
    [{"task": ["kitchen_push_kettle_burner", "kitchen_open_cabinet", "kitchen_rope"],
      "time_limit": [250],
      },
     {"task": ["dmc_cup_catch", "dmc_finger_spin", "dmc_walker_walk"],
      "time_limit": [1000],
     },
    ],
    [
     {"dr": [ABSENT],  # Oracle
      "sample_real_every": [3e6],
      "real_world_prob": [0],
      "outer_loop_version": [0],
     },

     {"dr": [PRESENT],  # Mean-centered Dr
      "sample_real_every": [3e6],
      "real_world_prob": [0],
      "outer_loop_version": [0],
      "mean_scale": [1],
      "dr_option": ["nonconflicting_dr", "all_dr"],
     },


     {"dr": [PRESENT],  # baseline
      "sample_real_every": [3e6],
      "real_world_prob": [0],
      "outer_loop_version": [0],
      "dr_option": ["nonconflicting_dr", "all_dr"],
      "mean_scale": [.1, 2, 5],
     },


     {"dr": [PRESENT],  # OL1
      "sample_real_every": [100],
      "outer_loop_version": [1],
      "dr_option": ["nonconflicting_dr", "all_dr"],
      "mean_scale": [.1, 2, 5],
     }
    ],


   # [{"buffer_size": [2000]}],  # TODO: should we add this?
    [{"seed": [0, 1, 2]}],
    [{"time_limit": [250]}],
    [{"steps": [1e6]}],
    [{"outer_loop_version": [1]}],
    [{"binary_prediction": [True]}],
    [{"alpha": [.05]}],
    [{"grayscale": [False]}],
    [{"ol1_episodes": [10]}],
    [{"individual_loss_scale": [False]}],
    [{"sim_params_loss_scale": [.001]}],
    [{"log_images": [False]}],
    [{"early_termination": [False]}],
    [{"eval_every": [1e3]}],
]








sweep_params_list = [
    sweep_params_1,
]

id = starting_id
total_num_commands = 0
args_dict = {}


for sweep_params in sweep_params_list:
    # Each param set has a group of parameters.  We will find each product of groups of parameters
    all_args = []
    for param_set1 in sweep_params:
        assert isinstance(param_set1, list)

        param_set1_args = []
        for param_set2 in param_set1:
            assert isinstance(param_set2, dict)

            # Find all combos same as we did before
            keys, values = zip(*param_set2.items())
            param_set2_product = [dict(zip(keys, vals)) for vals in product(*values)]
            param_set1_args += param_set2_product

        all_args.append(param_set1_args)

    all_product = list(product(*all_args))

    # Merge list of dictionaries into single one
    param_args = [{k: v for d in param_set0 for k, v in d.items()} for param_set0 in all_product]

    total_num_commands += len(param_args)


    command_strs = []
    arg_mapping_strs = []



    for args in param_args:
        full_id = f"ID{id:04d}"
        args_command = ""
        args_mapping = f"ID: {full_id}; Parameters:"
        for k, v in args.items():
            args_mapping += f" {k}: {v},"
            if v == PRESENT:
                args_command += f" --{k}"
            elif v == ABSENT:
                continue
            else:
                args_command += f" --{k} {v}"
        full_command = f'sbatch -J "{full_id}" --export=args="--id {full_id} {args_command}" slurm_wrapper.sh'
        command_strs.append(full_command)
        arg_mapping_strs.append(args_mapping)
        args_dict[full_id] = args
        id += 1


    # print("=" * 20)
    print("\n".join(command_strs))

print("TOTAL COMMANDS", total_num_commands)

# print("=" * 20)
# print("\n".join(arg_mapping_strs))

# We save this object in case it's easier to load it into Jupyter notebook later and extract info from it for plotting
# rather than doing it manually.
# print(args_dict)
with open(f'args_dict{starting_id}.pkl', 'wb') as f:
    pickle.dump(args_dict, f)

