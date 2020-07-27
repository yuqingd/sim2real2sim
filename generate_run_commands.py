from itertools import product
import pickle
import numpy as np

# Put these in for args like --dr where there is no value associated with the key.
PRESENT = 'present'
ABSENT = 'absent'

# This is to grid search. If you don't want to grid search, manually write in the param_args list.
starting_id = 884

sweep_params_1 = [
    [{"task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_push_kettle", ],
      "alpha": [0.1],
      "dr_option": ["partial_dr"],
      "seed": [2, 3, 4,],
      },
     {"task": ["kitchen_slide_kettle",],
      "alpha": [0.1],
      "dr_option": ["partial_dr"],
      "seed": [2, 3, 4, 5, 6],
      },
     {"task": ["kitchen_open_cabinet"],
      "alpha": [0.3],
      "dr_option": ["partial_dr"],
      "seed": [2, 3, 4, 5, 6],
      },
     {"task": ["kitchen_rope",],
      "alpha": [0.3],
      "dr_option": ["all_dr"],
      "seed": [2, 3, 4,],
      },
     {"task": ["dmc_cup_catch", "dmc_walker_walk", "dmc_finger_spin", "dmc_cheetah_run"],
      "alpha": [0.3],
      "dr_option": ["all_dr"],
      "time_limit": [1000],
      "random_crop": [True],
      "seed": [2, 3, 4,],
      }],
    # Oracle and baseline (suite 1, 2).  Note that alpha will not be used here and dr_option won't be used in the baseline.
    [{"dr": [PRESENT, ABSENT],
      "sample_real_every": [3e6],
      "real_world_prob": [0],
      "outer_loop_version": [0],},
     # outer loop 1, no inner loop (suite 5)
     {"dr": [PRESENT],
      "sample_real_every": [3e6],
      "real_world_prob": [0],
      "outer_loop_version": [1],
     },
     # outer loop conditions (suite 3,4,6)
    {"dr": [PRESENT],
      "sample_real_every": [100],
      "outer_loop_version": [0, 1, 2],
     }],
    [
     {"mean_scale": [5], "range_scale": [.5]},
     {"mean_scale": [1], "range_scale": [5]},
      ],
    [{"log_images": [False]}],
    [{"buffer_size": [2000]}],
    [{"early_termination": [False]}],
]


# ABLATIONS
sweep_params_2 = [
    # Suites
    [{"task": ["kitchen_push_kettle_burner",],
      "alpha": [0.1],
      "dr_option": ["partial_dr"],
      "mean_scale": [5], "range_scale": [.5],
      "outer_loop_version": [2],
      },
     {"task": ["kitchen_push_kettle",],
      "alpha": [0.1],
      "dr_option": ["partial_dr"],
      "mean_scale": [5], "range_scale": [.5],
      "outer_loop_version": [1],
      },
     {"task": ["kitchen_push_kettle",],
      "alpha": [0.1],
      "dr_option": ["partial_dr"],
      "mean_scale": [1], "range_scale": [5],
      "outer_loop_version": [2],
      },
     {"task": ["kitchen_rope",],
      "alpha": [0.3],
      "dr_option": ["all_dr"],
      "mean_scale": [.1,], "range_scale": [.01],
      "outer_loop_version": [1],
      },
     ],


    # Ablations
    [# Different amounts of sample_real_every
      {"sample_real_every": [10, 1000],
      },
     # Different amounts of eval_every
      {"eval_every": [1000, 100000],
       "sample_real_every": [100],
      },
    # Different amounts of interventions
    {"sample_real_every": [200],
     "num_real_world": [2],
    },
    {"sample_real_every": [1000],
     "num_real_world": [10],
    },
    ],

    # Constant parameters
    [{"dr": [PRESENT]}],
    [{"log_images": [False]}],
    [{"buffer_size": [2000]}],
    [{"early_termination": [False]}],
    [{"seed": [0, 1, 2]}],
]

# Different amounts of alpha
sweep_params_3 = [
    # Suites
[{"task": ["kitchen_push_kettle_burner",],
      "alpha": [0.3, 0.6, 0.9],
      "dr_option": ["partial_dr"],
      "mean_scale": [5], "range_scale": [.5],
      "outer_loop_version": [2],
      },
     {"task": ["kitchen_push_kettle",],
      "alpha": [0.3, 0.6, 0.9],
      "dr_option": ["partial_dr"],
      "mean_scale": [5], "range_scale": [.5],
      "outer_loop_version": [1],
      },
     {"task": ["kitchen_push_kettle",],
      "alpha": [0.3, 0.6, 0.9],
      "dr_option": ["partial_dr"],
      "mean_scale": [1], "range_scale": [5],
      "outer_loop_version": [2],
      },
     {"task": ["kitchen_rope",],
      "alpha": [0.1, 0.6, 0.9],
      "dr_option": ["all_dr"],
      "mean_scale": [.1,], "range_scale": [.01],
      "outer_loop_version": [1],
      },
     ],

    # Constant parameters
    [{"sample_real_every": [100]}],
    [{"dr": [PRESENT]}],
    [{"log_images": [False]}],
    [{"buffer_size": [2000]}],
    [{"early_termination": [False]}],
    [{"seed": [0, 1,2]}],
]





# 1176 (default)
# Only 3 seeds for ablations

sweep_params_list = [
    sweep_params_1,
    sweep_params_2, sweep_params_3
]

id = starting_id
total_num_commands = 0

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
    args_dict = {}


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
# with open('args_dict.pkl', 'wb') as f:
#     pickle.dump(args_dict, f)

