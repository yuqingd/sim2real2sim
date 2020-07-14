from itertools import product
import pickle
import numpy as np

# Put these in for args like --dr where there is no value associated with the key.
PRESENT = 'present'
ABSENT = 'absent'

# This is to grid search. If you don't want to grid search, manually write in the param_args list.


sweep_params = [
    [
      {"task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_push_kettle", "kitchen_slide_kettle"],
       "mass_mean": [np.round(1.08 * .1, 2)],
       "mass_range": [.01]},
      {"task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_push_kettle", "kitchen_slide_kettle"],
       "mass_mean": [np.round(1.08 * 2, 2), np.round(1.08 * 10)],
       "mass_range": [.1]},
      {"task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_push_kettle", "kitchen_slide_kettle"],
       "mass_mean": [1.08],
       "mass_range": [.5]},

      {"task": ["kitchen_open_microwave"],
       "mass_mean": [np.round(.26 * .1, 2)],
       "mass_range": [.01]},
      {"task": ["kitchen_open_microwave"],
       "mass_mean": [np.round(.26 * 2, 2), np.round(.26 * 10)],
       "mass_range": [.1]},
      {"task": ["kitchen_open_microwave"],
       "mass_mean": [.26],
       "mass_range": [.5]},

      {"task": ["kitchen_open_cabinet"],
       "mass_mean": [np.round(3.4 * .1, 2)],
       "mass_range": [.01]},
      {"task": ["kitchen_open_cabinet"],
       "mass_mean": [np.round(3.4 * 2, 2), np.round(3.4 * 10)],
       "mass_range": [.1]},
      {"task": ["kitchen_open_cabinet"],
       "mass_mean": [3.4],
       "mass_range": [.5]},

      {"task": ["kitchen_rope"],
       "mass_mean": [np.round(.5 * .1, 2)],
       "mass_range": [.01]},
      {"task": ["kitchen_rope"],
       "mass_mean": [np.round(.5 * 2, 2), np.round(.5 * 10)],
       "mass_range": [.1]},
      {"task": ["kitchen_rope"],
       "mass_mean": [.5],
       "mass_range": [.5]},
    ],

    [{"outer_loop_version": [0], "sample_real_every": [3e6], "real_world_prob": [0]},
     {"outer_loop_version": [1], "sample_real_every": [100], "alpha": [.3]},
     {"outer_loop_version": [2], "sample_real_every": [100], "alpha": [.5]},
    ],

    [{"batch_length": [10]}],
    [{"step_size": [.01]}],
    [{"step_repeat": [50]}],
    [{"time_limit": [200]}],
    [{"steps": ["2000000"]}],
    [{"seed": [0, 1]}],
    [{"simple_randomization": [PRESENT]}],
    [{"dr": [PRESENT]}],

]

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







# sweep_params = {
#    "task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_push_kettle", "kitchen_slide_kettle"],
#    "outer_loop_version": [1],  # GO back and try with OL0 but no outer loop
#    "sample_real_every": [100],
#    "seed": [0, 1],
#    "steps": [2000000],
#    "time_limit": [200],
#    "step_repeat": [50],
#    "step_size": [.01],
#    "batch_length": [10],
#    "alpha": [.3],
#    "kettle_mass": [np.round(1.08 * .1, 2), np.round(1.08 * 2, 2), np.round(1.08 * 10)],
#    "kettle_range": [.1],
# }
# keys, values = zip(*sweep_params.items())
# param_args = [dict(zip(keys, vals)) for vals in product(*values)]

# print(param_args)
print(len(param_args))



starting_id = 122

command_strs = []
arg_mapping_strs = []
args_dict = {}

id = starting_id
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


print("=" * 20)
print("\n".join(command_strs))

print("=" * 20)
print("\n".join(arg_mapping_strs))

# We save this object in case it's easier to load it into Jupyter notebook later and extract info from it for plotting
# rather than doing it manually.
print(args_dict)
with open('args_dict.pkl', 'wb') as f:
    pickle.dump(args_dict, f)

