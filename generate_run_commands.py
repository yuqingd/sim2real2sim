from itertools import product
import pickle

# Put these in for args like --dr where there is no value associated with the key.
PRESENT = 'present'
ABSENT = 'absent'

# This is to grid search. If you don't want to grid search, manually write in the param_args list.

sweep_params = {
    "task": ["kitchen_push_kettle_burner", "kitchen_slide_kettle_burner", "kitchen_pick_kettle_burner"],
    "outer_loop_version": [0, 1, 2],
    "sample_real_every": [100],
    "seed": [0, 1],
    "steps": [2000000],
    "time_limit": [200],
    "step_repeat": [10],
    "step_size": [.05],
    "batch_length": [10],
    "dr_option": ['inaccurate_large_range'],
    "dr": [PRESENT],
}


starting_id = 54

print(f"{1:02d}")


keys, values = zip(*sweep_params.items())
param_args = [dict(zip(keys, vals)) for vals in product(*values)]
print(param_args)
print(len(param_args))


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

