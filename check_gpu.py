import os

base_dir = "/home/olivia/Sim2Real/fair/slurm_logs"
files = os.listdir(base_dir)
print("NUM FILES", len(files))
start_id = 78
end_id = 121
for i in range(start_id, end_id + 1):
    file_name = [f for f in files if str(i) in f and f[-4:] == '.out']
    assert len(file_name) == 1, file_name
    file_name = file_name[0]
    with open(os.path.join(base_dir, file_name), "r") as f:
        text = f.read()
        if not ("GPUS found [PhysicalDevice" in text and "GPUs for rendering. Using device" in text):
            print("COULDN'T FIND GPU", file_name)

print("done")


