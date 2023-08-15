import datetime
import omegaqe
import os


def output(message, my_rank, _id, use_rank=False):
    if my_rank == 0:
        f = open(f"{omegaqe.LOGGING_DIR}/{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()
    elif use_rank:
        outdir = f"{omegaqe.LOGGING_DIR}/{_id}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        f = open(f"{outdir}/_{my_rank}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()

def get_workloads(N, world_size):
    workloads = [N // world_size for _ in range(world_size)]
    for iii in range(N % world_size):
        workloads[iii] += 1
    return workloads


def get_start_end(my_rank, workloads):
    my_start = 0
    for iii in range(my_rank):
        my_start += workloads[iii]
    my_end = my_start + workloads[my_rank]
    return my_start, my_end