import os, sys
from subprocess import PIPE, run
import ray
import time
import numpy as np
from numpy import genfromtxt
from evaluator import Evaluator
from scipy.sparse import csr_matrix

ray.init()

def evaluate(ev, out_file):
    preds = genfromtxt(out_file, delimiter=',')
    
    # Generate score_mat
    rows, cols, data = [], [], []
    for point_num in range(preds.shape[0]):
        for prediction_num in range(preds.shape[1]):
            rows.append(point_num)
            
            pred = preds[point_num][prediction_num]
            if pred >= ev.num_labels: # I don't know why this is happening
                pred = ev.num_labels - 1
            
            cols.append(pred)
            data.append(1.0 - (0.01 * prediction_num)) # Filler score to maintain sorting
    
    score_mat = csr_matrix((data, (rows, cols)), shape = (preds.shape[0], ev.num_labels))
    
    # Prints out relevant metrics
    print(ev.evaluate(score_mat))

@ray.remote
def run_vw(data_file, model_file, out_file):
    command = [
        '../vowpalwabbit/vw',
        '-t', 
        '-i', model_file, # model
        '-d', data_file,  # data input
        '-p', out_file,   # store predictions
        '--top_k', '50'
    ]
    result = run(command, stdout=PIPE, stderr=PIPE)
    # print(result.stderr.decode("utf-8"))
    # print(result.stdout.decode("utf-8"))
    
@ray.remote
def check_progress(total_lines, threads):
    prog = 0.0
    start_time = time.time()
    while prog < 99.8:
        num_done = 0
        for thread_num in range(threads):
            out_file = "tmp_dir/preds." + str(thread_num).zfill(2)
            if not os.path.exists(out_file): continue
            result = run(
                ("wc -l " + out_file).split(), 
                stdout=PIPE, stderr=PIPE, universal_newlines=True
            )
            num_done += int(result.stdout.strip().split()[0])

        prog = 100.0 * float(num_done) / float(total_lines)
        total_time = 100.0 * (time.time() - start_time) / (prog + 0.001)
        rem_time = total_time - (time.time() - start_time)
        print(
            "Progress: {:.2f}".format(prog) + "%", 
            "; Time Elapsed: {:.2f}".format(time.time() - start_time),
            "; Remaining Time: {:.2f}".format(rem_time),
            "[", num_done, "/", total_lines, "]", end = "\r"
        )
        time.sleep(10.0)
    print()

def main(model_file, test_data_file, data_path, threads = -1):
    if threads == -1: threads = os.cpu_count()
    print("Will evaluate on", threads, "threads")
    
    # Making temp dir to store temp datasets, predictions
    os.makedirs("tmp_dir", exist_ok = True)
    
    # Split the dataset in `test_data_file` into `threads` partitions.
    print("Splitting dataset into partitions..")
    result = run(
        ("wc -l " + test_data_file).split(), 
        stdout=PIPE, stderr=PIPE, universal_newlines=True
    )
    num_lines = int(result.stdout.strip().split()[0])
    
    # Will create smaller subsets which we can run on different cores
    split_command = "split -d -l " + str((num_lines // threads) + 1) + " " + test_data_file + " tmp_dir/small_test_data."
    run(split_command.split())
    
    # Run vw.predict on each subset on a different core
    print("Evaluating for each partition..")
    ret_ids = []
    for thread_num in range(threads):
        data_file = "tmp_dir/small_test_data." + str(thread_num).zfill(2)
        out_file = "tmp_dir/preds." + str(thread_num).zfill(2)
        ret_ids.append(run_vw.remote(data_file, model_file, out_file))
    ret_ids.append(check_progress.remote(num_lines, threads))
    ret = ray.get(ret_ids)
    
    # Merge output files
    print("Combining subset predictions..")
    output_combined_cmd = "cat "
    for thread_num in range(threads):
        output_combined_cmd += "tmp_dir/preds." + str(thread_num).zfill(2) + " "
    run(output_combined_cmd.split(), stdout = open("all_preds.txt", "w"))
    
if __name__ == "__main__":
    model_file = sys.argv[1]
    test_data_file = sys.argv[2]
    data_path = sys.argv[3]
    threads = int(sys.argv[4])
    
    main(model_file, test_data_file, data_path, threads = threads)
    
    # Run custom evaluator
    print("Evaluating..")
    ev = Evaluator(data_path)
    evaluate(ev, "all_preds.txt")
