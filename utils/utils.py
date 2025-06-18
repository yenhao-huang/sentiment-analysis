import time
import os

def make_dir_with_timestamp(base_output_dir, base_log_dir):
    timestamp = int(time.time())
    output_dir = os.path.join(base_output_dir, str(timestamp))
    log_dir = os.path.join(base_log_dir, str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return output_dir, log_dir
