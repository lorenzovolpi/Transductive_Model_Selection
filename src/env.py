import json
import os
import subprocess as sb

import cap
import quapy as qp

_hostname = sb.run(["hostname"], capture_output=True).stdout.decode("UTF-8").strip()
with open("env.json", "r") as f:
    ext_env = json.load(f).get(_hostname, cap.env)
cap.env |= ext_env

PROJECT = "tms"
root_dir = os.path.join(cap.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0

_valid_problems = ["binary", "multiclass"]
PROBLEM = "binary"
