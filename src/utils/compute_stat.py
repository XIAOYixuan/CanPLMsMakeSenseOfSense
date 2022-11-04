import sys
from pathlib import Path
import numpy as np


def get_acc(file_path):
    with open(file_path) as fp:
        for line in fp:
            line = line.strip()
            if "eval acc" not in line:
                continue
            acc = line.split()[-1]
    return float(acc)

if __name__ == '__main__':
    """ python this.py logs_dir task_name
    """
    if len(sys.argv) ==0:
        print("python this.py logs_dir task_name")
    
    accs = []
    if len(sys.argv) == 3:
        log_dir = sys.argv[1]
        task_name = sys.argv[2]

        p = Path(log_dir)
        for x in p.iterdir():
            if task_name not in str(x):
                continue
            print("done", x)
            acc = get_acc(x)
            accs.append(acc)
    elif len(sys.argv) == 2:
        with open(sys.argv[1]) as fp:
            for line in fp:
                line = line.strip()
                if "eval acc" not in line:
                    continue
                acc = float(line.split()[-1])
                accs.append(acc)

    print(f"total len {len(accs)}")
    accs = np.asarray(accs)
    print(accs.mean(), accs.std())
