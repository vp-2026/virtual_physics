"""Prevent duplicate paid runs, including older runners without a lock."""
import fcntl
import os
import subprocess
from pathlib import Path

def acquire_model_lock(directory, model):
    directory=Path(directory)
    handle=(directory/'run.lock').open('a+')
    try:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'A {model} run already owns this output store.') from None
    for name in ('canonical132.pid','pilot.pid'):
        path=directory/name
        if not path.exists():continue
        try:pid=int(path.read_text().strip())
        except ValueError:continue
        if pid==os.getpid():continue
        command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
        if 'virtual-physics-frontier-132/scripts/run.py' in command and f'--model {model} ' in command:
            handle.close()
            raise RuntimeError(f'A {model} run is already active; reuse its checkpoint after it exits.')
    handle.seek(0);handle.truncate();handle.write(str(os.getpid())+'\n');handle.flush()
    return handle
