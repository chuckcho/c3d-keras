import subprocess

#for i in range(9, 15):
for i in [1, 3]:
    print(i)
    cmd = ['python', 'python_subprocess.py', str(i)]
    # subprocess.check_output(["python", "python_subprocess.py", str(i)])
    # subprocess.call(["python", "python_subprocess.py", str(i)])
    # subprocess.Popen(cmd).wait()
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=35000)
    except subprocess.TimeoutExpired:
        print('Process Terminated')
        proc.kill()
        outs, errs = proc.communicate()
    finally:
        print('Process Completed')
        proc.kill()
        outs, errs = proc.communicate()
        
        
for i in range(0 + 15, 10 + 15):
    print(i)
    cmd = ['python', 'python_subprocess.py', str(i)]
    # subprocess.check_output(["python", "python_subprocess.py", str(i)])
    # subprocess.call(["python", "python_subprocess.py", str(i)])
    # subprocess.Popen(cmd).wait()
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=35000)
    except subprocess.TimeoutExpired:
        print('Process Terminated')
        proc.kill()
        outs, errs = proc.communicate()
    finally:
        print('Process Completed')
        proc.kill()
        outs, errs = proc.communicate()
