import socket
import subprocess
import sys

ip = '0.0.0.0'
port = 50021

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((ip, port))
    sock.listen()
    con, adr = sock.accept()
    with con, con.makefile(mode='rw', encoding="utf8") as sockfile:
        print('Connection from', adr)
        while True:
            line = sockfile.readline()
            if not line:
                break

            result = subprocess.check_output(["python3", "ai_executor.py", line])
            result = result.decode(sys.getdefaultencoding())

            lines = result.splitlines()
            print(str(lines))
            perf_data = lines[len(lines) - 1]

            sockfile.write(perf_data)
            sockfile.flush()

finally:
    sock.close()
