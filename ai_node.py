import socket
import json

from classify_image import ClassifyImage

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

            data = json.loads(line)
            parameters = data["parameters"]
            #print("DEBUG: got parameters: " + str(parameters))

            t_mean, t_stdev = ClassifyImage().measure(parameters)

            performance = {"time": t_mean, "stdev": t_stdev}
            #print("DEBUG: sending result: " + str(performance))
            perf_data = json.dumps(performance)
            sockfile.write(perf_data + '\n')
            sockfile.flush()

finally:
    sock.close()