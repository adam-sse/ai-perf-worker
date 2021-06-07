import json
import sys

from classify_image import ClassifyImage


data = json.loads(sys.argv[1])
parameters = data["parameters"]
#print("DEBUG: got parameters: " + str(parameters))

t_mean, t_stdev = ClassifyImage().measure(parameters)

performance = {"time": t_mean, "stdev": t_stdev}
#print("DEBUG: sending result: " + str(performance))
perf_data = json.dumps(performance)

print("\n")
print(perf_data)
