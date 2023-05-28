import ijson
import json

json_list = []
json_file = "Automotive.json"
with open(json_file, 'rb') as f:
    for line in f:
        json_obj = json.loads(line)
        json_list.append(json_obj)

output_file = "parsedJson.json"
with open(output_file, 'w') as f:
    json.dump(json_list, f)

