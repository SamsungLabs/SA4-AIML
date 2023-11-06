import csv
import collections
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--client_benchmark_file', type = Path, required = True, help = 'path to the model benchmark csv file for client')
required.add_argument('--server_benchmark_file', type = Path, required = True, help = 'path to the model benchmark csv file for server')
required.add_argument('--tensor_sizes_file', type = Path, required = True, help = 'path to the model tensor sizes csv file')
optional.add_argument('--output_path', type = Path, default = "./output.csv", help = 'path to the output csv file')
optional.add_argument('--model_size', type = float, default = 0, help = 'size of the TFLite model in kilobytes')

def find_avg_time(s: str) -> float:
    start = s.find(',')
    index = s.find(',', start + 1)
    index = index + 2
    end = s.find(',', index)
    val = float(s[index : end])
    return val

def find_name(s: str, n: int) -> str:
    start = 0
    while start >= 0 and n > 0:
        start = s.find(',', start + 1)
        n = n - 1
    if start < 0:
        return ""
    start = start + 3
    end = s.find(':', start + 1)
    end = end -1
    name = s[start : end]
    return name

def order_benchmark_data(file):
    line_dict = {}
    count = 0
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 0:
                continue
            s = ','.join(map(str, line))
            index = s.find(':')
            index = index + 1
            val = int(s[index : ])
            if line_dict.get(val) is not None:
                count = count + 1
            line_dict[val] = s
            
    ordered_dict = collections.OrderedDict(sorted(line_dict.items()))
    return ordered_dict

# Make the benchmark data consistent by finding the common layers run on both platforms
def extract_common_data(client_dict, server_dict):
    common_dict = {}
    common_dict['client_time'] = []
    common_dict['server_time'] = []
    
    for key in client_dict.keys():
        if server_dict.get(key) is None:
            continue
        s1 = client_dict[key]
        s2 = server_dict[key]
        common_dict['client_time'].append(find_avg_time(s1))
        common_dict['server_time'].append(find_avg_time(s2))
    
    return common_dict

#  Find the sizes of the common layers present in the benchmark data
def get_tensor_sizes(client_dict, server_dict, file):
    sizes_dict = {}
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0] == '':
                continue
            name = row[0]
            size = row[1]
            sizes_dict[name] = size
    
    sizes_list = []
    sizes_list.append(None)
    for key in client_dict.keys():
        if server_dict.get(key) is None:
            continue
        s = client_dict[key]
        name = find_name(s, 7)
        size = sizes_dict.get(name)
        if size is not None:
            val = float(size)
            val /= 1000
            sizes_list.append(val)
    
    return sizes_list

# Calculate cumulative times from the benchmark data
def get_cumm_time(benchmark_dict):
    client_time_list = benchmark_dict['client_time']
    server_time_list = benchmark_dict['server_time']
    if client_time_list is None or server_time_list is None:
        return None, None
    
    count = len(client_time_list)
    comp_time_client = []
    comp_time_server = []
    comp_time_client.append(0)
    for i in range(count):
        if i > 0:
            comp_time_client.append(float(client_time_list[i])+float(comp_time_client[i]))
            comp_time_server.append(float(server_time_list[i])+float(comp_time_server[i-1]))
        else:
            comp_time_client.append(float(client_time_list[0]))
            comp_time_server.append(float(server_time_list[0]))
    
    comp_time_server_2 = []
    comp_time_server_2.append(comp_time_server[count - 1])
    for i in range(count):
        comp_time_server_2.append(comp_time_server[count-1] - comp_time_server[i])
    return comp_time_client, comp_time_server_2

def get_model_sizes(model_size: float, num_layers: int):
    if num_layers == 0:
        return None
    sizes_list = []
    min_size = model_size / num_layers
    cur_size = 0
    for i in range(num_layers + 1):
        sizes_list.append(cur_size)
        cur_size += min_size
    return sizes_list
    
if __name__ == "__main__":
    args = parser.parse_args()
    client_benchmark_file = args.client_benchmark_file
    server_benchmark_file = args.server_benchmark_file
    sizes_file = args.tensor_sizes_file
    if not os.path.exists(client_benchmark_file):
        raise Exception("Invalid client benchmark file")
    if not os.path.exists(server_benchmark_file):
        raise Exception("Invalid server benchmark file")
    if not os.path.exists(sizes_file):
        raise Exception("Invalid tensor sizes file")

    ordered_client_benchmark_dict = order_benchmark_data(client_benchmark_file)
    ordered_server_benchmark_dict = order_benchmark_data(server_benchmark_file)
    
    common_benchmark_dict = extract_common_data(ordered_client_benchmark_dict, ordered_server_benchmark_dict)
    
    ordered_sizes_list = get_tensor_sizes(ordered_client_benchmark_dict, ordered_server_benchmark_dict, sizes_file)
    
    comp_time_client, comp_time_server = get_cumm_time(common_benchmark_dict)
    
    count = len(comp_time_client)
    field_names = ["Client time", "Server time", "Intermediate output size"]
    model_size = args.model_size
    model_sizes_list = []
    if model_size != 0:
        field_names.append("Intermediate model size")
        model_sizes_list = get_model_sizes(model_size, count)

    output_file = args.output_path
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        i = 0
        while i < count:
            temp_dict = {}
            temp_dict["Client time"] = comp_time_client[i]
            temp_dict["Server time"] = comp_time_server[i]
            temp_dict["Intermediate output size"] = ordered_sizes_list[i]
            if model_size != 0:
                temp_dict["Intermediate model size"] = model_sizes_list[i]
            writer.writerow(temp_dict)
            i = i + 1
