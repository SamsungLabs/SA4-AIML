import tensorflow as tf
import csv
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--model_path', type = Path, required = True, help = 'path to the TFLite model')
optional.add_argument('--output_path', type = Path, default = "./output.csv", help = 'path to the output')

# Get the name and size of the output of each tensor in the model
field_names = ["name", "size"]

def get_factor(dtype):
    if dtype == "<class 'numpy.float32'>":
        return 4
    elif dtype == "<class 'numpy.float16'>":
        return 2
    else:
        return 1

def main():
    args = parser.parse_args()
    model_path = args.model_path
    if not os.path.exists(args.model_path):
        raise Exception("Invalid model path")
    
    path = os.path.abspath(model_path)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    
    output_file = args.output_path
    with open(output_file, "w", newline='') as f:
      writer = csv.DictWriter(f, fieldnames = field_names)
      writer.writeheader()
      for tensor in tensor_details:
        temp_dict = {}
        temp_dict['name'] = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        size = get_factor(str(dtype))
        for x in shape:
          size = size * x
        temp_dict['size'] = size
        writer.writerow(temp_dict)
    
if __name__ == "__main__":
    main()
