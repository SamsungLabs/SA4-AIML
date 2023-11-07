An example of a successful log print for posenet_split_test.py:

$ python posenet_split_test.py --model_path=posenet/model-mobilenet_v1_100.pb --image_dir=posenet_split_test/images/. --output_dir=posenet_split_test/.
2023-09-21 12:29:49.740294: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2023-09-21 12:29:49.796260: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2023-09-21 12:29:50.831343: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2023-09-21 12:29:51.966277: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:51.966513: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:52.113530: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:52.113666: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:52.204172: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
2023-09-21 12:29:52.204222: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
Completed TFLite full model generation


2023-09-21 12:29:52.668011: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:52.668146: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:52.764901: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:52.765037: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:52.856058: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
2023-09-21 12:29:52.856108: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
2023-09-21 12:29:52.964945: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:52.965082: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:53.061569: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-09-21 12:29:53.061715: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-09-21 12:29:53.149491: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
2023-09-21 12:29:53.149542: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
Completed model splitting


Starting full model inference test
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Completed full model inference test. Avg. time taken per input: 99.90212500000001ms


Starting split model inference test
Completed split model inference test. Avg. time taken per input: 99.93037499999998ms


PASSED: Inference results match for input: posenet_split_test/images/two_on_bench.jpg
PASSED: Inference results match for input: posenet_split_test/images/riding_elephant.jpg
PASSED: Inference results match for input: posenet_split_test/images/skate_park_venice.jpg
PASSED: Inference results match for input: posenet_split_test/images/frisbee_2.jpg
PASSED: Inference results match for input: posenet_split_test/images/person_bench.jpg
PASSED: Inference results match for input: posenet_split_test/images/frisbee.jpg
PASSED: Inference results match for input: posenet_split_test/images/kyte.jpg
PASSED: Inference results match for input: posenet_split_test/images/looking_at_computer.jpg
PASSED: Inference results match for input: posenet_split_test/images/backpackman.jpg
PASSED: Inference results match for input: posenet_split_test/images/skiing.jpg
PASSED: Inference results match for input: posenet_split_test/images/with_computer.jpg
PASSED: Inference results match for input: posenet_split_test/images/soccer.png
PASSED: Inference results match for input: posenet_split_test/images/baseball.jpg
PASSED: Inference results match for input: posenet_split_test/images/tennis_standing.jpg
PASSED: Inference results match for input: posenet_split_test/images/tie_with_beer.jpg
PASSED: Inference results match for input: posenet_split_test/images/tennis_in_crowd.jpg
PASSED: Inference results match for input: posenet_split_test/images/snowboard.jpg
PASSED: Inference results match for input: posenet_split_test/images/fire_hydrant.jpg
PASSED: Inference results match for input: posenet_split_test/images/skate_park.jpg
PASSED: Inference results match for input: posenet_split_test/images/boy_doughnut.jpg
PASSED: Inference results match for input: posenet_split_test/images/on_bus.jpg
PASSED: Inference results match for input: posenet_split_test/images/multi_skiing.jpg
PASSED: Inference results match for input: posenet_split_test/images/truck.jpg
PASSED: Inference results match for input: posenet_split_test/images/tennis.jpg

