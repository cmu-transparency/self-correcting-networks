import sys
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np


def parse_line(line, typ):
  line = line.strip().split(",")
  return [typ(x) for x in line if x != ""]

def convert_nnet_to_keras(filename):
  if not filename.endswith(".nnet"):
    print("Nnet network should have the .nnet extension")
    return
  with open(filename, "r") as f:
    contents = f.readlines()
    contents = iter(contents)
    line = next(contents)
    while line.startswith("//"):
      # Skip comments at the beginning
      line = next(contents)
    # First line in nnet format is nbr_layers, input size, output size, max 
    # layer size
    nbr_layers = int(line.strip().split(",")[0])
    line = next(contents)
    # Second line is the input size, followed by the output size of each layer
    layers_sizes = parse_line(line, int)
    assert (len(layers_sizes) == nbr_layers + 1), \
      "The sizes of layers do not correspond to the number of layers"
    line = next(contents)

    model = Sequential()
    # Special handling of first layer to give input_shape
    model.add(Dense(
      layers_sizes[1], activation ='relu', input_shape=(layers_sizes[0],)))
    layers_sizes = layers_sizes[2:]
    for n in layers_sizes[:-1]:
      model.add(Dense(n, activation='relu'))

    model.add(Dense(layers_sizes[-1]))

    # Skip the 5 lines about constraints on inputs
    line = next(contents)
    line = next(contents)
    line = next(contents)
    line = next(contents)
    # line = next(contents)

    model.summary()

    print('>>>>> Parsing weights and biases')
    for l in model.layers:
      print(f'    layer: {l.name}')
      weights = []
      bias = []
      y = l.output_shape[1]
      x = l.input_shape[1]  
      # Parsing weights for this layer
      for i in range(y):
        line = next(contents)
        weight_i = parse_line(line, float)
        assert (len(weight_i) == x), weight_i
        weights.append(weight_i)
      # Transforming them into a numpy array
      weights = np.array(weights,dtype="float32") 
      # Transposing them to fit Keras format
      weights = np.transpose(weights)
      # Parsing bias for this layer
      for i in range(y):
        line = next(contents)
        bias_i = parse_line(line, float)
        assert(len(bias_i) == 1), bias_i
        bias.append(bias_i)
      bias = np.array(bias, dtype="float32").reshape((y,))
      # Replace weights in layer by those from the nnet file
      l.set_weights([weights, bias])

  filename = filename[:-5]
  # Replace .nnet extension by .h5 and save the model
  model.save(filename + ".h5")

if len(sys.argv) == 1:
  for filename in glob('models/*.nnet'):
    print(f'>>>>>>> converting {filename}')
    convert_nnet_to_keras(filename)
elif len(sys.argv) == 2:
  convert_nnet_to_keras(sys.argv[1])
else:
  print("Usage: python nnet_to_keras.py filename.nnet")