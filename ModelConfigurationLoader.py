import numpy as np
import os
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras import Input
from modelConfigCreator import summary
#Model  Loader
def dictToLayer(inputDict):
  """
    This function is a helper function used to 
    change a given dictionary to its corresponding layer. 
    It is used load the saved layers from numpy files.
    
    Args:
        inputDict: dictionary that contains info about the layer
    
    Returns:
        outputLayer: Layer after is has been changed from dictionary
  """
  outputLayer = []
  itemsDict = inputDict.item()
  if itemsDict.get("name") == "Conv2D":
    outputLayer = Conv2D(filters = itemsDict.get("filters"), 
                         kernel_size = itemsDict.get("kernel_size"), 
                         strides = itemsDict.get("strides"), 
                         padding = itemsDict.get("padding"),
                         activation= itemsDict.get("activation"))
  elif itemsDict.get("name") == "MaxPooling2D":
    size = itemsDict.get("size")
    outputLayer = MaxPooling2D(size)
  elif itemsDict.get("name") == "BatchNormalization":
    outputLayer = BatchNormalization()
  elif itemsDict.get("name") == "Dropout":
    outputLayer = Dropout(rate = itemsDict.get("rate"))
  return outputLayer
class ModelConfigurationLoader:
  """ Class which encapsulates the process of loading configurations at runtime"""
  def __init__(self, modelDir):
    """     
    Constructor for the ModelConfigurationLoader
    
    Args:
        modelDir: Directory which contains the model that needs to be loaded
    """
    self.modelOptions = dict()
    #Filter any .ipynb files
    configOptions = [x for x in os.listdir(modelDir) if not x.startswith(".")]
    for i in configOptions:
      self.modelOptions[i] = dict()
      layersToBeRemoved = np.load(modelDir+"/"+i+"/cfg.npy")
      self.modelOptions[i]["layersToBeRemoved"] = layersToBeRemoved
      for j in os.listdir(modelDir+"/"+i):
        if j!="cfg.npy":
          self.modelOptions[i][int(j[:-4])] = np.load(modelDir+"/"+i+"/"+j, allow_pickle=True)
  def loadConfig(self, name, initial_model):
    """
    This function loads a configuration to a given model at runtime
    
    Args:
        name: Name of the configuration that needs to be loaded

        initial_model: Keras model which will be adjusted to 
                    create the configurations
    
    Returns:
        result_model: The model with the adjusted configuration applied to it
    """   
    changedModel = clone_model(initial_model)

    changedModel.build(initial_model.input.shape.as_list()[1:])
    changedModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    changedModel.set_weights(initial_model.get_weights())

    layers = [l for l in changedModel.layers if not l.__class__.__name__=='InputLayer']
    for i in layers:
      i._name = i.name+"Old"
    input = Input(shape=tuple(changedModel.input.shape.as_list()[1:]))

    layers[0].trainable = False
    x = layers[0](input)
    for i in range(1, len(layers)):
      layers[i].trainable = False
      if layers[i].__class__.__name__=="Flatten":
        x = Flatten()(x)
      elif i in list(self.modelOptions[name].keys()):
        tmpLayer = dictToLayer(self.modelOptions[name][i])
        x = tmpLayer(x)
      elif i in self.modelOptions[name]["layersToBeRemoved"]:
        continue
      else:
        try:
          x = layers[i](x)
        except:
          print("Error Adding layer:", i, layers[i], layers[i].output.shape)
          print(summary(Model(inputs=input, outputs=x)))
          raise ValueError("Input shape invalid")
    result_model = Model(inputs=input, outputs=x)
    for i in self.modelOptions[name].keys():
      if isinstance(i, int):
        cond1 = self.modelOptions[name][i].item().get("name")=="Conv2D"
        cond2 = self.modelOptions[name][i].item().get("name")=="BatchNormalization"
        if cond1 or cond2:
          result_model.layers[i].set_weights(self.modelOptions[name][i].item().get("weights"))
    return result_model