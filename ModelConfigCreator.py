import numpy as np
import os
from keras.models import Model, clone_model
from keras.layers import Flatten
from keras import Input
#Model Saver
def layerToDict(layer):
  """
    This function is a helper function used to 
    change a given layer to a dictionary. It is used 
    to save the layers after they have been trained
    since saving the layer directly was problematic.
    
    Args:
        layer: Layer that is needs to be changed to dictionary
    
    Returns:
        outputDict: dictionary containing the layer changed to a dict
  """
  outputDict = dict()
  if layer.__class__.__name__=="Conv2D":
    outputDict["name"] = "Conv2D"
    outputDict["filters"] = layer.filters
    outputDict["kernel_size"] = layer.kernel_size[0]
    outputDict["strides"] = layer.strides[0]
    outputDict["padding"] = layer.padding
    outputDict["activation"] = layer.activation
    outputDict["weights"] = layer.get_weights()
  elif layer.__class__.__name__=="MaxPooling2D":
    outputDict["name"] = "MaxPooling2D"
    outputDict["size"] = layer.pool_size[0]
  elif layer.__class__.__name__=="BatchNormalization":
    outputDict["name"] = "BatchNormalization"
    outputDict["weights"] = layer.get_weights()
  elif layer.__class__.__name__=="Dropout":
    outputDict["name"] = "Dropout"
    outputDict["rate"] = layer.rate
  else:
    print("Invalid layer name.")
  return outputDict
def summary(model):
  """
    This function is a helper function used to 
    help debug problems if the added dictionary does not
    allow the model to compile correctly. This function is called
    to display the model layer by layer until the part that gives the error.
    
    Args:
        model: model to be displayed
  """
  layers = [l for l in model.layers if not l.__class__.__name__=='InputLayer']
  for i in range(len(layers)):
    print(i, ": ", layers[i].name, "\t\t\t",layers[i].output.shape[1:], "\t\t\t", layers[i].trainable)

class ModelConfigCreator:
  """ Class which encapsulates the process of creating configurations"""

  def __init__(self, modelName, configName):
    """     
    Constructor for the ModelConfigCreator
    
    Args:
        modelName: Name of the model which is used to 
                   name the directory for the configurations

        configName: Current configuration name which is used to name the 
                    folder for this configuration 
    """
    self.modelName = modelName
    self.configName = configName
    if not os.path.isdir(modelName):
      os.mkdir(modelName)

  def createModel(self, inputModel, layersToRemove, layersDict):
    """
    This function creates the configuration given the main model,
    the layersToRemove and the layersDict
    
    Args:
        inputModel: Keras model which will be adjusted to 
                    create the configurations
        layersToRemove: List containing the indices for the 
                    layers that need to be removed to create the configuration
        layersDict: Dictionary containing the new layers that need to 
                    be added in order to construct the configuration. The dictionary
                    should be in the format {idx: layer, idx: layer, . . .} where idx
                    is the index the layer will be placed in and layer is the layer that 
                    needs to be added.
    
    Returns:
        model: The model with the adjusted configuration applied to it
    """    
    self.layersToRemove = layersToRemove
    self.layersDict = layersDict
    self.savedIdx = []
    #create a deep clone of the model in order to keep original model as is
    model = clone_model(inputModel)

    #Build, compile and set weights to be same as original model
    model.build(inputModel.input.shape.as_list()[1:])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.set_weights(inputModel.get_weights())
    #List layers of the model
    layers = [l for l in model.layers if not l.__class__.__name__=='InputLayer']
    #Rename the layers to avoid conflicts in the names adding _old to old layers
    for i in layers:
      i._name = i.name+"Old"
    #Initialize input layer and set it as NOT trainable
    input = Input(shape=tuple(model.input.shape.as_list()[1:]))
    layers[0].trainable = False
    x = layers[0](input)
    skipped = 0
    #For each layer do the following
    for i in range(1, len(layers)):
      #Set layer to be NOT trainable (all layers except new ones should be not trainable)
      layers[i].trainable = False
      #Replace flatten layers as they were problematic
      if layers[i].__class__.__name__=="Flatten":
        x = Flatten()(x)
      #If the layer idx is in the dict of layers to be added we add the layer from the dict
      elif i in layersDict.keys():
        x = layersDict[i](x)
        self.savedIdx.append(i-skipped+1)
      #If the layer needs to be skipped then skip it
      elif i in layersToRemove:
        skipped+=1
        continue
      #Otherwise add the same layer that already exists
      else:
        try:
          x = layers[i](x)
        except:
          print("Error Adding layer:", i, layers[i], layers[i].output.shape)
          print(summary(Model(inputs=input, outputs=x)))
          raise ValueError("Input shape invalid")
    #Create the final model and return it
    result_model = Model(inputs=input, outputs=x)
    self.model = result_model
    return result_model
  def saveModel(self):
    """
    This function ONLY saves the new layers of the adjusted model and a .cfg
    file which contains the layers that need to be removed. The layers are saved with
    the name of the idx where they should be placed. This function will output n+1 files 
    where n is the number of newly added layers and the 1 represents the cfg file.
    """    
    os.mkdir(self.modelName+"/"+self.configName)
    np.save(self.modelName+"/"+self.configName+"/cfg", self.layersToRemove)
    for i in self.savedIdx:
      np.save(self.modelName+"/"+self.configName+"/"+str(i), layerToDict(self.model.layers[i]))