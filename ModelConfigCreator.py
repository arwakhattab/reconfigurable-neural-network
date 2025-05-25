import numpy as np
import os
from keras.models import Model, clone_model
from keras.layers import Flatten
from keras import Input
#Model Saver
def layerToDict(layer, inputShape=None):
    """
      This function is a helper function used to
      change a given layer to a dictionary. It is used
      to save the layers after they have been trained
      since saving the layer directly was problematic.

      Args:
          layer: Layer that is needs to be changed to dictionary
          inputShape: Shape of input for input layer

      Returns:
          outputDict: dictionary containing the layer changed to a dict
    """
    outputDict = dict()
    if layer.__class__.__name__ == "InputLayer":
        outputDict["name"] = "InputLayer"
        outputDict["shape"] = inputShape
    elif layer.__class__.__name__ == "Conv2D":
        outputDict["name"] = "Conv2D"
        outputDict["filters"] = layer.filters
        outputDict["kernel_size"] = layer.kernel_size[0]
        outputDict["strides"] = layer.strides[0]
        outputDict["padding"] = layer.padding
        outputDict["activation"] = layer.activation
        outputDict["weights"] = layer.get_weights()
    elif layer.__class__.__name__ == "MaxPooling2D":
        outputDict["name"] = "MaxPooling2D"
        outputDict["size"] = layer.pool_size[0]
    elif layer.__class__.__name__ == "BatchNormalization":
        outputDict["name"] = "BatchNormalization"
        outputDict["weights"] = layer.get_weights()
    elif layer.__class__.__name__ == "GlobalAveragePooling2D":
        outputDict["name"] = "GlobalAveragePooling2D"
    elif layer.__class__.__name__ == "Dropout":
        outputDict["name"] = "Dropout"
        outputDict["rate"] = layer.rate
    elif layer.__class__.__name__ == "Dense":
        outputDict["name"] = "Dense"
        outputDict["units"] = layer.units
        outputDict["activation"] = layer.activation
        outputDict["weights"] = layer.get_weights()
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

    def __init__(self, modelName, configName, model, saveToDisk=True):
        """
        Constructor for the ModelConfigCreator

        Args:
            modelName: Name of the model which is used to
                       name the directory for the configurations

            configName: Current configuration name which is used to name the
                        folder for this configuration

            model: Initial model with which the Creator is initialized

            saveToDisk: Whether the given model should be saved to disk now
        """
        self.modelName = modelName
        self.modelsPath = modelName + "/models"
        self.layersPath = modelName + "/layers"
        self.configName = configName
        self.model = model
        self.layerCount = 0
        self.layersId = []

        if not os.path.isdir(modelName):
            os.mkdir(modelName)

        if not os.path.isdir(self.modelsPath):
            os.mkdir(self.modelsPath)

        if not os.path.isdir(self.layersPath):
            os.mkdir(self.layersPath)

        for layer in model.layers:
            self.layerCount += 1
            self.layersId.append(self.layerCount)

        if saveToDisk:
            self.saveModel();

    def createModel(self, inputModel, newConfigName, layersToRemove, layersDict):
        """
        This function creates the configuration given the main model,
        the layersToRemove and the layersDict

        Args:
            inputModel: Keras model which will be adjusted to
                        create the configurations
            newConfigName: the name of the new configuration
            layersToRemove: List containing the indices for the
                        layers that need to be removed to create the configuration,
                        the indices are zero-based and do not include the input layer
            layersDict: Dictionary containing the new layers that need to
                        be added in order to construct the configuration. The dictionary
                        should be in the format {idx: layer, idx: layer, . . .} where idx
                        is the index the layer will be placed in and layer is the layer that
                        needs to be added. The indices are zero-based and do not include the
                        input layer

        Returns:
            model: The model with the adjusted configuration applied to it
        """
        newLayersId = []
        # create a deep clone of the model in order to keep original model as is
        model = clone_model(inputModel)

        # Build, compile and set weights to be same as original model
        model.build(inputModel.input.shape[1:])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.set_weights(inputModel.get_weights())
        # List layers of the model
        layers = [l for l in model.layers if not l.__class__.__name__ == 'InputLayer']
        # Rename the layers to avoid conflicts in the names adding _old to old layers
        for i in layers:
            i._name = i.name + "Old"
        # Initialize input layer and set it as NOT trainable
        input = Input(shape=tuple(model.input.shape[1:]))
        newLayersId.append(self.layersId[0])
        layers[0].trainable = False
        x = layers[0](input)
        newLayersId.append(self.layersId[1])
        # For each layer do the following
        for i in range(1, len(layers)):
            # Set layer to be NOT trainable (all layers except new ones should be not trainable)
            layers[i].trainable = False
            # Replace flatten layers as they were problematic
            if layers[i].__class__.__name__ == "Flatten":
                x = Flatten()(x)
                newLayersId.append(self.layersId[i + 1])
            # If the layer idx is in the dict of layers to be added we add the layer from the dict
            elif i in layersDict.keys():
                x = layersDict[i](x)
                self.layerCount += 1
                newLayersId.append(self.layerCount)
            # If the layer needs to be skipped then skip it
            elif i in layersToRemove:
                continue
            # Otherwise add the same layer that already exists
            else:
                try:
                    x = layers[i](x)
                    newLayersId.append(self.layersId[i + 1])
                except:
                    print("Error Adding layer:", i, layers[i], layers[i].output.shape)
                    print(summary(Model(inputs=input, outputs=x)))
                    raise ValueError("Input shape invalid")
        # Create the final model and return it
        result_model = Model(inputs=input, outputs=x)
        self.configName = newConfigName
        self.model = result_model
        self.layersId = newLayersId
        return result_model

    def saveModel(self):
        """
        This function ONLY saves the new layers of the adjusted model and a file which contains
        the ids of the layers of this configuration. Each layer is saved with in a .npy file
        whose name is the layer's id. This function will output n + 1 files
        where n is the number of newly added layers and the 1 represents the file
        describing the layers of the configuration.
        """
        np.save(self.modelsPath + "/" + self.configName, self.layersId)
        for i in range(len(self.layersId)):
            self.saveLayer(i)

    def saveLayer(self, idx, overwrite=False):
        """
        This function saves the layer at index idx in a file named after its unique number

        Args:
            idx: Index of the layer of the model to be saved

            overwrite: Flag to determine whether to overwrite the layer if it is already saved
        """
        layerPath = self.layersPath + "/" + str(self.layersId[idx])
        if not overwrite and os.path.exists(layerPath + ".npy"):
            return
        np.save(layerPath, layerToDict(self.model.layers[idx], self.model.input_shape))



