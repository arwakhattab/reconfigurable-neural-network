import numpy as np
import os
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D, Dense
from keras import Input
from ModelConfigCreator import summary, layerToDict
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
    layerName = itemsDict.get("name")
    if layerName == "InputLayer":
        outputLayer = Input(shape=itemsDict.get("shape")[1:])
    elif layerName == "Conv2D":
        outputLayer = Conv2D(filters=itemsDict.get("filters"),
                             kernel_size=itemsDict.get("kernel_size"),
                             strides=itemsDict.get("strides"),
                             padding=itemsDict.get("padding"),
                             activation=itemsDict.get("activation"))
    elif layerName == "MaxPooling2D":
        size = itemsDict.get("size")
        outputLayer = MaxPooling2D(size)
    elif layerName == "BatchNormalization":
        outputLayer = BatchNormalization()
    elif layerName == "GlobalAveragePooling2D":
        outputLayer = GlobalAveragePooling2D()
    elif layerName == "Dense":
        outputLayer = Dense(units=itemsDict.get("units"), activation=itemsDict.get("activation"))
    elif itemsDict.get("name") == "Dropout":
        outputLayer = Dropout(rate=itemsDict.get("rate"))
    return outputLayer
class ModelConfigurationLoader:
    """ Class which encapsulates the process of loading configurations at runtime"""
    def __init__(self, modelDir):
        """
        Constructor for the ModelConfigurationLoader

        Args:
            modelDir: Directory which contains the model that needs to be loaded
        """
        self.modelDir = modelDir
        self.currConfigName = ""
        self.currLayersId = []
        self.modelOptions = dict()
        self.layerOptions = dict()
        # Filter any .ipynb files
        configOptions = [x for x in os.listdir(modelDir + "/models")]
        for conf in configOptions:
            self.modelOptions[conf[:-4]] = np.load(modelDir + "/models/" + conf)
        for j in os.listdir(modelDir + "/layers"):
            self.layerOptions[int(j[:-4])] = np.load(modelDir + "/layers/" + j, allow_pickle=True)

    def loadInitialConfiguration(self, configName):
        """
        This function loads a full model initially

        Args:
            configName: Name of the configuration that needs to be loaded

        Returns:
            result_model: The loaded model
        """
        self.currLayersId = self.modelOptions[configName]
        inputLayer = dictToLayer(self.layerOptions[self.currLayersId[0]])
        self.layerOptions.pop(self.currLayersId[0])
        tmpLayer = dictToLayer(self.layerOptions[self.currLayersId[1]])
        x = tmpLayer(inputLayer)
        for i in range(2, len(self.currLayersId)):
            tmpLayer = dictToLayer(self.layerOptions[self.currLayersId[i]])
            x = tmpLayer(x)
        resultModel = Model(inputs=inputLayer, outputs=x)
        for i in range(1, len(self.currLayersId)):
            layerDict = self.layerOptions[self.currLayersId[i]].item()
            layerName = layerDict.get("name")
            if layerName == "Conv2D" or layerName == "BatchNormalization" or layerName == "Dense":
                resultModel.layers[i].set_weights(layerDict.get("weights"))
            resultModel.layers[i].trainable = False
            self.layerOptions.pop(self.currLayersId[i]) # Delete the used layers to avoid storing them twice
        resultModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return resultModel


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

        changedModel.build(initial_model.input.shape[1:])
        changedModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        changedModel.set_weights(initial_model.get_weights())

        layers = [l for l in changedModel.layers if not l.__class__.__name__ == 'InputLayer']
        for i in layers:
            i._name = i.name+"Old"
        input = Input(shape=tuple(changedModel.input.shape[1:]))

        layers[0].trainable = False
        x = layers[0](input)
        for i in range(1, len(layers)):
            layers[i].trainable = False
            if layers[i].__class__.__name__ == "Flatten":
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
                cond1 = self.modelOptions[name][i].item().get("name") == "Conv2D"
                cond2 = self.modelOptions[name][i].item().get("name") == "BatchNormalization"
                if cond1 or cond2:
                    result_model.layers[i].set_weights(self.modelOptions[name][i].item().get("weights"))
        return result_model
