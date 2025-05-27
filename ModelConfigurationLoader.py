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
    itemsDict = inputDict
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
            print(conf)
            self.modelOptions[conf[:-4]] = np.load(modelDir + "/models/" + conf)
        for j in os.listdir(modelDir + "/layers"):
            self.layerOptions[int(j[:-4])] = np.load(modelDir + "/layers/" + j, allow_pickle=True).item()

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
            layerDict = self.layerOptions[self.currLayersId[i]]
            layerName = layerDict.get("name")
            if layerName == "Conv2D" or layerName == "BatchNormalization" or layerName == "Dense":
                resultModel.layers[i].set_weights(layerDict.get("weights"))
            resultModel.layers[i].trainable = False
            self.layerOptions.pop(self.currLayersId[i])  # Delete the used layers to avoid storing them twice
        # resultModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.currConfigName = configName
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
        currConf = self.currLayersId
        newConf = self.modelOptions[name]
        layerIndexDict = dict()
        for i in range(len(currConf)):
            layerIndexDict[currConf[i]] = i - 1;
            # print(str(currConf[i]) + "   " + str(i))

        changedModel = clone_model(initial_model)
        changedModel.build(initial_model.input.shape[1:])
        changedModel.compile(optimizer='adam', loss='categorical_crossentropy')
        changedModel.set_weights(initial_model.get_weights())

        layers = [l for l in changedModel.layers if not l.__class__.__name__ == 'InputLayer']
        for i in layers:
            i._name = i.name+"Old"
        input = Input(shape=tuple(changedModel.input.shape[1:]))

        layers[0].trainable = False
        x = layers[0](input)

        for i in range(2, len(newConf)):
            layerId = newConf[i]
            if layerId in layerIndexDict:
                try:
                    idx = layerIndexDict[layerId]
                    layers[idx].trainable = False
                    x = layers[idx](x)
                except:
                    print("Error Adding layer:", i, layers[i], layers[i].output.shape)
                    print(summary(Model(inputs=input, outputs=x)))
                    raise ValueError("Input shape invalid")
            else:
                tmpLayer = dictToLayer(self.layerOptions[int(layerId)])
                x = tmpLayer(x)

        resultModel = Model(inputs=input, outputs=x)
        # print("\n")
        for i in range(len(newConf)):
            layerId = newConf[i]
            # print(str(layerId))
            if layerId not in layerIndexDict:
                # print("not in")
                layerDict = self.layerOptions[layerId]
                layerName = layerDict.get("name")
                if layerName == "Conv2D" or layerName == "BatchNormalization" or layerName == "Dense":
                    resultModel.layers[i].set_weights(layerDict.get("weights"))
                resultModel.layers[i].trainable = False
                self.layerOptions.pop(layerId)

        for i in range(1, len(currConf)):
            if currConf[i] not in newConf:
                self.layerOptions[int(currConf[i])] = layerToDict(layers[i - 1])

        self.currLayersId = newConf
        self.currConfigName = name
        return resultModel
