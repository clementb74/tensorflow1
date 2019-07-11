import * as Tensors from './prepare.js';
import * as Kernel from './kernel.js'


let xTrain, yTrain, xTest, yTest;


[xTrain, yTrain, xTest, yTest] = Tensors.getWineData(0.2);


let model = Kernel.makeModel();


Kernel.trainModel(model, xTrain, yTrain, xTest, yTest);
