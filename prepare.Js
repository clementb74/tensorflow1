import * as Wine from './wine.js';

function makeTensors(features, labels, testSplit) {


    const numExamples = features.length;
    if (numExamples !== labels.length) {
        throw new Error('data and split have different numbers of examples')
    }

    const indices = [];
    for (let index = 0; index < numExamples; index++) {
        indices.push(index);
        
    }

    tf.util.shuffle(indices);

    const shuffledFeatures = [];
    const shuffledLabels = [];
    for (let i = 0; i < numExamples; i++) {
        shuffledFeatures.push(features[indices[i]]);
        shuffledLabels.push(labels[indices[i]]);
        
    }

    console.log('Tableau des exmples mélangés :')
    console.log(shuffledFeatures);

    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;


    const xDims = shuffledFeatures[0].length;


    const xs = tf.tensor(shuffledFeatures);
    
    const ys = tf.tensor(shuffledLabels);

    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    xTrain.print();
    const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    xTest.print();
    const yTrain = ys.slice(0, numTrainExamples);
    yTrain.print();
    const yTest = ys.slice(numTrainExamples);
    yTest.print();
    return [xTrain, yTrain, xTest, yTest];
}

export function getWineData(testSplit) {
    return tf.tidy(() => {
        
        const features = [];
        const labels = [];

        for (const examples of Wine.WINE_DATA) {
            const label = examples[0];
            const sample = examples.slice(1, examples.length);
            features.push(sample);
            labels.push(label);
            
        }
          console.log(features, labels);
 
        const [xTrain, yTrain, xTest, yTest] = makeTensors(features, labels, testSplit);
        return [xTrain, yTrain, xTest, yTest];
    })
}