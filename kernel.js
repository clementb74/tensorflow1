

const BATCH_SIZE = 100;

export function makeModel()
{

   const model = tf.sequential();

   model.add(tf.layers.dense({
       units: 10,
       activation: 'relu',
       inputShape: [13],
       useBias: true,
       kernelInitializer: 'zeros',
       biasInitializer: 'zeros'

   }));


   model.add(tf.layers.dense({
       units: 1,
   }));



   model.summary();


   const optimizer = tf.train.adam(0.05);
   model.compile({
       optimizer: optimizer,
       loss: 'meanSquaredError',
   });

   return model;


}


export async function trainModel(model, xTrain, yTrain, xTest, yTest)
{
    const history = await model.fit(xTrain, yTrain, {
        batchSize: BATCH_SIZE,
        epochs: 10,
        evaluationSplit: 0.15,
        callbacks: {
            onEpochEnd: async (epoch, log) => { console.log(epoch, log); }
        }
    });

    // dispatchWeights(model);
    validate(model, xTrain, yTrain);
    let evaluation = model.evaluate(xTest, yTest, {batchSize: BATCH_SIZE});
    evaluation.print();
}


export const dispatchWeights = (model) => {
    model.weights.forEach(w => {
        console.log(w.name, w.shape, w.val.dataSync());
    });
}

function validate(model, xTrain, yTrain)
{
    tf.tidy(() => {
        let predictions = model.predict(xTrain).dataSync();
        let given = yTrain.dataSync();
        let error = 0;
        let success = 0;
        for (let i = 0; i < predictions.length; i++) {
            error += Math.pow((predictions[i] - given[i]), 2);
            console.log(`${predictions[i].toFixed(3)} | ${Math.round(predictions[i])} - ${given[i]}`);
            success += (Math.round(predictions[i]) == given[i]) ? 1 : 0;

        }
        console.log(`Succès : ${(success * 100 / predictions.length).toFixed(2)}% de prédictions justes`);
        console.log(`Erreur moyenne : ${Math.sqrt(error/predictions.length)}`);
    })
}
