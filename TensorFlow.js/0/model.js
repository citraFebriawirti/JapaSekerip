/*
r = row
c = columns
*/
const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],[6,1]);
const ys = tf.tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0],[6,1]);

const model = tf.sequential();
model.add(tf.layers.dense({units:1, inputShape:[1]}));
// Use mse because it's good to use when our output is continous
model.compile({loss:'meanSquaredError',optimizer:'sgd'});
model.summary();

// Define doTraining function
async function doTraining(model){
    // model.fit(xs, ys, epochs=500, callbacks=callbacks)
    // print(epochs, loss)
    const history = await model.fit(xs, ys, {
        epochs: 500,
        callbacks:{onEpochEnd:async(epoch, logs) => {
            // get the output through console.log
            console.log("Epoch :" + epoch + " Loss:" + logs.loss);
        }}});
}

doTraining(model).then(() => {
    // Predicted x = 19, in 1r and 1c, and get the output through window.alert
    alert(model.predict(tf.tensor2d([10],[1,1])));
})