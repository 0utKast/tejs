import "bootstrap/dist/css/bootstrap.css";
import * as tf from "@tensorflow/tfjs";
document.getElementById("hola").innerText = "Hola";

// Usamos un modelo secuencial para regresión lineal
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Seleccionamos pérdida y optimizador para modelo
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Altura y peso como los datos de entrenamiento
const height = tf.tensor2d([1.82, 1.70, 1.87, 1.54, 1.63, 1.72], [6, 1]);
const weight = tf.tensor2d([80, 75, 85, 65, 72, 75], [6, 1]);

// Entrenando el modelo
model.fit(height, weight, { epochs: 500 }).then(() => {
  // Usamos modelo para predeccir peso para una altura de 183 cm
  model.predict(tf.tensor2d([1.80], [1, 1])).print();
});
