class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize the input weights with random values
        this.inputWeights = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let weights = [];
            for (let j = 0; j < this.inputSize; j++) {
                weights.push(Math.random());
            }
            this.inputWeights.push(weights);
        }

        // Initialize the input biases with random values
        this.inputBiases = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            this.inputBiases.push(Math.random());
        }

        // Initialize the output weights with random values
        this.outputWeights = [];
        for (let i = 0; i < this.outputSize; i++) {
            let weights = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                weights.push(Math.random());
            }
            this.outputWeights.push(weights);
        }

        // Initialize the output biases with random values
        this.outputBiases = [];
        for (let i = 0; i < this.outputSize; i++) {
            this.outputBiases.push(Math.random());
        }
    }

    NeuralNetwork(weights) {
        let index = 0;

        // Inicializar los pesos y sesgos de la capa oculta con los valores entrenados
        this.inputWeights = [];
        this.inputBiases = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let w = [];
            for (let j = 0; j < this.inputSize; j++) {
                let val = weights[index++];
                if (!isNaN(val)) {
                    w.push(val);
                } else {
                    w.push(0);
                }
            }
            this.inputWeights.push(w);
            let bias = weights[index++];
            if (!isNaN(bias)) {
                this.inputBiases.push(bias);
            } else {
                this.inputBiases.push(0);
            }
        }

        // Inicializar los pesos y sesgos de la capa de salida con los valores entrenados
        this.outputWeights = [];
        this.outputBiases = [];
        for (let i = 0; i < this.outputSize; i++) {
            let w = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                let val = weights[index++];
                if (!isNaN(val)) {
                    w.push(val);
                } else {
                    w.push(0);
                }
            }
            this.outputWeights.push(w);
            let bias = weights[index++];
            if (!isNaN(bias)) {
                this.outputBiases.push(bias);
            } else {
                this.outputBiases.push(0);
            }
        }
    }


    backpropagation(inputs, labels, learningRate) {
        // Hacer una predicción para los inputs dados
        let prediction = this.predict(inputs);

        // Calcular el error entre la predicción y las etiquetas
        let error = labels[0] - prediction;

        // Calcular las activaciones de los neuronas de la capa oculta
        let hiddenLayerActivations = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let activation = this.inputBiases[i];
            for (let j = 0; j < this.inputSize; j++) {
                activation += this.inputWeights[i][j] * inputs[j];
            }
            hiddenLayerActivations.push(this.activate(activation));
        }

        // Calcular las activaciones de los neuronas de la capa de salida
        let outputActivations = [];
        for (let i = 0; i < this.outputSize; i++) {
            let activation = this.outputBiases[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                activation += this.outputWeights[i][j] * hiddenLayerActivations[j];
            }
            outputActivations.push(this.activate(activation));
        }

        // Calcular los gradientes de los pesos y sesgos de la capa de salida
        let outputWeightGradients = [];
        for (let i = 0; i < this.outputSize; i++) {
            let gradients = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                gradients.push(-error * this.activateDerivative(outputActivations[i]) * hiddenLayerActivations[j]);
            }
            outputWeightGradients.push(gradients);
        }
        let outputBiasGradients = [];
        // Ajustar los pesos y sesgos de la capa de salida
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.outputWeights[i][j] -= learningRate * outputWeightGradients[i][j];
            }
            this.outputBiases[i] -= learningRate * outputBiasGradients[i];
        }

        // Calcular los gradientes de los pesos y sesgos de la capa oculta
        let hiddenWeightGradients = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let gradients = [];
            for (let j = 0; j < this.inputSize; j++) {
                gradients.push(-error * this.activateDerivative(hiddenLayerActivations[i]) * inputs[j]);
            }
            hiddenWeightGradients.push(gradients);
        }
        let hiddenBiasGradients = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenBiasGradients.push(-error * this.activateDerivative(hiddenLayerActivations[i]));
        }

        // Ajustar los pesos y sesgos de la capa oculta
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                this.inputWeights[i][j] -= learningRate * hiddenWeightGradients[i][j];
            }
            this.inputBiases[i] -= learningRate * hiddenBiasGradients[i];
        }
    }




    train(inputs, labels, learningRate) {
        // Make a prediction for the given inputs
        let prediction = this.predict(inputs);

        // Calculate the error between the prediction and the labels
        let error = labels[0] - prediction;

        // Calculate the activations of the neurons in the hidden layer
        let hiddenLayerActivations = [];
        let hiddenLayerOutput = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let activation = this.inputBiases[i];
            for (let j = 0; j < this.inputSize; j++) {
                activation += this.inputWeights[i][j] * inputs[j];
            }
            hiddenLayerActivations.push(this.activate(activation));
            hiddenLayerOutput.push(hiddenLayerActivations[i]);
        }

        // Calculate the activations of the neurons in the output layer
        let outputActivations = [];
        for (let i = 0; i < this.outputSize; i++) {
            let activation = this.outputBiases[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                activation += this.outputWeights[i][j] * hiddenLayerActivations[j];
            }
            outputActivations.push(this.activate(activation));
        }

        // Calculate the gradients of the output layer weights and biases
        let outputWeightGradients = [];
        for (let i = 0; i < this.outputSize; i++) {
            let gradients = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                gradients.push(-error * this.activateDerivative(outputActivations[i]) * hiddenLayerOutput[j]);
            }
            outputWeightGradients.push(gradients);
        }
        let outputBiasGradients = [];
        for (let i = 0; i < this.outputSize; i++) {
            outputBiasGradients.push(-error * this.activateDerivative(outputActivations[i]));
        }

        // Calculate the gradients of the hidden layer weights and biases
        let hiddenWeightGradients = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let gradients = [];
            for (let j = 0; j < this.inputSize; j++) {
                gradients.push(-error * this.activateDerivative(hiddenLayerActivations[i]) * inputs[j]);
            }
            hiddenWeightGradients.push(gradients);
        }


        let hiddenBiasGradients = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenBiasGradients.push(-error * this.activateDerivative(hiddenLayerActivations[i]));
        }

        // Update the weights and biases using the gradients and the learning rate
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.outputWeights[i][j] -= learningRate * outputWeightGradients[i][j];
            }
            this.outputBiases[i] -= learningRate * outputBiasGradients[i];
        }


        let sumSquaredGradients = 0;
        // En cada iteración del proceso de entrenamiento...
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                if (i >= 0 && i < this.outputSize && j >= 0 && j < this.hiddenSize) {
                    // Actualizamos la suma acumulada de los gradientes al cuadrado
                    sumSquaredGradients += outputWeightGradients[i][j] ** 2;
                    // Actualizamos el peso utilizando el factor de aprendizaje adaptativo
                    if (sumSquaredGradients !== 0) {
                        this.outputWeights[i][j] -= learningRate * outputWeightGradients[i][j] / Math.sqrt(sumSquaredGradients);
                    }
                }
            }
        }
    }

    activateDerivative(x) {
        // Assume that the activate function is the sigmoid function
        //  return this.activate(x) * (1 - this.activate(x));

        return x > 0 ? 1 : 0;
    }

    activate(x) {
        // Assume that the activate function is the sigmoid function
        //  return 1 / (1 + Math.exp(-x));
        return Math.max(0, x);
    }

    mapWeights2() {
        // Inicializar los pesos y sesgos de la capa oculta aleatoriamente
        this.inputWeights = [];
        this.inputBiases = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let weights = [];
            for (let j = 0; j < this.inputSize; j++) {
                weights.push(Math.random() * 2 - 1);
            }
            this.inputWeights.push(weights);
            this.inputBiases.push(Math.random() * 2 - 1);
        }

        // Inicializar los pesos y sesgos de la capa de salida aleatoriamente
        this.outputWeights = [];
        this.outputBiases = [];
        for (let i = 0; i < this.outputSize; i++) {
            let weights = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                weights.push(Math.random() * 2 - 1);
            }
            this.outputWeights.push(weights);
            this.outputBiases.push(Math.random() * 2 - 1);
        }
    }

    mapWeights() {
        // Initialize the weight map with zeros
        let weights = [];
        for (let i = 0; i < this.inputSize; i++) {
            weights.push([]);
            for (let j = 0; j < this.hiddenSize; j++) {
                weights[i].push(0);
            }
        }

        // Map the weights from the input layer to the hidden layer
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                weights[i][j] = this.inputWeights[i][j];
            }
        }


        for (let i = 0; i < this.hiddenSize; i++) {
            weights.push([]);
            if (this.outputWeights[i]) {
                for (let j = 0; j < this.outputWeights[i].length; j++) {
                    weights[i].push(this.outputWeights[i][j]);
                }
            }
        }

        // console.log(weights);
        return weights;
    }




    predict(inputs) {
        // Validate the input data
        if (!Array.isArray(inputs)) {
            throw new Error('Input data must be an array');
        }
        if (inputs.length !== this.inputSize) {
            throw new Error(`Input data must contain ${this.inputSize} elements`);
        }

        // Calculate the activations of the neurons in the hidden layer
        let hiddenLayerActivations = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let activation = this.inputBiases[i];
            for (let j = 0; j < this.inputSize; j++) {
                activation += this.inputWeights[i][j] * inputs[j];
            }
            hiddenLayerActivations.push(this.activate(activation));
        }

        // Calculate the output of the hidden layer
        let hiddenLayerOutput = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenLayerOutput.push(this.activate(hiddenLayerActivations[i]));
        }

        // Calculate the activations of the neurons in the output layer
        let outputActivations = [];
        for (let i = 0; i < this.outputSize; i++) {
            let activation = this.outputBiases[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                activation += this.outputWeights[i][j] * hiddenLayerOutput[j];
            }
            outputActivations.push(this.activate(activation));
        }

        // Calculate the output of the network
        let output = [];
        for (let i = 0; i < this.outputSize; i++) {
            output.push(this.activate(outputActivations[i]));
        }

        return output;
    }



}




// const myNN = new NeuralNetwork(2, 34, 1);
// myNN.mapWeights2();
// // Train the neural network on the XOR gate using the inputs [[0, 0], [0, 1], [1, 0], [1, 1]] and the labels [0, 1, 1, 0]
// for (let i = 0; i < 100000; i++) {
//     myNN.train([0, 0], [0], 0.01);
//     myNN.train([0, 1], [1], 0.01);
//     myNN.train([1, 0], [1], 0.01);
//     myNN.train([1, 1], [0], 0.01);
//     //   myNN.mapWeights();
// }

// Map the weights of the neural network


// Test the neural network on the XOR gate
//console.log("should output a value close to 0" + "    " + myNN.predict([0, 0])); // should output a value close to 0
//console.log("should output a value close to 1" + "    " + myNN.predict([0, 1])); // should output a value close to 1
//console.log("should output a value close to 1" + "    " + myNN.predict([1, 0])); // should output a value close to 1
//console.log("should output a value close to 0" + "    " + myNN.predict([1, 1])); // should output a value close to 0


 let myNN2 = new NeuralNetwork(2, 12, 1);
myNN2.mapWeights();

// //const inputWeights = JSON.parse(fs.readFileSync('./inputWeights.json'));



// const fs = require('fs');

// // Write the inputWeights array to a file
// if(myNN.predict([0, 0]) == 0 && myNN.predict([0, 1]) > 0.98 && myNN.predict([1, 0]) > 0.98 && myNN.predict([1, 1]) == 0 )
// {
//     fs.writeFileSync('./inputWeights2.json', JSON.stringify(myNN.inputWeights));
//     fs.writeFileSync('./inputBiases2.json', JSON.stringify(myNN.inputBiases));
//     fs.writeFileSync('./outputWeights2.json', JSON.stringify(myNN.outputWeights));
//     fs.writeFileSync('./outputBiases2.json', JSON.stringify(myNN.outputBiases));
//     console.log("stored values");
// }


// myNN2.inputWeights =  JSON.parse(fs.readFileSync('./inputWeights.json'));
// myNN2.inputBiases =JSON.parse(fs.readFileSync('./inputBiases.json'));
// myNN2.outputWeights = JSON.parse(fs.readFileSync('./outputWeights.json'));
// myNN2.outputBiases = JSON.parse(fs.readFileSync('./outputBiases.json'));



// //console.log( myNN2.mapWeights());
// console.log("should output a value close to 0" + "    " + myNN2.predict([0, 0])); // should output a value close to 0
// console.log("should output a value close to 0" + "    " + myNN2.predict([1, 0])); // should output a value close to 0
// console.log("should output a value close to 0" + "    " + myNN2.predict([0, 1])); // should output a value close to 0
// console.log("should output a value close to 0" + "    " + myNN2.predict([0, 0])); // should output a value close to 0
// console.log("should output a value close to 0" + "    " + myNN2.predict([0, 0])); // should output a value close to 0
// console.log("should output a value close to 1" + "    " + myNN2.predict([0, 1])); // should output a value close to 1
// console.log("should output a value close to 1" + "    " + myNN2.predict([1, 0])); // should output a value close to 1
// console.log("should output a value close to 0" + "    " + myNN2.predict([1, 1])); // should output a value close to 0