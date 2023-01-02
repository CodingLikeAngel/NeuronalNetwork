
class DataGen {
    constructor(inputs, sets, min, binary) {
        this.dataset = this.Generate(inputs, sets, min, binary);
    }

    Generate(inputs, sets, min, binary) {
        const dataset = [];

        const rMax = 1;
        const rMin = min;

        for (let i = 0; i < sets; i++) {
            dataset[i] = [];
            for (let j = 0; j < inputs; j++) {
                dataset[i][j] = (Math.random() * (rMax - rMin)) + rMin;

                // Clamp for binary values.
                if (binary) {
                    if (dataset[i][j] < 0.5) {
                        dataset[i][j] = 0;
                    } else {
                        dataset[i][j] = 1;
                    }
                }
            }
        }

        dataset[0][0] = 0;
        dataset[0][1] = 0;

        dataset[1][0] = 0;
        dataset[1][1] = 1;

        dataset[2][0] = 1;
        dataset[2][1] = 0;

        dataset[3][0] = 1;
        dataset[3][1] = 1;

        return dataset;
    }

    get getDataset() {
        return this.dataset;
    }
}



class Ann {
    constructor(genotype, inputs, outputs) {
        const gen_size = genotype.length; //store the genontype size
        const blocks_length = inputs * outputs; //length of the blocks of the genotype
        let hidden = gen_size / blocks_length - 1; // -1 because of the i_o

        // Make sure "hidden" is a finite, non-negative integer
        if (!Number.isFinite(hidden) || hidden < 0) {
            throw new RangeError("Invalid value for 'hidden'");
        }
        hidden = Math.floor(hidden);

        //arrays for storing values of the neurons
        this.neurons_I = new Array(inputs).fill(0); //input value
        this.neurons_H = new Array(hidden).fill(0); //hidden value
        this.neurons_O = new Array(outputs).fill(0); //output value

        //arrays for storing mapping of weights
        this.mapping_I_O = new Array(inputs).fill(0).map(() => new Array(outputs).fill(0)); //input -> output mapping
        this.mapping_H_I = new Array(hidden).fill(0).map(() => new Array(inputs).fill(0)); //hidden -> input mapping
        this.mapping_H_O = new Array(hidden).fill(0).map(() => new Array(outputs).fill(0)); //hidden -> output mapping

        //arrays for storing values of the weights
        this.weights_I_O = new Array(inputs).fill(0).map(() => new Array(outputs).fill(0)); //input -> output weight
        this.weights_H_I = new Array(hidden).fill(0).map(() => new Array(inputs).fill(0)); //hidden -> input weight
        this.weights_H_O = new Array(hidden).fill(0).map(() => new Array(outputs).fill(0)); //hidden -> output weight

        this.weights_H_BIAS = new Array(hidden).fill(0); //hidden bias weight
        this.weights_O_BIAS = new Array(outputs).fill(0); //output bias weight

        this.WeightMapping(genotype, gen_size, blocks_length, inputs, hidden, outputs);
    }

    WeightMapping(genotype, gen_size, blocks_length, inputs, hidden, outputs) {
        for (let i = 0; i < gen_size; i++) {
            const val = genotype[i];

            //input ->  output connections mapping
            if (i < blocks_length) {
                let input = 0;
                let substraction = i;

                while (substraction >= outputs) {
                    input++;
                    substraction -= outputs;
                }

                this.mapping_I_O[input][substraction] = val;
            }

            //hidden connections mapping
            else {
                let hidden_neuron = 0;
                let substraction = i - blocks_length;

                while (substraction >= blocks_length) {
                    hidden_neuron++;
                    substraction -= blocks_length;
                }

                let input_index, output_index;

                if (inputs - 1 === 0) {
                    input_index = 0;
                } else {
                    input_index = Math.floor(substraction / (inputs - 1));
                }

                output_index = substraction % (outputs);
                if (hidden_neuron < hidden) {

                    //input ->  hidden connections mapping
                    if (this.mapping_H_I[hidden_neuron][input_index] == null || this.mapping_H_I[hidden_neuron][input_index] === 0) {
                        this.mapping_H_I[hidden_neuron][input_index] = val;
                    }

                    //hidden ->  output connections mapping
                    if (this.mapping_H_O[hidden_neuron][output_index] == null || this.mapping_H_O[hidden_neuron][output_index] === 0) {
                        this.mapping_H_O[hidden_neuron][output_index] = val;
                    }

                }
            }
        }
        this.printWeightMapping(inputs, hidden, outputs);
    }


    printWeightMapping(inputs, hidden, outputs) {
        console.log("###_WEIGHT MAPPING_###\n");
        console.log("IO_WM: ");
        for (let i = 0; i < inputs; i++) {
            for (let j = 0; j < outputs; j++) {
                console.log(`I[${i}] -> O[${j}]:__ `, this.mapping_I_O[i][j]);
                if (j + 1 < outputs) {
                    console.log();
                } else {
                    console.log("\n\n");
                }
            }
        }
        console.log("\n\n");

        console.log("HI_WM: ");
        for (let i = 0; i < hidden; i++) {
            for (let j = 0; j < inputs; j++) {
                console.log(`H[${i}] -> I[${j}]:__ `, this.mapping_H_I[i][j]);
                if (j + 1 < inputs) {
                    console.log();
                } else {
                    console.log("\n\n");
                }
            }
        }
        console.log("\n\n");

        console.log("HO_WM: ");
        for (let i = 0; i < hidden; i++) {
            for (let j = 0; j < outputs; j++) {
                console.log(`H[${i}] -> O[${j}]:__ `, this.mapping_H_O[i][j]);
                if (j + 1 < outputs) {
                    console.log();
                } else {
                    console.log("\n\n");
                }
            }
        }
        console.log("#############\n");
    }
}


class Trainer {

   
    constructor(ann, learn_factor) {
        this.learn_factor = learn_factor;
        this.ann = ann;
        this.length_I = ann.neurons_I.length;
        this.length_H = ann.neurons_H.length;
        this.length_O = ann.neurons_O.length;
        this.errors_H = [];
        this.errors_O = [];
        this.constants = {
        DEBUG : false,
        ACTIVATION : "RELU",
        }
    }

  
    WeightsGen() {
        const rand = Math.random();
        for (let i = 0; i < this.length_H; i++) {
            // bias
            this.ann.weights_H_BIAS[i] = rand * (1 - -1) + -1;
            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_H_I[i][j] === 1) {
                    this.ann.weights_H_I[i][j] = rand * (1 - -1) + -1;
                }
            }
            for (let j = 0; j < this.length_O; j++) {
                if (this.ann.mapping_H_O[i][j] === 1) {
                    this.ann.weights_H_O[i][j] = rand * (1 - -1) + -1;
                }
            }
        }
        for (let i = 0; i < this.length_O; i++) {
            // bias
            this.ann.weights_O_BIAS[i] = rand * (1 - -1) + -1;
            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_I_O[j][i] === 1) {
                    this.ann.mapping_I_O[j][i] = rand * (1 - -1) + -1;
                }
            }
        }
    }

    FeedForward(dataset, datasetIteration) {
        // reset neurons
        this.ann.neurons_I = new Array(this.length_I); //input value
        this.ann.neurons_H = new Array(this.length_H); //length_H value
        this.ann.neurons_O = new Array(this.length_O); //output value

        //output of input neurons
        for (let i = 0; i < this.length_I; i++) {
            //fill input neurons with values in this iteration of dataset 
            this.ann.neurons_I[i] = dataset[datasetIteration][i];
        }

        //real output of length_H neurons
        for (let i = 0; i < this.length_H; i++) {
            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_H_I[i][j] == 1) {
                    if (!this.ann.neurons_H[i]) {
                        this.ann.neurons_H[i] = 0;
                    }
                    this.ann.neurons_H[i] += this.ann.neurons_I[j] * this.ann.weights_H_I[i][j];
                }
            }

            this.ann.neurons_H[i] += this.ann.weights_H_BIAS[i];


            // if (Const.AFUNC == Activation.TANH) {
            //     // Utilizar la función tanh de JavaScript para calcular el valor hiperbólico tangente
            //     ann.neurons_H[i] = Math.tanh(ann.neurons_H[i]);
            // } else if (Const.AFUNC == Activation.SIGMOID) {
            //     // Utilizar la función sigmoid definida por el usuario para calcular el valor sigmoide
            //     ann.neurons_H[i] = Sigmoid(ann.neurons_H[i]);
            // } else if (Const.AFUNC == Activation.UMBRAL) {
            //     // Utilizar un umbral para clasificar el valor como 0 o 1
            //     if (ann.neurons_H[i] < 0.5) {
            //         ann.neurons_H[i] = 0;
            //     } else if (ann.neurons_H[i] >= 0.5) {
            //         ann.neurons_H[i] = 1;
            //     }
            // }
            // else if (Const.AFUNC == Activation.RELU) {
            // Utilizar la función ReLU para calcular el valor rectificado linealmente
            this.ann.neurons_H[i] = Math.max(0, this.ann.neurons_H[i]);
            // }

            for (let i = 0; i < this.length_O; i++) {
                // First i_o
                for (let j = 0; j < this.length_I; j++) {
                    if (this.ann.mapping_I_O[j][i] === 1) {
                        if (!this.ann.neurons_O[i]) {
                            this.ann.neurons_O[i] = 0;
                        }
                        this.ann.neurons_O[i] += this.ann.neurons_I[j] * this.ann.weights_I_O[j][i];
                    }
                }
                // Then h_o
                for (let j = 0; j < this.length_H; j++) {
                    if (this.ann.mapping_H_O[j][i] === 1) {
                        if (!this.ann.neurons_O[i]) {
                            this.ann.neurons_O[i] = 0;
                        }
                        this.ann.neurons_O[i] += this.ann.neurons_H[j] * this.ann.weights_H_O[j][i];
                    }
                }

                this.ann.neurons_O[i] += this.ann.weights_O_BIAS[i];

                // Activation function
                // if (Const.AFUNC === Activation.TANH) {
                //     // mytan
                //     // annx.neurons_O[i] = HyperbolicTan(annx.neurons_O[i]);
                //     // tanh
                //     this.ann.neurons_O[i] = Math.tanh( this.ann.neurons_O[i]);

                // }
            }
            //	if(Const.DEBUG)
            //		PrintNeuronsValues(dataset_iteration);
        }
    }

    BackPropagation() {
        this.errors_O = new Array(this.length_O);

        for (let i = 0; i < this.length_O; i++) {
            // if (Const.AFUNC === Activation.TANH) {
            //     errors_O[i] = (1 - Math.pow(ann.neurons_O[i], 2)) * (this.ExpectedValue_XOR(i, ann) - ann.neurons_O[i]);
            // } else if (Const.AFUNC === Activation.SIGMOID) {
            //     errors_O[i] = ann.neurons_O[i] * (1 - ann.neurons_O[i]) * (this.ExpectedValue_XOR(i, ann) - ann.neurons_O[i]);
            // } else if (Const.AFUNC === Activation.UMBRAL) {
            //     errors_O[i] = 1 * (this.ExpectedValue_XOR(i, ann) - ann.neurons_O[i]);
            // }

            this.errors_O[i] = Math.max(0, this.ann.neurons_H[i]);
            // if (Const.DEBUG) {
            //   console.log(`output error_ ${i}____${errors_O[i]}`);
            // }
        }
        //length_H errors
        this.errors_H = new Array(this.length_H);

        for (let i = 0; i < this.length_H; i++) {
            let sum_Eo_Who = 0;

            for (let j = 0; j < this.length_O; j++) {
                if (this.ann.mapping_H_O[i][j] === 1) {
                    sum_Eo_Who += this.ann.weights_H_O[i][j] * this.errors_O[j];
                }
            }

            //   if (Const.AFUNC === Activation.TANH) {
            //     errors_H[i] = (1 - Math.pow(ann.neurons_H[i], 2)) * sum_Eo_Who;
            //   } else if (Const.AFUNC === Activation.SIGMOID) {
            //     errors_H[i] = ann.neurons_H[i] * (1 - ann.neurons_H[i]) * sum_Eo_Who;
            //   } else if (Const.AFUNC === Activation.UMBRAL) {
            //     errors_H[i] = 1 * sum_Eo_Who;
            //   }
            // else if (Const.AFUNC === Activation.RELU) {
            this.errors_H[i] = sum_Eo_Who * (this.ann.neurons_H[i] > 0 ? 1 : 0);
            // }

            Math.max(0, ann.neurons_H[i]);

            //   if (Const.DEBUG) {
            //     console.log(`hidden error_ ${i}____${errors_H[i]}`);
            //   }
        }
    }


    ExpectedValue_XOR(output, ann) {
        switch (output) {
            case 0:
                if (this.ann.neurons_I[0] !== this.ann.neurons_I[1]) {
                    //  if (Const.DEBUG) console.log("EXPECTED___" + 1);
                    return 1;
                } else {
                    // if (Const.DEBUG) console.log("EXPECTED___" + 0);
                    return 0;
                }
            default:
                return 999999;
        }
    }


    TrainingOffline(trainingIterations) {
        // Primero, genera pesos aleatorios
        this.WeightsGen();

        const sets = 4;
        const min = 0;
        const binary = true;

        const datagen = new DataGen(this.length_I, sets, min, binary);
        //  if (Const.DEBUG) {
        //    datagen.PrintDataSet();
        // }

        const dataset = datagen.getDataset;

        let lastEvalError = Number.MAX_VALUE;
        let evalError = 0;
        for (let i = 0; i < trainingIterations; i++) {
            console.log(`____________________________ITERATION___${i + 1} of ${trainingIterations}`);

            //   if (Const.DEBUG) {
            //     PrintWeights();
            //   }

            // if (Const.ETYPE === EvalType.EARLY_STOP && i % Const.RANGE === 0) {
            //     annLast = ann;
            // evalError /= 100;
            // if (evalError > lastEvalError) {
            //     console.log('VICTORYYYYYYYY');
            //     console.log(`FINISHED IN ITERATION:__  ${i - 1}`);

            //  //   ann = annLast;

            //     this.FeedForward(dataset, 0);
            //     this.FeedForward(dataset, 1);
            //     this.FeedForward(dataset, 2);
            //     this.FeedForward(dataset, 3);
            //     break;
            // } else if (i !== 0) {
            //     lastEvalError = evalError;
            //     evalError = 0;
            // }
            // }

            // RESET DELTAS
            this.deltas_I_O = new Array(this.length_I).fill().map(() => new Array(this.length_O).fill(0));
            this.deltas_H_I = new Array(this.length_H).fill().map(() => new Array(this.length_I).fill(0));
            this.deltas_H_O = new Array(this.length_H).fill().map(() => new Array(this.length_O).fill(0));

            this.deltas_H_BIAS = new Array(this.length_H).fill(0);
            this.deltas_O_BIAS = new Array(this.length_O).fill(0);

            let error = 0;
            for (let j = 0, max = dataset.length; j < max; j++) {
                this.FeedForward(dataset, j);
                this.BackPropagation();
                this.DeltaWeights();

                let currentError = 0;
                for (let k = 0; k < this.length_O; k++) {
                    currentError = this.ExpectedValue_XOR(0) - this.ann.neurons_O[k];
                    error += currentError ** 2;
                }

                console.log(`________________________________________________________EXPECTED_____${this.ExpectedValue_XOR(0)}`);
                console.log(`__________________________________________________________NEURON_____${this.ann.neurons_O[0]}`);
                console.log("");
            }

            error /= dataset.length;
            evalError += Math.pow(error, 2);

            console.log("________________________________________________________GLOBAL_ERROR___" + error);
            console.log("\n");

            // if (Const.ETYPE === EvalType.FITNESS && error < Const.FITNESS) {
            //   console.log("VICTORYYYYYYYY");
            //   console.log("FINISHED IN ITERATION:__  " + i);

            //   FeedForward(dataset, 0);
            //   FeedForward(dataset, 1);
            //   FeedForward(dataset, 2);
            //   FeedForward(dataset, 3);
            //   break;
            // }


            this.WeightsCorrection();
         //   this.ann.printWeightMapping(this.length_I, this.length_H, this.length_O);
        }

        // if(Const.ETYPE == EvalType.EARLY_STOP)
        // {
        // 	FeedForward(dataset,0);
        // 	FeedForward(dataset,1);
        // 	FeedForward(dataset,2);
        // 	FeedForward(dataset,3);
        // }
        console.log("END");
    }


    DeltaWeights() {
        //deltas of i_o
        for (let i = 0; i < this.length_O; i++) {
            //bias
            this.deltas_O_BIAS[i] += this.learn_factor * this.errors_O[i];

            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_I_O[j][i] == 1)
                    this.deltas_I_O[j][i] += this.learn_factor * this.errors_O[i] * this.ann.neurons_I[j];
            }
        }

        //deltas of h_o
        for (let i = 0; i < this.length_H; i++) {
            //bias
            this.deltas_H_BIAS[i] += this.learn_factor * this.errors_H[i];

            for (let j = 0; j < this.length_O; j++) {
                if (this.ann.mapping_H_O[i][j] == 1)
                    this.deltas_H_O[i][j] += this.learn_factor * this.errors_O[j] * this.ann.neurons_H[i];
            }
        }

        //deltas of i_h
        for (let i = 0; i < this.length_H; i++) {
            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_H_I[i][j] == 1)
                    this.deltas_H_I[i][j] += this.learn_factor * this.errors_H[i] * this.ann.neurons_I[j];
            }
        }

        //   if (Const.DEBUG) PrintDeltas();
    }

    WeightsCorrection() {
        // weights of i_o
        for (let i = 0; i < this.length_I; i++) {
            for (let j = 0; j < this.length_O; j++) {
                if (this.ann.mapping_I_O[i][j] == 1) {
                    this.ann.weights_I_O[i][j] += this.deltas_I_O[i][j];
                }
            }
        }

        // weights of h_o
        for (let i = 0; i < this.length_O; i++) {
            // bias
            this.ann.weights_O_BIAS[i] += this.deltas_O_BIAS[i];

            for (let j = 0; j < this.length_H; j++) {
                if (this.ann.mapping_H_O[j][i] == 1) {
                    this.ann.weights_H_O[j][i] += this.deltas_H_O[j][i];
                }
            }
        }
        // weights of i_h
        for (let i = 0; i < this.length_H; i++) {
            // bias
            this.ann.weights_H_BIAS[i] += this.deltas_H_BIAS[i];

            for (let j = 0; j < this.length_I; j++) {
                if (this.ann.mapping_H_I[i][j] == 1) {
                    this.ann.weights_H_I[i][j] += this.deltas_H_I[i][j];
                }
            }
        }

        //   if (Const.DEBUG) {
        //     System.out.println("WEIGHTS CORRECTED");
        //   }
    }


}


function setGenotype() {
    // const genotype = [];
    // for (let i = 0; i < 4; i++) {
    //     genotype.push(1);
    // }

    // return genotype;
    let randomString = [];

    let randomNumber = 0;


  randomNumber = Math.floor(Math.random() * 50);


    // Generamos una cadena de 50 elementos
    for (let i = 0; i < randomNumber; i++) {
      // Generamos un número aleatorio entre 0 y 1
      let randomNumber = Math.random();
      
      // Si el número aleatorio es menor que 0.5, asignamos 0 a la cadena,
      // de lo contrario, asignamos 1
      if (randomNumber < 0.5) {
        randomString.push(0);
      } else {
        randomString.push(1);
      }
    }
    if(randomString % 2 == 0)
    {
        randomString.push(1);
    }
    // Imprimimos la cadena generada
   // console.log(randomString);
  
   return randomString;

}

//const ann = new Ann([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1], 2, 3);

const genotype_xor = setGenotype();
const ann = new Ann(genotype_xor, 2, 1); // genotype, inputs and outputs
const trainer = new Trainer(ann, 0.01); // genotype, inputs and outputs
trainer.TrainingOffline(3000);