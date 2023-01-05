import { DataGen } from "./DataGen.js";
class Ann {
  constructor(genotype, inputs, outputs) {
    const gen_size = genotype.length; //store the genontype size
    const blocks_length = inputs * outputs; //length of the blocks of the genotype
    let hidden = gen_size / blocks_length - 1; // -1 because of the i_o

    // Make sure "hidden" is a finite, non-negative integer
    if (!Number.isFinite(hidden) || hidden < 0) {
      hidden = Math.round(hidden);
    }
    hidden = Math.floor(hidden);

    //arrays for storing values of the neurons
    this.neurons_I = new Array(inputs).fill(0); //input value
    this.neurons_H = new Array(hidden).fill(0); //hidden value
    this.neurons_O = new Array(outputs).fill(0); //output value

    //arrays for storing mapping of weights
    this.mapping_I_O = new Array(inputs)
      .fill(0)
      .map(() => new Array(outputs).fill(0)); //input -> output mapping
    this.mapping_H_I = new Array(hidden)
      .fill(0)
      .map(() => new Array(inputs).fill(0)); //hidden -> input mapping
    this.mapping_H_O = new Array(hidden)
      .fill(0)
      .map(() => new Array(outputs).fill(0)); //hidden -> output mapping

    //arrays for storing values of the weights
    this.weights_I_O = new Array(inputs)
      .fill(0)
      .map(() => new Array(outputs).fill(0)); //input -> output weight
    this.weights_H_I = new Array(hidden)
      .fill(0)
      .map(() => new Array(inputs).fill(0)); //hidden -> input weight
    this.weights_H_O = new Array(hidden)
      .fill(0)
      .map(() => new Array(outputs).fill(0)); //hidden -> output weight

    this.weights_H_BIAS = new Array(hidden).fill(0); //hidden bias weight
    this.weights_O_BIAS = new Array(outputs).fill(0); //output bias weight

    this.deltas_H_BIAS = [];
    this.deltas_O_BIAS = [];

    this.WeightMapping(
      genotype,
      gen_size,
      blocks_length,
      inputs,
      hidden,
      outputs
    );
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

        output_index = substraction % outputs;
        if (hidden_neuron < hidden) {
          //input ->  hidden connections mapping
          if (
            this.mapping_H_I[hidden_neuron][input_index] == null ||
            this.mapping_H_I[hidden_neuron][input_index] === 0
          ) {
            this.mapping_H_I[hidden_neuron][input_index] = val;
          }

          //hidden ->  output connections mapping
          if (
            this.mapping_H_O[hidden_neuron][output_index] == null ||
            this.mapping_H_O[hidden_neuron][output_index] === 0
          ) {
            this.mapping_H_O[hidden_neuron][output_index] = val;
          }
        }
      }
    }
    // this.printWeightMapping(inputs, hidden, outputs);
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
      DEBUG: false,
      ACTIVATION: "RELU",
    };
  }

  WeightsGen() {
    const rand = Math.random();
    for (let i = 0; i < this.length_H; i++) {
      // bias
      this.ann.weights_H_BIAS[i] = Math.random() * (1 - -1) + -1;
      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_H_I[i][j] === 1) {
          this.ann.weights_H_I[i][j] = Math.random() * (1 - -1) + -1;
        }
      }
      for (let j = 0; j < this.length_O; j++) {
        if (this.ann.mapping_H_O[i][j] === 1) {
          this.ann.weights_H_O[i][j] = Math.random() * (1 - -1) + -1;
        }
      }
    }
    for (let i = 0; i < this.length_O; i++) {
      // bias
      this.ann.weights_O_BIAS[i] = Math.random() * (1 - -1) + -1;
      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_I_O[j][i] === 1) {
          this.ann.mapping_I_O[j][i] = Math.random() * (1 - -1) + -1;
        }
      }
    }
  }

  FeedForward(dataset, datasetIteration) {
    for (let i = 0; i < this.length_I; i++) {
      this.ann.neurons_I[i] = dataset[datasetIteration][i];
    }

    for (let i = 0; i < this.length_H; i++) {
      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_H_I[i][j] == 1) {
          this.ann.neurons_H[i] +=
            this.ann.neurons_I[j] * this.ann.weights_H_I[i][j];
        }
      }

      this.ann.neurons_H[i] += this.ann.weights_H_BIAS[i];
      this.ann.neurons_H[i] = this.relu(this.ann.neurons_H[i]);
    }

    for (let i = 0; i < this.length_O; i++) {
      //first i_o
      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_I_O[j][i] == 1) {
          this.ann.neurons_O[i] +=
            this.ann.neurons_I[j] * this.ann.weights_I_O[j][i];
        }
      }
      //then h_o
      for (let j = 0; j < this.length_H; j++) {
        if (this.ann.mapping_H_O[j][i] == 1) {
          this.ann.neurons_O[i] +=
            this.ann.neurons_H[j] * this.ann.weights_H_O[j][i];
        }
      }

      this.ann.neurons_O[i] += this.ann.weights_O_BIAS[i];
      this.ann.neurons_O[i] = this.relu(this.ann.neurons_O[i]);
    }
  }

  relu(x) {
    return x > 0 ? 1 : 0;
  }

  BackPropagation() {
    this.errors_O = new Array(this.length_O);

    for (let i = 0; i < this.length_O; i++) {
      this.errors_O[i] =
        1 * (this.ExpectedValue_XOR(i) - this.ann.neurons_O[i]);
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
      this.errors_H[i] = sum_Eo_Who * this.ann.neurons_H[i];
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
    this.WeightsGen();

    const sets = 4;
    const min = 0;
    const binary = true;

    const datagen = new DataGen(this.length_I, sets, min, binary);

    const dataset = datagen.getDataset;
    let error = 0;

    let lastEvalError = Number.MAX_VALUE;
    let evalError = 0;
    for (let i = 0; i < trainingIterations; i++) {
      // console.log(
      //     `____________________________ITERATION___${i + 1
      //     } of ${trainingIterations}`
      // );

      this.ann.deltas_I_O = new Array(this.length_I)
        .fill()
        .map(() => new Array(this.length_O).fill(0));
      this.ann.deltas_H_I = new Array(this.length_H)
        .fill()
        .map(() => new Array(this.length_I).fill(0));
      this.ann.deltas_H_O = new Array(this.length_H)
        .fill()
        .map(() => new Array(this.length_O).fill(0));

      this.ann.deltas_H_BIAS = new Array(this.length_H).fill(0);
      this.ann.deltas_O_BIAS = new Array(this.length_O).fill(0);

      for (let j = 0, max = dataset.length; j < max; j++) {
        this.FeedForward(dataset, j);
        this.BackPropagation();
        this.DeltaWeights();

        let currentError = 0;
        for (let k = 0; k < this.length_O; k++) {
          currentError = this.ExpectedValue_XOR(0) - this.ann.neurons_O[k];
          error += currentError ** 2;
        }

        // console.log(
        //     `________________________________________________________EXPECTED_____${this.ExpectedValue_XOR(
        //         0
        //     )}`
        // );
        // console.log(
        //     `__________________________________________________________NEURON_____${this.ann.neurons_O[0]}`
        // );
        // console.log("");
      }
      error /= dataset.length;
      evalError += Math.pow(error, 2);

      // console.log(
      //     "________________________________________________________GLOBAL_ERROR___" +
      //     error
      // );
      // console.log("\n");

      this.WeightsCorrection();

      if (error == 0) {
        //  console.log("win" + j);
        return { error: error, iteration: i };
      }
    }

    //        console.log("END");

    return { error: error, iteration: -1 };
  }

  DeltaWeights() {
    //deltas of i_o
    for (let i = 0; i < this.length_O; i++) {
      //bias
      this.ann.deltas_O_BIAS[i] += this.learn_factor * this.errors_O[i];

      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_I_O[j][i] == 1)
          this.deltas_I_O[j][i] +=
            this.learn_factor * this.errors_O[i] * this.ann.neurons_I[j];
      }
    }

    //deltas of h_o
    for (let i = 0; i < this.length_H; i++) {
      //bias
      this.ann.deltas_H_BIAS[i] += this.learn_factor * this.errors_H[i];

      for (let j = 0; j < this.length_O; j++) {
        if (this.ann.mapping_H_O[i][j] == 1)
          this.ann.deltas_H_O[i][j] +=
            this.learn_factor * this.errors_O[j] * this.ann.neurons_H[i];
      }
    }

    //deltas of i_h
    for (let i = 0; i < this.length_H; i++) {
      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_H_I[i][j] == 1)
          this.ann.deltas_H_I[i][j] +=
            this.learn_factor * this.errors_H[i] * this.ann.neurons_I[j];
      }
    }

    //   if (Const.DEBUG) PrintDeltas();
  }

  WeightsCorrection() {
    // weights of i_o
    for (let i = 0; i < this.length_I; i++) {
      for (let j = 0; j < this.length_O; j++) {
        if (this.ann.mapping_I_O[i][j] == 1) {
          this.ann.weights_I_O[i][j] += this.ann.deltas_I_O[i][j];
        }
      }
    }

    // weights of h_o
    for (let i = 0; i < this.length_O; i++) {
      // bias
      this.ann.weights_O_BIAS[i] += this.ann.deltas_O_BIAS[i];

      for (let j = 0; j < this.length_H; j++) {
        if (this.ann.mapping_H_O[j][i] == 1) {
          this.ann.weights_H_O[j][i] += this.ann.deltas_H_O[j][i];
        }
      }
    }
    // weights of i_h
    for (let i = 0; i < this.length_H; i++) {
      // bias
      this.ann.weights_H_BIAS[i] += this.ann.deltas_H_BIAS[i];

      for (let j = 0; j < this.length_I; j++) {
        if (this.ann.mapping_H_I[i][j] == 1) {
          this.ann.weights_H_I[i][j] += this.ann.deltas_H_I[i][j];
        }
      }
    }
  }
}

function setGenotype() {
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
  if (randomString % 2 == 0) {
    randomString.push(1);
  }
  // Imprimimos la cadena generada
  //   console.log(randomString);

  return randomString;
}

//const ann = new Ann([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1], 2, 3);

function reproduce(individuo1, individuo2, tasaMutación = 0.01) {
  // Cruzar los genes de los individuos para crear un nuevo individuo
  let nuevoIndividuo = "";
  for (let i = 0; i < individuo1.genotype_xor.length; i++) {
    if (Math.random() < 0.5) {
      nuevoIndividuo += individuo1.genotype_xor[i];
    } else {
      nuevoIndividuo += individuo2.genotype_xor[i];
    }
  }

  // Mutar el nuevo individuo al azar
  for (let i = 0; i < nuevoIndividuo.length; i++) {
    if (Math.random() < tasaMutación) {
      nuevoIndividuo[i] = nuevoIndividuo[i] === "0" ? "1" : "0";
    }
  }
  //console.log(nuevoIndividuo);
  return nuevoIndividuo;
}

let individuals = [{}];

for (let i = 0; i < 10000; i++) {
  const genotype_xor = setGenotype();
  const ann = new Ann(genotype_xor, 2, 1); // genotype, inputs and outputs
  const trainer = new Trainer(ann, 0.01); // genotype, inputs and outputs
  const fitness = trainer.TrainingOffline(5000);

  if (fitness.error == 0 && fitness.iteration > 0) {
    let individual = {
      genotype_xor: genotype_xor,
      iteration: fitness.iteration,
    };
    individuals.push(individual);
    //    console.log(individual);
  }
}

let best = individuals.map((element) => element.iteration);
best.sort((a, b) => a - b);
let num1 = best[0];
let num2 = best[1];

console.log(num1, num2); // Output: 1 2

const first = individuals.findIndex((element) => {
  return element.iteration == num1;
});

const second = individuals.findIndex((element) => {
  return element.iteration == num2;
});

console.log(individuals[first]);
console.log(individuals[second]);

if (
  individuals[first].genotype_xor &&
  individuals[first].genotype_xor.length > 0 &&
  individuals[second].genotype_xor &&
  individuals[second].genotype_xor.length > 0
) {
  const child = reproduce(individuals[first], individuals[second]);

  // Convertir la cadena en un array utilizando el método split()
  let array = child.split("");

  // Convertir cada elemento del array a un número utilizando el método map()
  let arrayNum = array.map((element) => parseInt(element));

  //console.log(arrayNum); // Output: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

  const ann = new Ann(arrayNum, 2, 1); // genotype, inputs and outputs
  const trainer = new Trainer(ann, 0.01); // genotype, inputs and outputs
  const fitness = trainer.TrainingOffline(1000);
}
