export class DataGen {
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
                dataset[i][j] = Math.random() * (rMax - rMin) + rMin;

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
