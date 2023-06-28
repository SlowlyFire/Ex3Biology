import sys


class Prigma:
    def __init__(self, weights, n1, n2):
        self.weights = weights
        self.nuro1 = Nuroin(n1)
        self.nuro2 = Nuroin(n2)
        self.andgate = Nuroin(-150)
        self.fitness = 0

    def calculate_fitness(self, bits, expected):
        nuro1_result = self.nuro1.output(bits, [a for (a, b) in self.weights])
        nuro2_result = self.nuro2.output(bits, [b for (a, b) in self.weights])
        result = self.andgate.output([nuro1_result, nuro2_result], self.weights)
        if result == expected:
            self.fitness += 1

    def test(self, bits, expected):
        nuro1_result = self.nuro1.output(bits, [a for (a, b) in self.weights])
        nuro2_result = self.nuro2.output(bits, [b for (a, b) in self.weights])
        result = self.andgate.output([nuro1_result, nuro2_result], self.weights)
        return result == expected

    def run(self,bits):
        nuro1result = self.nuro1.output(bits, [a for (a, b) in self.weights])
        nuro2result = self.nuro2.output(bits, [b for (a, b) in self.weights])
        result = self.andgate.output([nuro1result, nuro2result], self.weights)
        return result

class Nuroin:
    def __init__(self, bias):
        self.bias = bias

    def sigmoid(self, x):
        """Sigmoid function"""
        return x > 0

    def output(self, inp, weights):
        w = 0
        j = 0
        for i in inp:
            w += i * weights[j]
            j += 1
        sig = w + self.bias
        return self.sigmoid(sig)


def main3(wnetX_file, nnX_file):
    weights = []
    n1 = 0.0
    n2 = 0.0
    # Read input strings from wnetX file and assign to weights, and to n1, n2 (biases)
    with open(wnetX_file, 'r') as f_weights_and_biases:
        lines = f_weights_and_biases.readlines()
        for line in lines[:16]:
            weight = float(line.strip())
            weights.append((weight, -weight))
        n1 = float(lines[16].strip())
        n2 = float(lines[17].strip())

    print("Weights list:")
    for weight_tuple in weights:
        print(weight_tuple)

    # Find the best Prigma
    best_p = Prigma(weights, n1, n2)

    # Read input strings from nnX file and classify them
    output_lines_include_classification = []
    with open(nnX_file, 'r') as f_nnX_without_classification:
        for line in f_nnX_without_classification:
            bits = [int(bit) for bit in line.strip()]
            classification = best_p.run(bits)
            output_lines_include_classification.append(line.strip() + ' ' + str(classification) + '\n')

    # Write the output to a file
    output_file = 'output.txt'
    with open(output_file, 'w') as f_output:
        f_output.writelines(output_lines_include_classification)

    print("Output file '{}' created successfully.".format(output_file))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please provide the 'wnetX' file and 'nnX' file (without classifications) as command-line arguments.")
        sys.exit(1)
    wnetX_file = sys.argv[1]
    nnX_file = sys.argv[2]
    main3(wnetX_file, nnX_file)
