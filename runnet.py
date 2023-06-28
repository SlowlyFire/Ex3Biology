import sys

class  prigma:
    def __init__ (self,wights,n1,n2):
        self.wights=wights
        self.nuro1= nuroin(n1)
        self.nuro2 = nuroin(n2)
        self.andgate= nuroin(-150)
        self.fitnees=0


    def calculatefittnes(self,bits,expected):
      w1=[]
      w2=[]
      for a,b in self.wights:
          w1.append(a)
          w2.append(b)
      nuro1result=self.nuro1.output(bits,w1)
      nuro2result = self.nuro2.output(bits,w2)
      result=self.andgate.output([nuro1result,nuro2result],[100,100])
      if result== expected:
          self.fitnees=self.fitnees+1

    def test (self,bits,expected):
        w1 = []
        w2 = []
        for a, b in self.wights:
            w1.append(a)
            w2.append(b)
        nuro1result = self.nuro1.output(bits, w1)
        nuro2result = self.nuro2.output(bits, w2)
        result = self.andgate.output([nuro1result, nuro2result], [100, 100])
        return (result == expected)

    def run(self,bits):
        w1 = []
        w2 = []
        for a, b in self.wights:
            w1.append(a)
            w2.append(b)
        nuro1result = self.nuro1.output(bits, w1)
        nuro2result = self.nuro2.output(bits, w2)
        result = self.andgate.output([nuro1result, nuro2result], [100, 100])
        return result

class nuroin :

     def __init__(self, bias ):
        self.bias=bias

     def  sigmoid(self,x):
        """Sigmoid function"""
        return (x>0)
       # return (1.0 / (1.0 + np.exp(-x)))

     def output (self,inp,whights):
         w=0
         j=0
         for i in inp:

             w=w+ i*whights[j]
             j=j+1
         sig=( w + self.bias)
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
            weights.append((-weight, weight))
        n1 = float(lines[16].strip())
        n2 = float(lines[17].strip())

    print("Weights list:")
    for weight_tuple in weights:
        print(weight_tuple)

    # Find the best Prigma
    best_p = prigma(weights, n1, n2)
   
    # Read input strings from nnX file and classify them
    output_lines_include_classification = []
    with open(nnX_file, 'r') as f_nnX_without_classification:
        for line in f_nnX_without_classification:
            bits = [int(bit) for bit in line.strip()]
            classification = best_p.run(bits)
            if classification:
                classification_int = 1
            else:
                classification_int = 0
            
            output_lines_include_classification.append(line.strip() + '   ' + str(classification_int) + '\n')

    # Write the output to a file
    output_file = 'output_with_classification_' + nnX_file +'.txt'
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
