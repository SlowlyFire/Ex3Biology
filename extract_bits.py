import sys

# Check if the filename is provided as a command-line argument
if len(sys.argv) < 2:
    print("In order to parse the 16bits strings from the file without the classification, "
          "Please provide the input filename as a command-line argument.")
    sys.exit(1)

input_file = sys.argv[1]
output_file = 'output_' + input_file  +'.txt'

with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
    for line in f_input:
        line = line.strip()
        if line:
            bits = line.split()[0]
            f_output.write(bits + '\n')

print("Extraction complete. The 16-bit strings without the classification have been saved to", output_file)
