import sys

def compare_files(file_1, file_2):
    with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        num_diff_lines = 0

        # Compare each line of the two files as strings
        for line1, line2 in zip(lines1, lines2):
            if line1.strip() != line2.strip():
                num_diff_lines += 1

    return num_diff_lines


def main3(file_1, file_2):
    num_diff_lines = compare_files(file_1, file_2)
    print("Number of different lines: " + str(num_diff_lines))
    # compare_files(file_1, file_2)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please provide the 2 files to compare as command-line arguments.")
        sys.exit(1)
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    main3(file_1, file_2)