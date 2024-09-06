def convert_line_to_csv(line):

    line = line.replace('\t', ' ')

    line = line.strip()

    parts = line.split()
    csv_line = ', '.join(parts)
    
    return csv_line

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            csv_line = convert_line_to_csv(line)
            outfile.write(csv_line + '\n')

input_file = 'auto-mpg.txt'
output_file = 'auto-mpg.csv'
process_file(input_file, output_file)