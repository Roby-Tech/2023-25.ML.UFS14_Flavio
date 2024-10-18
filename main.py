import os

print("###################")

def list_files_and_directories(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

model_dir = os.environ['SM_MODEL_DIR']

with open(model_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono il modello di ml')
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono i log del training')

input_dir = os.environ['SM_INPUT_DIR']
list_files_and_directories(input_dir)

with open(input_dir + '/data/training/train.csv', 'r') as fp:
    lines = len(fp.readlines())
    print('######Total Number of lines', lines)
    with open(output_dir + '/output_lines.txt', 'w') as f:
        f.write(f'Ciao, le righe sono: {lines}')