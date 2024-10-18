import os

model_dir = os.environ['SM_MODEL_DIR']

with open(model_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono il modello di ml')
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono i log del training')

input_dir = os.environ['SM_OUTPUT_DIR']
with open(input_dir + '/train.csv', 'r') as fp:
    lines = len(fp.readlines())
    print('######Total Number of lines', lines)