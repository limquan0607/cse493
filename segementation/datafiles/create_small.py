import json
import random

# Load the dataset JSON
with open('/home/nguye/project/LungTumor-Segmentation-main/dataset/Task06_Lung/dataset.json', 'r') as file:
    dataset = json.load(file)

# Get the training data
training_data = dataset['training']

# Randomly select 1/5 of the training data
num_samples = len(training_data)
num_samples_to_select = num_samples // 5
random_samples = random.sample(training_data, num_samples_to_select)

# Create a new dataset JSON with the selected samples
new_dataset = {
    'name': dataset['name'],
    'description': dataset['description'],
    'reference': dataset['reference'],
    'licence': dataset['licence'],
    'relase': dataset['relase'],
    'tensorImageSize': dataset['tensorImageSize'],
    'modality': dataset['modality'],
    'labels': dataset['labels'],
    'numTraining': num_samples_to_select,
    'numTest': dataset['numTest'],
    'training': random_samples,
    'test': dataset['test']
}

# Write the new dataset JSON to a file
with open('/home/nguye/project/LungTumor-Segmentation-main/dataset/SmallDataSet/dataset.json', 'w') as file:
    print("Saving json file in new dataset")
    json.dump(new_dataset, file, indent=4)
