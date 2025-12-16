from datasets import load_dataset

# Download and load MNIST
dataset = load_dataset("ylecun/mnist")

# Access the splits
train_data = dataset['train']
test_data = dataset['test']

# Check what you have
print(dataset)
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Look at a sample
print(train_data[0])