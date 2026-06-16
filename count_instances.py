import os
from collections import defaultdict

def count_instances(label_dir):
    total_instances = 0
    class_counts = defaultdict(int)

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
                total_instances += len(lines)

                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    return total_instances, dict(class_counts)


# Paths
train_labels = "split_ordered_ori_dataset/train/labels"
valid_labels = "split_ordered_ori_dataset/valid/labels"

# Count
train_total, train_classes = count_instances(train_labels)
valid_total, valid_classes = count_instances(valid_labels)

# Print results
print("=== TRAIN SET ===")
print("Total instances:", train_total)
print("Per class:", train_classes)

print("\n=== VALID SET ===")
print("Total instances:", valid_total)
print("Per class:", valid_classes)

print("Total instances:", train_total + valid_total)