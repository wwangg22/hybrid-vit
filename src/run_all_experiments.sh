# Batch script to run all experiments for the hybrid ViT comparison

echo "=========================================="
echo "Starting Hybrid ViT Experiments"
echo "=========================================="

# MNIST Experiments
echo ""
echo "=========================================="
echo "MNIST Experiments"
echo "=========================================="

echo ""
echo "Training Vanilla ViT on MNIST (Baseline)..."
python train.py --config configs/vanilla_vit_mnist.yml

echo ""
echo "Training Alternating ViT on MNIST..."
python train.py --config configs/alternating_vit_mnist.yml

echo ""
echo "Training Performer-First ViT on MNIST..."
python train.py --config configs/performer_first_vit_mnist.yml

echo ""
echo "Training Regular-First ViT on MNIST..."
python train.py --config configs/regular_first_vit_mnist.yml

# CIFAR-10 Experiments
echo ""
echo "=========================================="
echo "CIFAR-10 Experiments"
echo "=========================================="

echo ""
echo "Training Vanilla ViT on CIFAR-10 (Baseline)..."
python train.py --config configs/vanilla_vit_cifar10.yml

echo ""
echo "Training Alternating ViT on CIFAR-10..."
python train.py --config configs/alternating_vit_cifar10.yml

echo ""
echo "Training Performer-First ViT on CIFAR-10..."
python train.py --config configs/performer_first_vit_cifar10.yml

echo ""
echo "Training Regular-First ViT on CIFAR-10..."
python train.py --config configs/regular_first_vit_cifar10.yml

echo ""
echo "=========================================="
echo "nb_features Ablation Study (MNIST)"
echo "=========================================="

echo ""
echo "Training Alternating ViT with nb_features=64..."
python train.py --config configs/alternating_vit_mnist_nb64.yml

echo ""
echo "Training Alternating ViT with nb_features=128..."
python train.py --config configs/alternating_vit_mnist.yml

echo ""
echo "Training Alternating ViT with nb_features=256..."
python train.py --config configs/alternating_vit_mnist_nb256.yml

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""