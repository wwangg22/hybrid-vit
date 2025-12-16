echo ""
echo "Training Vanilla ViT on CIFAR-10 with Long Sequence..."
python train.py --config configs/vanilla_vit_cifar10_long_seq.yml

echo ""
echo "Training Alternating ViT on CIFAR-10 with Long Sequence..."
python train.py --config configs/alternating_vit_cifar10_long_seq.yml

echo ""
echo "Training Performer-First ViT on CIFAR-10 with Long Sequence..."
python train.py --config configs/performer_first_vit_cifar10_long_seq.yml