for network in resnet18 resnet32 vgg16
do
echo "Number of ReLUs for" $network "...";
python count_relus.py --network=$network --dataset=cifar;
python count_relus.py --network=$network --dataset=tinyimagenet;
python count_relus.py --network=$network --dataset=imagenet;
done