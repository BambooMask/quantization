# Repr training strategy

[RePr: Improved Training of Convolutional Filters](https://arxiv.org/abs/1811.07275v3)

1. define the network N
2. train using N
3. prune N to M using inter-filter orthogonality 
4. train using M
5. reinitialize back to N
6. repeat 2,3,4,5 for n times

discussion

1. Using Repr in the beginning stage,  not in final stage for stability
2. every time, the network shrinks to small size, the performance will drop a lot
3. don't use larger prune_ratio. It makes training stage very unstable 

training performance

![train_perf](.\image\train_perf.png)

the final performance on test set is 89.6, not better than normal training strategy from this trial