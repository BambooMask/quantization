# ternary w

# eight networks

### paper: [ternary weight network](arxiv/1605.04711v2)

## base idea

- extend binary weight network to [-1,0,1], enhancing the ability of network with x16 or x32 compression
- quantize weight off line by minimizing the difference of quantized weight and float point weight

$$
\alpha, W = min_{\alpha, W} J(\alpha, W^t) = ||W-\alpha W^t|| \\
\alpha > 0, W^t_i \in \{-1,0,1\}
$$

​	solving such problem, wo get
$$
\delta = \frac{0.7}{n}\sum {|W_i|}\\
W^t_i = 1, W_i > \delta \\
W^t_i = 0, -\delta < W_i < \delta \\
W^t_i = -1, W_i < \delta \\
\alpha = \frac{<W^t, W>}{<W^t,W^t>}
$$

## cifar10

- network: resnet20
- optimizer: SGD + momentum(0.9)
- lr_scheduler: CosineAnnealing

| ID   | quan_lr | constr_w | constr_a | lr_base | epoch | test_acc | drop_acc |
| ---- | ------- | -------- | -------- | ------- | ----- | -------- | -------- |
| 1    | no      | [-1,0,1] | none     | 0.01    | 90    | 89.79    | -0.23    |
| 2    | yes     | [-1,0,1] | none     | 0.01    | 90    | 89.60    | -0.04    |

- quan_lr:  quantize first and last layer
- constr_w: quantize weight to
- constr_a: quantize activation to
- lr_base: base learning rate

training process for weight quantization with or without quantizing first and last layer

![twn_resnet20_quan_weight](.\image\twn_resnet20_quan_weight.png)

- unstable when learning rate is larger. because quantized weight is computed off line. when learning rate is large, weight change quickly, so the ternary weight and alpha. so the performance will fluctuates.
- perfect quantization for this network.







