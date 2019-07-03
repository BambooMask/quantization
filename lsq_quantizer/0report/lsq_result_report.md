# learned step size quantization

### paper:  [learned step size quantization](arxiv/1902.08153v1)

## cifar10 

* network: resnet20
* optimizer: SGD + momentum(0.9)
* lr_scheduler: CosineAnnealing

| ID   | quan_lr | constr_w | constr_a  | lr_base | epoch | test_acc | drop_acc |
| ---- | ------- | -------- | --------- | ------- | ----- | -------- | -------- |
| 1    | no      | [-1,0,1] | none      | 0.01    | 90    | 89.69    | -0.13    |
| 2    | yes     | [-1,0,1] | none      | 0.01    | 90    | 89.55    | -0.01    |
| 3    | no      | [-1,0,1] | [0,1,2,3] | 0.01    | 90    | 87.68    | 1.88     |
| 4    | yes     | [-1,0,1] | [0,1,2,3] | 0.01    | 90    | 87.24    | 2.32     |

- quan_lr:  quantize first and last layer
- constr_w: quantize weight to
- constr_a: quantize activation to
- lr_base: base learning rate

- training process for weight quantization with or without quantizing first and last layer

  ![resnet20_quan_weight_test_acc](.\image\resnet20_quan_weight_test_acc.png)

- training process for all quantization with or without quantizing first and last layer

![resnet20_quan_all_test_acc](.\image\resnet20_quan_all_test_acc.png)

-  because the mission is easy. the influence of first and last layers' quantization and activations' quantization is not obvious in this dataset.
- for weight quantization, the model achieves better performance after quantization. this is because quantization is a way to enhance model's generalization.

##  imagenet

- network: resnet18
- optimizer: SGD + momentum(0.9)
- lr_scheduler: CosineAnnealing

| ID   | quan_lr | constr_w | constr_a  | lr_base | epoch | acc_top1 | acc_top5 | drop_top1 | drop_top5 |
| ---- | ------- | -------- | --------- | ------- | ----- | -------- | -------- | --------- | --------- |
| 1    | no      | [-1,0,1] | none      | 0.01    | 90    | 69.43    | 88.69    | 0.26      | 0.38      |
| 2    | yes     | [-1,0,1] | none      | 0.01    | 90    | 67.43    | 87.70    | 2.27      | 1.37      |
| 3    | no      | [-1,0,1] | [0,1,2,3] | 0.01    | 90    | 64.24    | 85.47    | 5.46      | 3.53      |
| 4    | yes     | [-1,0,1] | [0,1,2,3] | 0.01    | 90    | 61.49    | 83.65    | 8.21      | 5.35      |

- quan_lr:  quantize first and last layer
- constr_w: quantize weight to
- constr_a: quantize activation to
- lr_base: base learning rate

- for some reasons, the training process can't be put here. wait to update
- for imagenet dataset, we can see the influence of first and last layers' quantization.
- for just weight quantization, we can achieve comparable performance with float point weight.
- for quantizing weight and activation with first and last layers, there is a significant gap between float point model and quantized model. this is not be showed in paper.

## analysis

- quantization performance factors
  - scale initial value: not to big, keep enough nozero weight
  - scale learning rate: down when to big
  - lr_scheduler: influence on final result 

 













