# Agile Power

Yao Lu, Qijun Zhang, and Zhiyao Xie, "Unleashing Flexibility of ML-based Power Estimators Through Efficient Development Strategies," ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED), 2024. [[paper]](https://zhiyaoxie.com/files/ISLPED24_AgileDev.pdf)

We propose AgileDevelop and AgileTransfer, efficient ML-based power model development strategies. The AgileDevelop minimizes the overhead involved in model development from scratch by coverage-based sampling, encompassing both signal space and power space. The AgileTransfer enables the transfer of existing models to updated design RTLs with negligible additional costs by generating pseudo labels through source design datasets and calibrating them with sampled ground truth power. 

The strategies significantly reduce the development overhead from 3 days to 1 hour while maintaining a high accuracy. Such lightweight development strategies lower the overhead of adopting ML-based power models by design teams, and thus AgileDevelop and AgileTransfer are compelling additions to the VLSI designers' toolbox.

## Introduction
The implementation mainly includes two parts:
1) AgileDevelop: the coverage-based sampling strategy that develops ML-based power models based on a tiny amount of labeled data.
2) AgileTransfer: the strategy to transfer the power model of a design to a new version of it. 

## Quick Start
We prepared an example dataset in "example_data". So you can just run the AgileDevelop and AgileTransfer, and the result will be visualized as figures in "result_figure". (The example dataset is coming soon!)

```
python AgileDevelop.py
python AgileTransfer.py
```

