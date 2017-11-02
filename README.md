# Higher-order Integration of Hierarchical Convolutional Activations for Fine-grained Visual Categorization
The implementation of “Higher-order Integration of Hierarchical Convolutional Activations for Fine-grained Visual Categorization” by Sijia Cai, Wangmeng Zuo and Lei Zhang.

## Environment
- Ubuntu 14.04
- NVIDIA GeForce GTX TITAN X
- MatConvNet (v1.0-beta20)

## Usage
- First you need to download the FGVC datasets (**[CUB][1]**, **[Aircraft][2]** 
	and **[Cars][3]**) and unzip them into the folder “./datasets”
- Download the pretrained **[VGG16][4]** model and put it into the folder “./models/pretrainedmodels”
- Compile MatConvNet with the GPU and cuDNN support
- Train: run hihcamain.m by setting runPhase=train
- Test: run hihcamain.m by setting runPhase=test

## Trained models and visualization
Coming soon …

## Citation
If you find the codes of this repository useful, please cite the following paper:

```
`@inproceedings{cai2017higher,
  title={Higher-Order Integration of Hierarchical Convolutional Activations for Fine-Grained Visual Categorization},
  author={Cai, Sijia and Zuo, Wangmeng and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={511--520},
  year={2017}
}
```
`
## Contact
For any question, please contact
```
`cssjcai@gmail.com
````

[1]:	http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[2]:	http://www.robots.ox.ac.uk/%5C~vgg/data/fgvc-aircraft/
[3]:	http://ai.stanford.edu/%5C~jkrause/cars/car%5C_dataset.html
[4]:	http://www.vlfeat.org/matconvnet/pretrained/