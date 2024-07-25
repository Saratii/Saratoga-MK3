# Saratoga MK3 https://github.com/Saratii/Saratoga-MK3
Saratoga MK3 is a Java tool for Artificial Intelligience Architecture and Deep Learning focused on neural networking and image classification.

It is modeled off of two popular Python libraries: Pytorch and Tensorflow.
Saratoga MK3 does not have as many tools as these libraries have; however, this is arguably easier to use, easier to run, and much more configurable. It is perfect for someone who wants to understand the ins and outs of how AI works instead of just importing a train function from Tensorflow.


Saratoga MK3 is built and tested on Java 19.0.2.
The challenging aspect of this project is that no machine learning or linear algebra libraries were installed, and all the math and algorithms were written and derived by hand.
It features multivariable vector calculus and differential linear algebra along with many optimization algorithms, including SIMD operations and parallel processing over the CPU.


Main.java features an already-built model with working parameters and sizes. The hyperparameters and model can be changed, and they are fully abstracted and modular. The modules of the model are correct. If you receive an exception or error while messing with the model or hyperparameters, it is because only certain combinations of sizes are mathematically possible.
I.e., dense layer inputs must match the size of the previous dense layer outputs. The number of threads for multi-threading can be configured to fit your hardware. The hyperparameters in main can be configued in a way that you may test image classification (which is fast) without having to retrain the model (which is slow). The program automatically writes architecture structure and parameter values to a txt file with a build regex. 
Saratoga MK3 also utilizes python to show a training graph which is often a useful tool to see model performance. 

```
cd src 
python graph.py
```


Images can be loaded by YOU into the model by creating folders in the root directory and then just changing the strings for where the folders are loaded. Currently, the model is loaded with 600 images of two classes each. It works for as many classes as you need. It is untested on colored 3-4 channel images or 3 dimensional images (9-12 channel) but the code supports it. 


AI works by fitting an extremely complex math equation with parameters to output the correct class over many iterations and images. The parameters are updated by calculating multi-dimensional gradients and Jacobians for the parameter with respect to loss. The larger the model is and the larger the input data, the longer the runtime per epoch will be. Do not expect it to finish quickly, as it is computing an obscene amount of matrix operations. If you need more memory, you may change the Java JVM memory parameters in preferences or by using the command line. With my very limited testing, I was able to fit a model of 100 classes of animal with 95% accuracy. Train time was several days. Gradient values and layer output equations were tested for accuracy and performance against Pytorch. Accuracy was perfect and output the same values as pytorch. Performance in most cases was far better than pytorch due to pytorch having bloat and slow Auto Grad (on the cpu). 


I plan to continue expanding the functionality and performance.
I also plan to port it to Rust or C++.


Sources for the math: Although all gradients were calculated by hand and all equations were derived by hand.
https://arxiv.org/pdf/1502.01852.pdf
https://www.sciencedirect.com/science/article/abs/pii/092523129390006O
https://iopscience.iop.org/article/10.1088/1742-6596/1004/1/012028/meta
https://ieeexplore.ieee.org/abstract/document/8462533
https://www.nature.com/articles/nature14539
https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
Images from https://www.kaggle.com/datasets/alessiocorrado99/animals10 



