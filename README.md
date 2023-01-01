Basic DeepLearning model in Python that predicts numbers form handritten number dataset

It uses:
* Rectified Linear Unit (ReLU) as an activation function
* Softmax funtion to transform the output to a probability distribution. It is also differentiable, which means that it can be used in conjunction with optimization algorithms such as stochastic gradient descent (SGD) or Adam (Adaptive Moment Estimation)
* Adam optimizer. unlike SGD learning-rate is adaptive instead of fixed which results in a more rapid convergence rate
* Sparse categorical crossentropy as loss function which is best used when target labels are integers
* Sparse categorical accuracy metric which computes the prediction accuracy (best used with Sparse categorical crossentropy)
    
Output:
* 3 iterations resulted in accuracy ≈ 0.9762
    
P.S. Trained model is also inluded in source files as 'demo.model'
