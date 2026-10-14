# LABORATORY REPORT: LAB 02

## A Neural Network from Scratch (Chapter 3)

* **Student Name:** Khalid Chalabi
* **Student ID:** 73716
* **Course:** Machine Learning Systems / Artificial Intelligence
* **Institution:** Akademia Finansów i Biznesu Vistula w Warszawie
* **Instructor:** mgr. inż. Kumar Nalinaksh

---

## 1. Introduction and Dataset Preparation

The objective of this laboratory session was to design, implement, and evaluate a two-hidden-layer neural network completely from scratch using Python and `NumPy`, without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

* **Dataset:** We utilized the `load_digits` dataset from scikit-learn, which consists of $8 \times 8$ pixel grayscale images of handwritten digits (64 input features per sample).


* **Preprocessing:** Input feature values (ink brightness) were scaled to the range $[0, 1]$ by dividing by 16.0. Target labels were converted into one-hot encoded vectors across 10 classes (digits 0 through 9).


* **Data Splitting:** The dataset was split into 70% training material and 30% test exam data using stratified sampling with `random_state = 42` to ensure balanced class distribution.



---

## 2. Network Architecture & Mathematical Design

The network architecture features two hidden layers followed by an output softmax classification layer:

* **Input Layer:** 64 features.


* **Hidden Layer 1:** Width $W$ (tested with $W = 16$ and $W = 64$), activated via the Rectified Linear Unit (ReLU) function.


* **Hidden Layer 2:** Width $W$, activated via ReLU.


* **Output Layer:** 10 nodes corresponding to the digit classes, activated via Softmax to yield probability distributions summing to 1.0.


* **Parameter Shapes & Initialization:** Weights were initialized from a normal distribution $N(0, 0.01)$.


* Layer 1 weights: $(64, W)$, biases: $(W,)$.


* Layer 2 weights: $(W, W)$, biases: $(W,)$.


* Output layer weights: $(W, 10)$, biases: $(10,)$.





---

## 3. Forward Propagation and Loss Function

* **Forward Pass:** Data flows sequentially through linear transformations ($z = XW + b$) and non-linear ReLU activations, culminating in Softmax scores for the output layer. Verification confirmed that output shapes matched `(batch_size, 10)` and every row's probability sum equaled 1.0 within a tolerance of $1 \times 10^{-6}$.


* **Cross-Entropy Loss:** Implemented to penalize incorrect probability predictions. A sanity check on an untrained, randomly initialized network confirmed that the initial loss approximated the expected theoretical value of $\ln(10) \approx 2.3026$.



---

## 4. Backpropagation & Numerical Gradient Verification (Lie Detector)

To ensure that analytical gradients derived via the chain rule were correct, a numerical gradient check was conducted:

* 10 random weight parameters were nudged by $\pm 1 \times 10^{-5}$.


* Analytical gradients were compared against numerical finite-difference approximations.


* **Result:** The relative error remained below $1 \times 10^{-6}$ for all tested weights, successfully passing the gradient check ($\ge 9/10$ required).



---

## 5. Model Training & Hyperparameter Configurations

The network was trained over 50 epochs using mini-batches of size 32 with randomized data shuffling per epoch. Training was executed across 6 distinct workshop configurations combining hidden widths ($W \in \{16, 64\}$) and learning rates ($\text{lr} \in \{0.001, 0.01, 0.1\}$):

* Loss trajectories per epoch were tracked and successfully saved to `results/loss_curves.png`.


* Final test set accuracy was recorded for each configuration. Higher learning rates ($\text{lr} = 0.1$) and larger widths ($W = 64$) yielded faster convergence and superior classification accuracy.



---

## 6. Computational Timing Analysis (Stopwatch Duel)

A performance benchmark measuring the execution time over 100 iterations of forward passes versus backward passes revealed:

* **Forward Pass Speed:** Faster execution time as it primarily consists of sequential matrix multiplications ($XW$) and element-wise activation functions.


* **Backward Pass Speed:** Slower execution time due to the computation of multiple transposed matrix multiplications, derivative chains across multiple layers, and gradient updates. This demonstrates that backpropagation is computationally heavier than forward inference.



---

## 7. Conclusion

Lab 02 successfully demonstrated the fundamental mechanics of deep learning systems from scratch. By implementing matrix-level forward passes, cross-entropy optimization, backpropagation calculus, and numerical validation, we gained a robust understanding of how neural networks learn patterns from raw image data. All required evidence (gradient verification, loss curve graphs, and timing analysis) was successfully generated and verified.