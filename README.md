Question 1:
Using Java, please develop a neural network that mimics the behavior of an AND boolean gate. Weights, biases, and the activation function must all be implemented by you. Do not use any libraries for the functionality of the neural net; you implement it from scratch. You need to answer the following questions: 
a) How many hidden layers have you used?
b) What are the weights and biases of each node?
You should provide a simple driver to test it.

Question 2:
Includes back propagation 
Using Python and any of the libraries it provides, please implement a feed-forward neural network for the following training set:
| A | B | C | D | E |                   output                      |
| 1 | 1 | 1 | 0 | 0 |                          1                          |
| 1 | 1 | 1 | 0 | 1 |                          1                          |
| 0 | 0 | 0 | 1 | 0 |                          0                          |
| 0 | 0 | 0 | 1 | 1 |                          0                          |
| 0 | 0 | 1 | 0 | 0 |                          1                          |
| 0 | 0 | 1 | 0 | 1 |                          1                          |
| 0 | 0 | 1 | 1 | 0 |                          1                          |
| 0 | 0 | 1 | 1 | 1 |                          1                          |
| 0 | 1 | 0 | 0 | 0 |                          0                          |
| 0 | 1 | 0 | 0 | 1 |                          0                          |
| 0 | 1 | 0 | 1 | 0 |                          0                          |
| 0 | 1 | 0 | 1 | 1 |                          0                          |
| 0 | 1 | 1 | 0 | 0 |                          1                          |
| 0 | 1 | 1 | 0 | 1 |                          1                          |
| 0 | 1 | 1 | 1 | 0 |                          1                          |
| 0 | 1 | 1 | 1 | 1 |                          1                          |
| 1 | 0 | 0 | 0 | 0 |                          1                          |
| 1 | 0 | 0 | 0 | 1 |                          1                          |
| 1 | 0 | 0 | 1 | 0 |                          1                          |
| 1 | 0 | 0 | 1 | 1 |                          1                          |
| 1 | 0 | 1 | 0 | 0 |                          1                          |
| 1 | 0 | 1 | 0 | 1 |                          1                          |
| 1 | 0 | 1 | 1 | 0 |                          1                          |
| 1 | 0 | 1 | 1 | 1 |                          1                          |

Use the following testing set to check how good the neural net is in classification:
| A | B | C | D | E |                   output                      |
| 1 | 1 | 0 | 0 | 0 |                          1                          |
| 1 | 1 | 0 | 0 | 1 |                          1                          |
| 0 | 0 | 0 | 0 | 0 |                          0                          |
| 0 | 0 | 0 | 0 | 1 |                          0                          |
| 1 | 1 | 0 | 1 | 0 |                          1                          |
| 1 | 1 | 0 | 1 | 1 |                          1                          |
| 1 | 1 | 1 | 1 | 0 |                          1                          |
| 1 | 1 | 1 | 1 | 1 |                          1                         |

You need to think of/ examine / contemplate about / consider / meditate about 😊 / answer the following questions:
a) How many hidden layers have you used? And why?
b) How many nodes in each hidden layer and why that number of nodes in particular?
c) What is the activation function that you used and why? Did you use the same activation function in all layers? Why?
d) What learning algorithm did you use to train the neural net and why?
e) Can you use one hidden layer only to solve this problem? If yes, how many nodes are you going to have in it? And why?
f) Can we use 5 hidden layers? Is that a good idea? Justify your answer.
g) How did the neural net do in classifying the testing set? Comment on how good or bad it learned the function from the training set.
