# Workshop Activity on Genetic Algorithm and Neural Network 
### NAME: MANIKANDAN R
### REG NO: 212222220022
### AIM 
To study and implement the concepts of Genetic Algorithm for optimization and Neural 
Network for pattern recognition using Python. 
### OBJECTIVE 
 To understand how Genetic Algorithms mimic the process of natural selection. 
 To explore how Neural Networks learn from data through training and weight 
adjustment. 
 To implement basic programs demonstrating both techniques. 
 To compare how both methods solve computational intelligence problems. 
### PROGRAM 
a) Genetic Algorithm – Find Maximum of a Function 

import random 

def fitness(x): 

 return x * x # maximize x^2 
 
population = [random.randint(-10, 10) for _ in range(6)] 

for generation in range(5): 

 fitness_scores = [fitness(x) for x in population] 
 
 parents = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:2] 
 
 new_population = [parents[0][0], parents[1][0]] 
 
 while len(new_population) < 6: 
 
 p1, p2 = random.choice(parents)[0], random.choice(parents)[0] 
 
 child = (p1 + p2) // 2 
 
 if random.random() < 0.3: 
 
 child += random.randint(-2, 2) 
 
 new_population.append(child) 
 
 population = new_population 
 
 print(f"Generation {generation+1}: {population}") 
 
best = max(population, key=fitness) 

print("Best solution:", best) 

### Neural Network – Simple XOR Problem using TensorFlow 
import tensorflow as tf 

import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]]) 

Y = np.array([[0],[1],[1],[0]]) 

model = tf.keras.Sequential([ 

 tf.keras.layers.Dense(4, input_dim=2, activation='relu'), 
 
 tf.keras.layers.Dense(1, activation='sigmoid') 
]) 


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(X, Y, epochs=500, verbose=0) 

print("Predictions:") 

print(model.predict(X)) 


### OUTPUT

Genetic Algorithm Output (Sample):

<img width="846" height="295" alt="image" src="https://github.com/user-attachments/assets/8c02fcc9-b239-4816-9f87-067cde70bb09" />


### Neural Network Output (Sample):
<img width="840" height="161" alt="image" src="https://github.com/user-attachments/assets/0c4fe146-83ac-4d45-bc2b-f85c7ebc849b" />


### RESULT

The Genetic Algorithm successfully optimized the function by evolving better solutions over 
generations, while the Neural Network accurately learned the XOR logic pattern after training.
Thus, the experiment demonstrates the eƯ ectiveness of both evolutionary computation and 
machine learning techniques for problem-solving.
