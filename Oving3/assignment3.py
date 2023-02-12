import numpy as np

def forward(transition_matrix, observation_matrices, evidences):
  """ Computes the probability for a state given an evidence """
  
  # The first day has a 50/50 chance for rainy/sunny
  probability = np.array([0.5, 0.5])

  # Empyt list that eventually will hold all normalized forward-messages
  # as vectors of probabilities in a given inference situation
  forwards = []

  for i in range(len(evidences)):

    # For each iteration the underneath code updates the predicted probability
    # from the formula f_t:n = alpha * O * T^T * f_t-1 where 
    # O is the Observation-Matrix, T is the Transition-Matrix, 
    # alpha is the normalizing term and f_t-1 is the prior probability

    # By default the observation matrix for when there is an umbrella will be used
    o_matrix = observation_matrices[0]

    # If there are no umbrellas then use the complementary matrix for observations 
    if evidences[i] == 0:
      
      # Updates the observation matrix to be used
      o_matrix = observation_matrices[1]
    
    # Computes the dot-product of two matrices with np.dot()
    temp_p_matrix = np.dot(o_matrix, np.transpose(transition_matrix))

    # Computes the vector-matrix product with np.dot()
    probability = np.dot(temp_p_matrix, probability)
    
    # Alpha is the normalizing term 
    alpha = 1 / sum(probability) 

    # Normalizes the probability so that the sum of probabilities adds up to 1. 
    probability = probability * alpha

    # Stores the normalized forward message for this iteration 
    forwards.append(probability)

  # Return both the filtered probability and the "steps" to get there as forward messages
  return probability, forwards

if __name__ == '__main__':
  
  # Transition matrix
  T = np.array([[0.7, 0.3], [0.3, 0.7]])

  # Observation matrix
  O = np.array([[0.9, 0], [0, 0.2]])

  # Complementary Observation matrix
  O_not = np.array([[0.1, 0], [0, 0.8]])

  # List with matrices for use when getting evidence = true or false
  matrices = [O, O_not]

  task = 2

  if task == 1:
    
    # Observation sequence for task 1
    observations = [1, 1]

    # Filtering process implemented with filtering operation
    p = forward(T, matrices, observations)
    
    print(f"{p[0]} is the probability for rain at day 2 given evidence of umbrella at day 1 and 2.")

  if task == 2:
    
    # Observation sequence for task 2
    observations = [1, 1, 0, 1, 1]

    # Filtering process implemented with filtering operation
    p = forward(T, matrices, observations)

    messages = p[1]

    print("------------------------------------")
    print(f"{p[0][0]} is the probability for rain at day 5 given the sequence of evidences.")
    print("------------------------------------")
    print("----------FORWARD MESSAGES----------")
    for message in messages:
        print("----------")
        print(message)