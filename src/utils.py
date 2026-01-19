# define single qubit pair remote gate probability
p = 0.7
def compute_probability(n,p):
    return 1 - (1-p)**(n)
def compute_probability_with_distance(n,p, distance):
    total_p = (p**distance)
    return 1 - (1-total_p)**(n)