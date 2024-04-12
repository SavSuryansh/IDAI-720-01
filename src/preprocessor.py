from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)

    # Calculate P(y) as the probability of each class in Y
    count_y = Counter(Y)
    prob_y = {cls: cnt / len(Y) for cls, cnt in count_y.items()}

    # Combine sensitive attributes with Y for easier computation
    data_sensitive = X[A]
    data_sensitive['Y'] = Y  # Add Y temporarily to the DataFrame for calculation purposes

    # Initialize the array for sample weights
    sample_weight = np.zeros(len(Y))

    # Calculate probabilities and weights for each data point
    for idx, data_row in data_sensitive.iterrows():
        # Create a tuple of attribute values for the current row
        attr_values = tuple(data_row[attr] for attr in A)
        class_label = data_row['Y']

        # Calculate P(a) as the product of probabilities of each attribute value
        prob_a = np.prod([Counter(data_sensitive[attr])[data_row[attr]] / len(Y) for attr in A])

        # Calculate P(a, y) as the joint probability of attributes and class label
        filter_condition = (data_sensitive[A] == pd.Series(attr_values, index=A)).all(axis=1) & (
                data_sensitive['Y'] == class_label)
        prob_ay = len(data_sensitive[filter_condition]) / len(Y)

        # Compute the weight for the current data point
        sample_weight[idx] = prob_y[class_label] * prob_a / prob_ay

    # Clean up by removing the temporary 'Y' column
    data_sensitive.drop(columns=['Y'], inplace=True)
    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
