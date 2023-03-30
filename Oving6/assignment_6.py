import numpy as np
from pathlib import Path
from typing import Tuple
import random
import math


class Node:
    """ Node class used to build the decision tree"""

    def __init__(self, children={}, parent=None, attribute=None, value=None):
        self.children = children
        self.parent = parent
        self.attribute = attribute
        self.value = value

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    # TODO implement the importance function for both measure = "random" and measure = "information_gain"

    # Allocate a random number as importance to each attribute.
    if measure == "random":
        # Picks a random number to be used as index for the most important attribute
        return random.randint(0, len(attributes) - 1)

    if measure == "information_gain":
        # Counts the number of times type 1 and 2 occurs in examples respectively
        total_ones = list(examples[:, -1]).count(1)
        total_twos = list(examples[:, -1]).count(2)

        # Probability of each outcome
        P_one = total_ones/(total_ones + total_twos)
        P_two = total_twos/(total_ones + total_twos)

        # Calulate total entropy for examples
        entropy = P_one * math.log2(P_one) + P_two * math.log2(P_two)

        # Empty dict to hold all pairs of attributes and their information-gain value
        importanceDict = {}

        # For every attribute insert the attribute and its corresponding information-gain into importanceDict
        for var in attributes:
            importanceDict[var] = gain(var)
        maxGain = np.argmax(list(importanceDict.values()))

        # For every attribute-Gain pair return the attribute-index whose key is maxGain
        for key, value in importanceDict.items():
            if value == maxGain:
                return key


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node's parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # If examples is empty then return PLURALITY-VALUE(parent examples)
    if len(examples) == 0:
        node.value = plurality_value(parent_examples)
        return node

    # If all examples have the same classification then return the classification
    arrayToCheck = np.unique(examples[:, -1])
    if len(arrayToCheck) == 1:
        node.value = arrayToCheck[0]
        return node

    # If attributes is empty then return PLURALITY-VALUE(examples)
    if len(attributes) == 0:
        node.value = plurality_value(examples)
        return node

    # A ← argmax of IMPORTANCE (a, examples) where a ∈ attributes
    importantAttribute = importance(attributes, examples, measure)

    # Initializes a new decision tree with root test A
    node.attribute = importantAttribute

    # for each value v of A do
    for value in np.unique(examples[:, importantAttribute]):

        # exs ← {e : e ∈ examples and e.A = v}
        exs = np.array(
            [example for example in examples if example[importantAttribute] == value])

        # attributes - A
        attributes = np.delete(attributes, importantAttribute)

        # subtree ← LEARN-DECISION-TREE(exs, attributes - A, examples)
        subtree = learn_decision_tree(
            exs, attributes, examples, node, value, measure)

        # add a branch to tree with label (A = v) and subtree subtree
        node.children[value] = subtree

    return node


def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test


if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    # measure = "information_gain"
    measure = "random"
    attributes = np.arange(0, train.shape[1] - 1, 1, dtype=int)

    tree = learn_decision_tree(examples=train, attributes=attributes,
                               parent_examples=None, parent=None, branch_value=None, measure=measure)

    # print(f"Training Accuracy {accuracy(tree, train)}")
    # print(f"Test Accuracy {accuracy(tree, test)}")
