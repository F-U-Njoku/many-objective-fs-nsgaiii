import random
import time
import math
import openml as oml
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.util.solution import get_non_dominated_solutions
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from jmetal.operator import SBXCrossover, PolynomialMutation, SPXCrossover, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime

le = LabelEncoder()


def fetch_data(number):
    dataset = oml.datasets.get_dataset(number)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute)
    return X, y, attribute_names, len(attribute_names)


def get_no_evals(features, subset):
    # no_evals = 0
    # for i in range(subset + 1):
    #    no_evals += (features - i)
    no_evals = math.ceil((0.5 * features * (features + 1)) * 0.5)
    return no_evals


def get_pop_size(n):  # n is the number of feature
    upper = 1 + (math.log(1 - (0.99 ** (1 / n)), 10) / math.log(0.5, 10))
    if math.floor(upper) % 2:
        return math.floor(upper) - 1
    else:
        return math.floor(upper)


def get_metric(classifier, X, y):
    if y.nunique() == 2:
        scores = cross_val_score(classifier, X=X, y=y, scoring="f1", cv=10).mean()
    else:
        scores = cross_val_score(classifier, X=X, y=y,
                                 scoring=make_scorer(f1_score, average='macro'), cv=10).mean()
    return scores


class featSel(BinaryProblem):

    def __init__(self, x: pd.DataFrame, y: pd.Series, pop_size: int, classifier):

        super(featSel, self).__init__()
        self.F = x
        self.target = y
        self.number_of_bits = x.shape[1]
        self.number_of_objectives = 4
        self.number_of_variables = 1
        self.number_of_constraints = 1
        self.obj_labels = ["subset size", "accuracy", "f1", "vif"]
        self.count = 0
        self.pop_size = pop_size
        self.classifier = classifier
        idx, _, _ = MRMR.mrmr(self.F.values, le.fit_transform(self.target), n_selected_features=self.pop_size)
        self.rank = idx.tolist()

    def evaluate(self, solution: BinarySolution) -> BinarySolution:

        idx = []

        self.__evaluate_constraints(solution)
        # check constraint
        if solution.constraints[0] == 0:
            check = random.randrange(self.number_of_bits)
            solution.variables[0] = [True if _ == check
                                     else False for _ in range(self.number_of_bits)]

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                idx.append(self.F.columns[index])
        accuracy = cross_val_score(estimator=self.classifier, X=self.F[idx], y=self.target,
                                   scoring="balanced_accuracy", cv=10).mean()
        if self.target.nunique() == 2:
            f1 = cross_val_score(estimator=self.classifier, X=self.F[idx], y=self.target,
                                 scoring="f1", cv=10).mean()
        else:
            f1 = cross_val_score(estimator=self.classifier, X=self.F[idx], y=self.target,
                                 scoring=make_scorer(f1_score, average='macro'),
                                 cv=10).mean()
        # VIF dataframe
        if len(self.F[idx].columns) < 2:
            vif_score = 0
        else:
            vif_data = pd.DataFrame()
            vif_data["feature"] = self.F[idx].columns
            # calculating VIF for each feature
            vif_data["VIF"] = [vif(self.F[idx].values, i) for i in range(len(self.F[idx].columns))]
            vif_score = vif_data["VIF"].mean()

        solution.objectives[0] = sum(solution.variables[0])
        solution.objectives[1] = accuracy * -1.0
        solution.objectives[2] = f1 * -1.0
        solution.objectives[3] = vif_score 

        if self.count < self.pop_size:
            # print(solution.objectives[0], ",", solution.objectives[1], ",", solution.objectives[2], ",",
            #       solution.get_binary_string())
            print(*solution.objectives, solution.get_binary_string(), sep=",")
            self.count += 1

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = [True if _ == self.rank[-1] else False for _ in range(self.number_of_bits)]

        self.rank.pop()

        return new_solution

    def __evaluate_constraints(self, solution: BinarySolution) -> None:
        solution.constraints = [0 for _ in range(self.number_of_constraints)]
        x = solution.variables
        if sum(x[0]) == 0:
            solution.constraints[0] = 0
        else:
            solution.constraints[0] = 1

    def get_name(self) -> str:
        return 'Feature selection'


def experiment(data_spec, classifier):
    run_times = []

    X, y, attribute_names, feat = fetch_data(data_spec[0])
    try:
        X = pd.DataFrame(X, columns=attribute_names)
    except:
        X = pd.DataFrame.sparse.from_spmatrix(data=X, columns=attribute_names)
    y = pd.Series(y)
    X = X.dropna()

    if data_spec[0] == 41142:
        X.drop(columns=X.columns[X.nunique() < 100], inplace=True)
        attribute_names = X.columns
    # Select 3000 instances
    if data_spec[0] == 273:
        random.seed(1111)
        x_idx = random.sample(range(X.shape[0]), 1334)
        X = X.iloc[x_idx, :]
        attribute_names = X.columns
        y = y[x_idx]
    if data_spec[0] == 40666:
        X.drop(columns=["molecule_name", "conformation_name"], inplace=True)
        attribute_names = X.columns

    population_size = get_pop_size(X.shape[1])
    max_evaluations = get_no_evals(X.shape[1], data_spec[1])
    if population_size >= max_evaluations:
        max_evaluations = population_size + 1

    problem = featSel(X, y, population_size, classifier)

    print("Data ID", ",", str(data_spec[0]))
    print("Features", ",", str(X.shape[1]))
    print("Features", ",", str(X.shape[0]))

    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=BitFlipMutation(probability=(1 / X.shape[1])),
        crossover=SPXCrossover(probability=1),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    print("Initial population")
    for run in range(1):
        start = time.time()
        algorithm.run()
        end = time.time()
        solutions = get_non_dominated_solutions(algorithm.get_result())

        run_times.append(round((end - start), 4))

    print("Final population")
    final_sorted = []
    for i in range(len(solutions)):
        subset = []
        for index, bits in enumerate(list(map(int, solutions[i].get_binary_string()))):
            if bits:
                subset.append(X.columns[index])

        selected = [attribute_names[j] for j in range(len(solutions[i].get_binary_string())) \
                    if solutions[i].get_binary_string()[j] == "1"]
        selected = "; ".join(selected)

        current_solution = solutions[i].objectives
        current_solution.append(selected)

        final_sorted.append(tuple(current_solution))

    final_pop = sorted(final_sorted)
    for ind in final_pop:
        print(*ind, sep=",")
    print("Time", ",", str(np.mean(run_times)))


if __name__ == '__main__':
    nb = GaussianNB()
    le = LabelEncoder()
    baseTree = DecisionTreeClassifier(random_state=0)
    knn = KNeighborsClassifier()
    for value in [[40666, 168]]:
        experiment(value, nb)
