# A data-science pipeline to enable the Interpretability of Many-Objective Feature Selection (MOFS)
In this repository, you will find the codebase for the experiments we used to test our proposed methodology to solve the interpretability shortfall of MOFS.
## Prerequisites
To run the scripts, prerequisite libraries in **requirements.txt** file must be installed. Before doing so, I recommend setting up a [Virtual Environment (VE)](https://docs.python.org/3/library/venv.html) for this experiment, after which you install the required libraries in the VE with the following command:
```
pip install -r requirements.txt
```
## Four objectives
Three arguments are required to execute the ```four_objectives.py``` script:
- file path: this is the path to the file for execution.
- target feature: the name of the target feature.
- classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
To execute the script, you can just use the command below with the arguments specified.
```
python four_objectives.py [file path] [target feature] ["dt"|"lr"]
```
## Six objectives
In addition to the three arguments needed above, the ```six_objectives.py``` script requires a fourth one:
- sensitive feature: this is the sensitive attribute on which fairness should be achieved.
To execute the script, you can just use the command below with the arguments specified.
```
python six_objectives.py [file path] [target feature] ["dt"|"lr"] [sensitive feature]
```
## Report for Eight datasets
Besides the two use cases presented in the paper, we also worked with eight datasets and four objectives (subset size, accuracy, F1-score, and VIF). Following our proposed methodology for interpretability, we went ahead and created the dashboard below to present the results. We invite you to interact with it (click on the image below).
[![Report on dashboard](https://github.com/F-U-Njoku/many-objective-fs-nsgaiii/blob/main/dashboard.jpg)](https://lookerstudio.google.com/u/0/reporting/f254a8cb-39f5-40db-a0d9-0da9d07e0589/page/B53hD)
