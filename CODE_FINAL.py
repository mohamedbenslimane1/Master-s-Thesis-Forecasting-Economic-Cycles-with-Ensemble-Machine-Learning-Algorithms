import os
os.system('clear')
import pandas as pd
import numpy as np
import warnings
import sklearn



# CLassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor

#Performance
from sklearn.metrics import accuracy_score

# Cross validation
from sklearn.model_selection import GridSearchCV


import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import seaborn as sns
from mlxtend.evaluate import mcnemar_table, mcnemar

#Performance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error, precision_recall_curve, log_loss


def plot_predictions(y_label=None, y_probs: list = None, opt_threshold: list = None, names=None, acc: list = None):
    """
    Plots predicted probabilities or binary classifications of multiple models over time.

    Args:
    y_label (pd.Series): Actual binary classification labels for the target variable
    y_probs (list of pd.Series): Predicted probabilities or binary classifications for each model
    opt_threshold (list of float or None): "Optimal" classification threshold for each model.
        If None, default threshold of 0.5 is used.
    names (list of str): Name of each model to use as label for the plot
    acc (list of pd.Series) : Accuracy of the model imputing unknown Y's
        If not None plot the accuracy for each model

    Returns:
    None
    """

    # Create a plot with the specified size
    fig, ax = plt.subplots(figsize=(12, 8))

    # For each set of predicted probabilities or binary classifications
    for i, y_prob in enumerate(y_probs):
        # Plot the probabilities or classifications over time and add a label with the model name
        ax.plot(y_prob.index, y_prob, label=names[i])

        # If "optimal" classification threshold is not provided, set it to default threshold of 0.5
        if opt_threshold is None:
            threshold = 0.5
            # Draw a horizontal line at the default threshold for comparison
            ax.axhline(threshold, color='grey', lw=2, alpha=0.7)
        else:
            # If "optimal" classification threshold is provided, plot it as a line
            ax.plot(opt_threshold[i], color='grey', lw=2, alpha=0.7, label='"Optimal" Threshold')

    if acc is not None:
        for i, acc in enumerate(acc):
            ax.plot(acc.iloc[204:], label="Accuracy of predicting the last 11 missing Y's", linestyle="-")
            # Add point to accuracy plot


    # Shade areas on the plot where the actual labels are 1 or 0
    ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                    color='green', alpha=0.1, transform=ax.get_xaxis_transform())
    ax.fill_between(y_label.index, 0, 1, where=y_label == 0,
                    color='red', alpha=0.1, transform=ax.get_xaxis_transform())

    # Add a title and labels for the x and y axes
    ax.set_title('Predictions')
    ax.set_xlabel('Time')
    ax.legend(loc='best')
    plt.show()


def compute_sharpe_ratio(monthly_return_pct):
    """
    Compute Sharpe Ratio given monthly returns in percentage

    Args:
    - monthly_return_pct (pandas.Series): Monthly returns in percentage

    Returns:
    - sharpe_ratio (float): The Sharpe Ratio of the given monthly returns
    """

    # Calculate annualized average daily return
    avg_monthly_return = monthly_return_pct.mean()
    avg_annual_return = avg_monthly_return * 12  # 252 trading days in a year

    # Calculate annualized standard deviation of daily returns
    std_monthly_return = monthly_return_pct.std()
    std_annual_return = std_monthly_return * np.sqrt(12)

    # Calculate Sharpe Ratio : Assume risk-free rate of 2%
    sharpe_ratio = (avg_annual_return - 0.02) / std_annual_return

    return sharpe_ratio



def make_confusion_matrices(cfs: list,
                            group_names=None,
                            categories: list = None,
                            count=True,
                            percent=True,
                            cbar=True,
                            xyticks=True,
                            xyplotlabels=True,
                            sum_stats=True,
                            figsize=None,
                            cmap='Blues',
                            title=None,
                            labels= None,
                            preds: list = None):

    blanks = ['' for i in range(cfs[0].size)]

    if group_names and len(group_names) == cfs[0].size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    fig, axs = plt.subplots(1, len(cfs), figsize=figsize)

    for i, cf in enumerate(cfs):
        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:

            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            specificity = cf[1, 1] / sum(cf[1, :])
            sensitivity = cf[0, 0] / sum(cf[0, :])

            f1_score_ = f1_score(y_true=labels, y_pred=preds[i])
            mse = mean_squared_error(labels, preds[i])
            roc = roc_auc_score(labels, preds[i])
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nspecificity={:0.3f}\nsensitivity={:0.3f}\nF1 Score={:0.3f}\nMse={:0.3f}" \
                         "\n ROC_AUC Score={:0.3f}".format(

                accuracy, precision, specificity, sensitivity, f1_score_, mse, roc)

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories,
                    ax=axs[i])

        if xyplotlabels:
            axs[i].set_ylabel('True label')
            axs[i].set_xlabel('Predicted label' + stats_text)
        else:
            axs[i].set_xlabel(stats_text)

        if title:
            axs[i].set_title(title[i])

    plt.show()








def show_confusion_matrix(labels ,  preds:list ,names: list):
    """

    Renvoie la matrice de confusion du modéle

    """
    label=["True Neg","False Pos","False Neg","True Pos"]
    categories = ["Slowdown", "Acceleration"]

    confusion_dfs = []
    for i, pred in enumerate(preds):
        confusion_dfs.append(confusion_matrix(labels,pred))

    make_confusion_matrices(cfs = confusion_dfs,categories=categories,group_names=label,labels=labels
                               ,preds=preds, title=names)



def store_predictions(y_hat, name):
    """
    Stores the predicted labels and probabilities for a given method in a .txt file.

    Parameters:
    -----------
    y_hat_label : pandas DataFrame
        The predicted labels.
    y_hat_probs : pandas DataFrame
        The predicted probabilities.
    name : str
        The name of the method used to make the predictions.

    Returns:
    --------
    None
    """
    # create a new pandas dataframe to hold the predictions and probabilities

    # write the data to a text file
    filename = f"{name}.txt"
    y_hat.to_csv(filename, sep='\t', index=True)



def read_predictions(filename):
    """
    Reads predicted labels and probabilities from a .txt file.

    Parameters:
    -----------
    filename : str
        The name of the .txt file containing the predictions.

    Returns:
    --------
    y_hat_label : pandas DataFrame
        The predicted labels.
    y_hat_probs : pandas DataFrame
        The predicted probabilities.
    name : str
        The name of the method used to make the predictions.
    """
    # read the data from the file into a pandas dataframe
    y_hat = pd.read_csv(filename, sep='\t')
    y_hat["dates"] = pd.to_datetime(y_hat["dates"])
    y_hat.set_index("dates", drop=True, inplace=True)
    y_hat = y_hat.astype("float")
    # separate the labels and probabilities into separate dataframes

    # extract the method name from the filename
    #name = filename.split('_')[0]

    return y_hat


def compare_models(y_true, y_model1, y_model2,alpha=.05):
    """
    Compare the performance of two models using McNemar's test.

    Args:
        y_true (numpy.ndarray): True binary labels for each observation in the test data.
        y_model1 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 1.
        y_model2 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 2.

    Returns:
        tuple: A tuple containing the test result (reject or fail to reject the null hypothesis),
               the p-value of the test, and the test statistic.
    """

    H0 = "There is no significant difference in the performance of the two models"
    H1 = "One of the models performs significantly better than the other"

    # Calculate the counts of true positives, false positives, false negatives, and true negatives for each model.
    tb = mcnemar_table(y_target=y_true,
                       y_model1=y_model1,
                       y_model2=y_model2)

    # Calculate the McNemar's test statistic and p-value.
    chi2, p_value = mcnemar(ary=tb, corrected=True)


    # Determine whether to reject or fail to reject the null hypothesis based on the p-value.
    if p_value < 0.05:
        test_result = "Reject the null hypothesis"
    else:
        test_result = "Fail to reject the null hypothesis"

    # Determine the best performing model in terms of the decision variable.
    model1_wins = 0
    model2_wins = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_model1[i] == 1 and y_model2[i] == 0:
            model1_wins += 1
        elif y_true[i] == 1 and y_model1[i] == 0 and y_model2[i] == 1:
            model2_wins += 1
        elif y_true[i] == 0 and y_model1[i] == 1 and y_model2[i] == 0:
            model2_wins += 1
        elif y_true[i] == 0 and y_model1[i] == 0 and y_model2[i] == 1:
            model1_wins += 1

    if model1_wins > model2_wins:
        best_model = "Model 1"
    elif model2_wins > model1_wins:
        best_model = "Model 2"
    else:
        best_model = "Both models perform equally well"
    results = f"Results of McNemar's test:\nNull Hypothesis (H0): {H0}\n" \
              f"Alternative Hypothesis (H1): {H1}\n\nMcNemar Table:\n{tb}\n" \
              f"\nMcNemar's test statistic: {chi2:.3f}\np-value: {p_value:.3f}\nSignificance level: {alpha:.3f}\n\n{test_result} " \
              f"\n\nBest performing model in terms of the decision variable: {best_model}"

    print(results)


    return test_result, p_value, chi2


def get_optimal_thresholds(models, X, Y):
    """
    This function takes in a dictionary of models, a dataset of features, and a dataset of true labels.
    It returns a dictionary of optimal thresholds for each model in the given dictionary.

    Args:
    - models: a dictionary of trained models
    - X: a dataset of features associated with the dataset of true labels
    - Y: a dataset of  true labels

    Returns:
    - opt_thresholds: a dictionary of optimal thresholds for each model in the given dictionary
    """
    opt_thresholds = {}
    for name, model in models.items():
        # Get ROC and Precision-Recall curves for the current model
        fpr, tpr, thresholds = roc_curve(Y, model.predict_proba(X)[:, 1])
        precision, recall, thresholds = precision_recall_curve(Y, model.predict_proba(X)[:, 1])

        # Calculate Metrics for the current model
        gmeans = np.sqrt(tpr * (1 - fpr))
        J = tpr - fpr
        fscore = (2 * precision * recall) / (precision + recall)

        # Evaluate thresholds for the current model
        scores = [log_loss(Y, (model.predict_proba(X)[:, 1] >= t).astype('int')) for t in
                  thresholds]
        ix_1 = np.argmin(scores)
        ix_2 = np.argmin(np.abs(J))
        ix_3 = np.argmax(fscore)
        ix_4 = np.argmax(gmeans)
        opt_threshold = {"log_score": thresholds[ix_1], "g_means": thresholds[ix_4], "J": thresholds[ix_2],
                         "f1_score": thresholds[ix_3]}

        opt_thresholds[name] = opt_threshold

    return opt_thresholds


def plot_before_after_transformation(data, transformed_data):

    x = np.arange(len(data))
    col = data.columns

    # Get the mean and standard deviation of the data before and after transformation
    stats_before, stats_after = data.describe(), transformed_data.describe()
    stats_before, stats_after = stats_before.loc[["mean", "std"]].T, stats_after.loc[["mean", "std"]].T

    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    fig_, axs_ = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Plot the data before transformation on the first subplot
    axs[0].plot(x, data, label=col)
    axs[0].set_title('Before Transformation')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Data Values')

    # Plot the data after transformation on the second subplot
    axs[1].plot(x, transformed_data, label=col)
    axs[1].set_title('After Transformation')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Transformed Data Values')



    # Plot the mean and standard deviation of the data before transformation on the third subplot
    axs_[0].bar(x=stats_before.index, height=stats_before["mean"], yerr=stats_before["std"], width=0.3, align='center')
    axs_[0].set_title('Before Transformation')
    axs_[0].set_ylabel('Meand and Std')


    # Plot the mean and standard deviation of the data after transformation on the fourth subplot
    axs_[1].bar(x=stats_after.index, height=stats_after["mean"], yerr=stats_after["std"], width=0.3, align='center')
    axs_[1].set_title('After Transformation')
    axs_[1].set_ylabel('Mean and Std')




    # Add legend and grid to all subplots
    for ax in axs.flat:
        ax.legend(loc='best')
        ax.grid(True)

    for ax in axs_.flat:
        ax.legend(loc='best')
        ax.grid(True)

    plt.show()



def plot_autocorrelation(df):
    """
    Plots the autocorrelation of a given time series.

    Args:
    df (pd.DataFrame): a pandas DataFrame containing the time series data.

    Returns:
    None: The function displays the plot.
    """

    # Create a figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot the autocorrelation function with lags up to 60
    plot_acf(df, ax=ax, lags=60)

    # Set the x and y labels and the title for the plot
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')




def plot_feature_importance(model, X):
    # Calculate SHAP values
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    # Calculate feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    feature_names = X.columns.tolist()

    # Sort feature importance in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot feature importance with SHAP values

    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importance with SHAP Values')
    plt.show()

def plot_shap(models, X):
    explainer_rf = shap.TreeExplainer(models["rf_"])
    explainer_gb = shap.TreeExplainer(models["gb_"])
    explainer_logit = shap.explainers.Linear(models["logit_"],X)

    shap_values_rf = explainer_rf.shap_values(X)
    shap_values_gb = explainer_gb.shap_values(X)
    shap_values_logit = explainer_logit.shap_values(X)


    shap.summary_plot(shap_values_rf, features=X, feature_names=X.columns)
    shap.summary_plot(shap_values_gb, features=X, feature_names=X.columns)
    shap.summary_plot(shap_values_logit, features=X, feature_names=X.columns)
















class Models:
    """
    A class for initializing prediction models and their parameters.
    """

    def __init__(self, Y: pd.DataFrame, X: pd.DataFrame,
                 date_split: int, step_ahead: int, tuning: dict, method:dict) -> None:
        """
        Initialize the "models" object and the variables necessary for prediction.

        :param Y: The variable to be predicted
        :param X: The covariate matrix
        :param date_split: The date from which to predict Y
        :param step_ahead: The prediction step
        :param tuning: dictionary (boolean) of model tuning operations
        """

        self.date_split = date_split
        self.step_ahead = step_ahead
        self.tuning = tuning
        self.method = method
        self.X = X
        self.Y = pd.DataFrame(np.nan, index=Y.index, columns=["true_label"])
        self.Y["true_label"] = Y

        self.Y_hat = pd.DataFrame(np.nan, index=self.Y.index, columns=["true_label"])
        self.Y_hat["true_label"] = self.Y["true_label"]
        # Add new models here
        for model in ["logit_", "rf_", "gb_"]:
            self.Y_hat[model + 'label'] = self.Y.iloc[:date_split]
            self.Y_hat[model + 'probs'] = None
            self.Y_hat[model + "opt_threshold_log_score"] = None
            self.Y_hat[model + "opt_threshold_g_means"] = None
            self.Y_hat[model + "opt_threshold_J"] = None
            self.Y_hat[model + "opt_threshold_f1_score"] = None
            self.Y_hat[model + "opt_label_log_score"] = None
            self.Y_hat[model + "opt_label_g_means"] = None
            self.Y_hat[model + "opt_label_J"] = None
            self.Y_hat[model + "opt_label_f1_score"] = None

        # Stores most important variables

        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)

        # Stores accuracy of imputation

        self.acc_f_time = pd.DataFrame(np.nan,index=Y.index, columns = ["logit_acc","rf_acc","gb_acc"])

    def models(self):

        # Booleans to choose which method of imputation we are using
        method_1, method_2 = self.method["method_1"], self.method["method_2"]

        # Booleans to choose methods to enhance/tune the model
        normalize, resample, threshold_tuning, pca, params_tuning =self.tuning["normalize"],self.tuning["resample"],\
                                                                   self.tuning["threshold_tuning"], self.tuning["pca"],\
                                                                   self.tuning["params_tuning"]




        print(str(self.step_ahead) + " step-ahead training and predicting...")

        range_data_split = range(self.date_split, len(self.X))
        most_imp = []

        for id_split in range_data_split:

            # This method consists of predicting missing Y's that we've
            # supposed unknown for 12 months before our current prediction

            if method_1:

                print("predicting  Y_%i | X[0:%i,:] with method 1, date:%s" % (id_split, id_split - self.step_ahead, self.Y.index[id_split].strftime('%Y-%m-%d')))
                # Every 12 months, we erase potential errors
                # we could have done due to imputation

                Y_train_tilde_logit = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]
                Y_train_tilde_rf = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]
                Y_train_tilde_gb = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]

                # Loop on missed values of Y

                for i in range(0, self.step_ahead + 1):

                    if i != self.step_ahead:

                        print("\t predicting missing value Y_%i, date :%s" % (id_split - self.step_ahead + i, self.Y.index[id_split - self.step_ahead + i].strftime('%Y-%m-%d')))

                    else:

                        print("\t predicting Y_%i with imputation, date:%s" % (id_split - self.step_ahead + i, self.Y.index[id_split - self.step_ahead + i].strftime('%Y-%m-%d')))


                    X_train_US = self.X.iloc[0:id_split - self.step_ahead + i, :]
                    X_test_US = self.X.iloc[id_split - self.step_ahead + i:, :]


                    # If True normalizes data

                    if normalize:

                        # Here we can also use minmax_norm()

                        X_train_US, X_test_US = standardization_norm(X_train_US, X_test_US)[0].copy(), standardization_norm(X_train_US, X_test_US)[1].copy()

                        # Drop nan observations generated by normalization

                        if X_train_US.isna().any().any():

                            aux_logit = pd.concat([X_train_US, Y_train_tilde_logit], axis=1).dropna().copy()
                            Y_train_tilde_logit = aux_logit.iloc[:, -1].copy()
                            X_train_US = aux_logit.iloc[:, :-1].copy()

                            aux_rf = pd.concat([X_train_US, Y_train_tilde_rf], axis=1).dropna().copy()
                            Y_train_tilde_rf = aux_rf.iloc[:, -1].copy()
                            X_train_US = aux_rf.iloc[:, :-1].copy()

                            aux_gb = pd.concat([X_train_US, Y_train_tilde_gb], axis=1).dropna().copy()
                            Y_train_tilde_gb = aux_gb.iloc[:, -1].copy()
                            X_train_US = aux_gb.iloc[:, :-1].copy()







                    # If True resamples train data
                    # Check function implementation for more details

                    if resample:

                        # Auxiliary variables to keep the format of the Y_train_tile_model df
                        # Otherwise we get errors if resample is True

                        aux_logit = Y_train_tilde_logit.copy()
                        aux_rf = Y_train_tilde_rf.copy()
                        aux_gb = Y_train_tilde_gb.copy()

                        resampled_logit = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_logit], axis=1))
                        resampled_rf = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_rf], axis=1))
                        resampled_gb = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_gb], axis=1))

                        Y_train_tilde_logit = resampled_logit.iloc[:, -1]
                        Y_train_tilde_rf = resampled_rf.iloc[:, -1]
                        Y_train_tilde_gb = resampled_gb.iloc[:, -1]

                        X_train_US = resampled_logit.iloc[:, :-1]


                    # If True performs a PCA on train set

                    if pca:
                        most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                        most_imp.append(most_important_features)
                        X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(
                            most_important_features)

                    # If True selects the best hyperparams of the model
                    if params_tuning and id_split == len(self.X) - 1:
                        params_grid_logit = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                        params_grid_gb = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


                        params_grid_rf = {'max_depth': [None] + list(range(4, 8, 2)),
                                       'min_samples_split': range(2, 8, 2),
                                       'n_estimators': [100, 200, 1000]}
                        grid_search_cv_logit = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42),
                                                      params_grid_logit,
                                                      scoring="neg_log_loss", cv=5)
                        grid_search_cv_gb = GridSearchCV(XGBClassifier(random_state=42),
                                                      params_grid_gb,
                                                      scoring="neg_log_loss", cv=5)
                        grid_search_cv_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                                                      params_grid_rf,
                                                      scoring="neg_log_loss", cv=5)
                        model_logit = grid_search_cv_logit.fit(X_train_US, Y_train_tilde_logit)
                        model_gb = grid_search_cv_gb.fit(X_train_US, Y_train_tilde_rf)
                        model_rf = grid_search_cv_rf.fit(X_train_US, Y_train_tilde_gb)


                    # Create model for each iteration to predict the next unknown Y

                    else:

                        model_logit = LogisticRegression(max_iter=5000, random_state=42, C=10).fit(X_train_US,
                                                                                             Y_train_tilde_logit)
                        model_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US,
                                                                                         Y_train_tilde_rf)
                        model_gb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                                      random_state=42).fit(X_train_US,Y_train_tilde_gb)





                    models = {"logit_": model_logit, "rf_": model_rf, "gb_": model_gb}

                    if threshold_tuning:

                        warnings.warn("Threshold tuning not implemented yet for meth_1")


                        print("\t \t threshold tuning for the %i th observation" % id_split)

                        #opt_threshold = get_optimal_thresholds(models, X_train_US, Y_train_tilde_logit)

                    # Get back the nonresampled Y_train_tilde to modify it

                    if resample:

                        Y_train_tilde_logit = aux_logit.copy()
                        Y_train_tilde_rf = aux_rf.copy()
                        Y_train_tilde_gb = aux_gb.copy()


                    # Store predictions of missing Y's

                    Y_train_tilde_logit.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_logit.predict(X_test_US)[
                    0].copy()
                    Y_train_tilde_rf.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_rf.predict(X_test_US)[
                    0].copy()
                    Y_train_tilde_gb.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_gb.predict(X_test_US)[
                    0].copy()

                # Variable to compute the accuracy of predicting the missing Y's

                true_y = self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead: id_split], "true_label"]


                # At the end of loop, one can check that id_split == id_split - self.step_ahead + i

                for (model_name, model) in zip(models, models.values()):

                    self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead + i], model_name + "probs"] = \
                        '{:.2f}'.format(model.predict_proba(X_test_US)[0][1].copy())
                    self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead + i], model_name + "label"] = \
                    model.predict(X_test_US)[
                        0].copy()

                    # Compute accuracy of previous imputed Ys

                    imputed_y = self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead: id_split], model_name + "label"]
                    acc = accuracy_score(true_y, imputed_y)
                    self.acc_f_time.loc[id_split, model_name+ "acc"] = acc

                    if threshold_tuning:
                        for model in ["logit_", "rf_", "gb_"]:
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_log_score"] = \
                            opt_threshold[model]["log_score"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_g_means"] = \
                            opt_threshold[model]["g_means"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_J"] = opt_threshold[model][
                                "J"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_f1_score"] = \
                            opt_threshold[model]["f1_score"]

                print(self.Y_hat.filter(["true_label", "logit_label"]).iloc[198:id_split + 10])


            else:

                print("predicting  Y_%i | X from 0 to %i, date:%s" % (id_split, id_split - self.step_ahead-1, self.Y.index[id_split].strftime('%Y-%m-%d')))
                Y_train_US = self.Y.loc[self.Y.index[0:id_split - self.step_ahead],"true_label"]
                X_test_US = self.X.iloc[id_split:, :]

                if method_2:
                    print('\t with method 2 \n')
                    for i in range(0,self.step_ahead):
                        Y_train_US[self.Y.index[id_split - self.step_ahead + i + 1]] = Y_train_US[id_split - self.step_ahead-1]
                    X_train_US = self.X.iloc[0:id_split, :]
                else:
                    X_train_US = self.X.iloc[0:id_split - self.step_ahead, :]

                print(Y_train_US)
                if resample:

                    resampled = resample_dataframe(pd.concat([X_train_US, Y_train_US], axis=1))
                    Y_train_US = resampled.iloc[:, -1]
                    X_train_US = resampled.iloc[:, :-1]
                print(Y_train_US)
                # If True normalizes data

                if normalize:

                    # Here we can also use standardization_norm()

                    X_train_US, X_test_US = standardization_norm(X_train_US, X_test_US)

                    # Drop nan observations generated by normalization

                    if X_train_US.isna().any().any():
                        aux = pd.concat([X_train_US, Y_train_US], axis=1).dropna()
                        Y_train_US = aux.iloc[:, -1]
                        X_train_US = aux.iloc[:, :-1]

                if params_tuning and id_split == len(self.X) - 1:
                    params_grid_logit = {'penalty': ['l1', 'l2'],
                                         'C': [0.01,0.1, 1.0, 10.0,100],
                                         'fit_intercept': [True, False],
                                         'solver': ['liblinear', 'saga']}

                    params_grid_gb = {'learning_rate': [0.05, 0.1, 0.2],
                                              'n_estimators': [50, 100, 200],
                                              'max_depth': [3, 4, 5],
                                              # 'min_samples_split': [2, 4, 6],
                                              # 'subsample': [0.6, 0.8, 1.0],
                                              'max_features': ['sqrt', 'log2', None]}

                    params_grid_rf = {'n_estimators': [50, 100, 200],
                                      # 'max_depth': [None, 10, 20, 30],
                                      # 'min_samples_split': [2, 5, 10],
                                      # 'min_samples_leaf': [1, 2, 4],
                                      'max_features': ['sqrt', 'log2', None]}
                    print('\t Params tuning for Logit \n')
                    grid_search_cv_logit = GridSearchCV(LogisticRegression(max_iter=5000,random_state=42),
                                                        params_grid_logit,
                                                        scoring="f1", cv=5)
                    print('\t Params tuning for GB \n')

                    grid_search_cv_gb = GridSearchCV(XGBClassifier(random_state=42),
                                                     params_grid_gb,
                                                     scoring="neg_log_loss", cv=5)

                    print('\t Params tuning for RF \n')

                    grid_search_cv_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                                                     params_grid_rf,
                                                     scoring="neg_log_loss", cv=5)

                    model_logit = grid_search_cv_logit.fit(X_train_US, Y_train_US)
                    print("\t\t",model_logit.best_params_,"\n")

                    model_gb = grid_search_cv_gb.fit(X_train_US, Y_train_US)
                    print("\t\t",model_gb.best_params_,"\n")

                    model_rf = grid_search_cv_rf.fit(X_train_US, Y_train_US)
                    print("\t\t",model_rf.best_params_,"\n")


                    models = {"logit": model_logit, "rf": model_rf, "gb": model_gb}


                else:

                    print('\t Logit \n')
                    model_logit = LogisticRegression(max_iter=5000, random_state=42, C=1).fit(X_train_US, Y_train_US)

                    print('\t RF \n')
                    model_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US, Y_train_US)

                    print('\t GB \n')
                    model_gb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                             random_state=42).fit(X_train_US, Y_train_US)


                    models = {"logit_": model_logit, "rf_": model_rf, "gb_": model_gb}

                if threshold_tuning:

                    print("\t \t threshold tuning for the %i th observation" % id_split)

                    opt_threshold = get_optimal_thresholds(models, X_train_US,Y_train_US)

                models = {"logit_":model_logit,"rf_":model_rf,"gb_":model_gb}

                for (model_name,model) in zip(models,models.values()):
                    p = '{:.2f}'.format(model.predict_proba(X_test_US)[0][1])
                    self.Y_hat.loc[self.Y.index[id_split],model_name + "probs"] = p
                    self.Y_hat.loc[self.Y.index[id_split],model_name + "label"] = model.predict(X_test_US)[0]

                    if threshold_tuning:
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_log_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["log_score"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_log_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["log_score"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_g_means'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["g_means"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_J'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["J"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_f1_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["f1_score"])




                if threshold_tuning:
                    for model in ["logit_", "rf_", "gb_"]:
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_log_score"]= '{:.2f}'.format(opt_threshold[model][
                            "log_score"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_g_means"] = '{:.2f}'.format(opt_threshold[model][
                            "g_means"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_J"]= '{:.2f}'.format(opt_threshold[model]["J"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_f1_score"] = '{:.2f}'.format(opt_threshold[model][
                            "f1_score"])

        importance_logit = model_logit.coef_[0]
        importance_rf = model_rf.feature_importances_
        importance_gb = model_gb.feature_importances_

        feature_importance = {"logit_": importance_logit,"rf_": importance_rf,"gb_": importance_gb}

        if pca:
            return models, self.Y_hat, most_imp[-1]
        else:
            return models, self.Y_hat, self.acc_f_time, feature_importance
        

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import yahoofinancials as yf
import statsmodels.api as sm


def risk_free_index_processing():
    history_13w_ustb = yf.YahooFinancials('^IRX').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')
    history_10y_ustb = yf.YahooFinancials('^TNX').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')

    df1 = pd.DataFrame(history_13w_ustb['^IRX']['prices'])
    df2 = pd.DataFrame(history_10y_ustb['^TNX']['prices'])
    df1.drop('date', axis=1, inplace=True)
    df2.drop('date', axis=1, inplace=True)

    df1.index = pd.to_datetime(df1['formatted_date'])
    df2.index = pd.to_datetime(df2['formatted_date'])

    df1["price"] = df1["adjclose"]
    df2["price"] = df2["adjclose"]

    df1 = df1.filter(["price"])
    df2 = df2.filter(["price"])

    # df = df.resample('D').ffill()
    df1 = df1.resample('D').mean()  # Resample to daily frequency and aggregate using mean
    df2 = df2.resample('D').mean()
    # df = df.resample('D').ffill()
    df1 = df1.interpolate()
    df2 = df2.interpolate()# Interpolate missing values using linear interpolation
    df1 = df1[df1.index.day == 15]
    df2 = df2[df2.index.day == 15]

    return df1,df2

def risky_index_processing():
    history_sp = yf.YahooFinancials('^GSPC').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')
    df = pd.DataFrame(history_sp['^GSPC']['prices'])
    df.drop('date', axis=1, inplace=True)
    df.index = pd.to_datetime(df['formatted_date'])
    df["price"] = df["adjclose"]
    df = df.filter(["price"])
    df = df.resample('D').ffill()
    df = df[df.index.day == 15]
    filename = "sp500_historical_data.txt"
    df.to_csv(filename, sep='\t', index=True)
    return df


def resample_dataframe(df, over=True, under=True):

    # Count the number of labels in the dataframe
    label_counts = df["true_label"].value_counts()

    # Determine the minority and majority classes
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()


    if over:
        # Determine the number of samples to keep from the minority class
        majority_count = label_counts[majority_label]
        # Sample the minority class
        aux = (df[df["true_label"] == minority_label].sample(n=majority_count, replace=True, random_state=42)).copy()
        minority_df = pd.DataFrame(aux, index=aux.index)

    else:
        aux = df[df["true_label"] == minority_label].copy()
        minority_df = pd.DataFrame(aux, index=aux.index)

    if under:
        minority_count = label_counts[minority_label]
        # Sample the majority class

        aux = (df[df["true_label"] == majority_label].sample(n=minority_count, replace=True, random_state=42)).copy()
        majority_df = pd.DataFrame(aux, index=aux.index)
    else:
        aux = df[df["true_label"] == majority_label].copy()
        majority_df = pd.DataFrame(aux, index=aux.index)

    # Concatenate the minority and majority samples


    balanced_df = pd.concat([minority_df, majority_df], axis=0, ignore_index=True)



    return balanced_df


def minmax_norm(X_train, X_test):
    """
       Normalize the data in X_train and X_test using a MinMaxScaler object.

       Parameters:
       X_train (pandas DataFrame): Training data set to normalize.
       X_test (pandas DataFrame): Testing data set to normalize.

       Returns:
       tuple: A tuple containing two pandas DataFrames with the normalized data
              from X_train and X_test respectively.
   """
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Normalize the data in X_train and X_test using the trained scaler
    X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_normalized,X_test_normalized


def standardization_norm(X_train, X_test):
    """
       Normalize the data in X_train and X_test using a StandardScaler object.

       Parameters:
       X_train (pandas DataFrame): Training data set to normalize.
       X_test (pandas DataFrame): Testing data set to normalize.

       Returns:
       tuple: A tuple containing two pandas DataFrames with the normalized data
              from X_train and X_test respectively.
   """

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Normalize the data in X_train and X_test using the trained scaler
    X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_normalized,X_test_normalized

def PCA_(data, important_features=False, n_comp=.99):
    """
    Effectue une PCA sur la matrice X

    :param data:
    :param important_features:
    :param n_comp:
    :return:
    """
    pca = PCA(n_components=n_comp)
    pca.fit_transform(data)
    n_pcs = pca.n_components_
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = data.columns
    most_important_features = [*set([initial_feature_names[most_important[i]] for i in range(n_pcs)])]
    if important_features:
        print(most_important_features)
    return most_important_features

class Data:
    def __init__(self,BDD_path, BDD_sheet):
        self.raw_df = pd.read_excel(BDD_path, sheet_name=BDD_sheet)
        self.df = None

    def data_processing(self,resample=False):
        """

        :param resample: Si vrai utilise une méthode de réchantillonage des données
        :return:
        """
        self.df = self.raw_df.iloc[1:,:]
        self.df.columns = self.raw_df.iloc[0,:]
        self.df.set_index("dates", drop=True, inplace=True)
        self.df = self.df.astype("float")
        if resample:
            self.df = self.df.resample('D',axis=0).interpolate('linear')

        print("Data processed succesfully")

    def target(self,resample=False):
        if resample:
            return self.df.iloc[:,-1].astype('int')
        return self.df.iloc[:,-1]

    def covariates(self,returns=False, log=False, ts_analysis=False, wavelet=False, diff=False):
        if returns:
            aux = self.df.iloc[:, :-1].loc[:, (self.df.iloc[:, :-1] > 0).all()]

            df_returns = aux.pct_change()

            # Rename the columns of the returns DataFrame to match the original columns
            if log :
                aux = self.df.iloc[:,:-1].loc[:, (self.df.iloc[:,:-1] > 0).all()]

                df_returns = np.log(aux.iloc[:, 1:]) - np.log(aux.iloc[:, 1:].shift(1))

                df_returns.columns = [column + 'log_change' for column in df_returns.columns]
            else:
                df_returns.columns = [column + '_change' for column in df_returns.columns]

            new_dataset = pd.concat([self.df.iloc[:, :-1], df_returns], axis=1)

            # Replace the first row (became NaN due to .pct()) by interpolation
            new_dataset.iloc[0] = new_dataset.iloc[1]


            return new_dataset

        elif ts_analysis:
            cov = self.df.iloc[:, :-1].copy()
            cov[cov.columns + '_rolling_avg'] = cov[cov.columns].rolling(window=3).mean()
            cov.fillna(method="bfill",inplace=True)

            # Add a new column with the seasonal decomposition of the original column
            for col in cov.columns:
                decomposition = sm.tsa.seasonal_decompose(cov[col], model='additive', period=3)
                cov[col+ '_trend'] = decomposition.trend
                cov[col+ '_seasonal'] = decomposition.seasonal
                cov[col + '_residual'] = decomposition.resid
                cov.fillna(method="bfill",inplace=True)
                cov.fillna(method="ffill", inplace=True)

            return cov

        elif diff:

            cov = self.df.iloc[:, :-1].copy()

            for col in cov.columns:
                for i in range(50):
                    cov[f"{col}_diff_{i}"] = cov[col].diff(periods=i)

            cov.fillna(method="bfill", inplace=True)

            return cov



        else:

            return self.df.iloc[:,:-1]


    def data_summary(self):

        print(self.df.head())
        print(self.df.describe())
        print(self.target().value_counts())

    def stationarity(self):
        """
        Check si la série temporelle Y est stationnaire ou non.
        :return: le résultat du test
        """

        print('Dickey-Fuller Test: H0 = non stationnaire vs H1 = stationnaire')
        dftest = adfuller(self.target(),autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=["Statistique de test","p-value","lags","nobs"])
        for key,value in dftest[4].items():
            dfoutput['valeur critique(%s)'%key]=value
        print(dfoutput)


        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tabulate
from tabulate import tabulate
import PIL
from PIL import Image, ImageDraw, ImageFont


class Portfolio:

    def __init__(self, initial_capital:float,risk_free_index=None,risky_index=None,strategy=None, y_pred=None):

        self.capital = initial_capital
        self.y_pred = y_pred
        self.portfolio_history = pd.concat([pd.DataFrame(np.nan,index=y_pred.index, columns=["portfolio_value"]),
                                            pd.DataFrame(np.nan,index=y_pred.index, columns=["portfolio_returns"]),
                                            pd.DataFrame(np.nan,index=y_pred.index, columns=["B&H_strategy"])], axis=1)
        self.rf_rate = risk_free_index[0]
        self.bond_rate = risk_free_index[1]
        self.risky_assets = risky_index
        self.strategy = strategy

    def simulation(self):
        n_risky_assets_held = 0
        current_cash = self.capital

        # Portfolio value starts with current cash
        self.portfolio_history["portfolio_value"].iloc[0] = current_cash


        if self.strategy == "dynamic":
            for date in range(1,len(self.y_pred)-1):
                print(date)
                print(current_cash)
                print(n_risky_assets_held)

                # Bond return in %
                bond_return = (1+self.bond_rate['price'].iloc[date])**(1/12)-1

                # When we have cash placed in riskfree asset
                if current_cash > 0 and date > 1:
                    # Add monthly gains from riskfree cash allocation
                    current_cash += current_cash * bond_return/100

                    # Model predicts acceleration
                    if self.y_pred.iloc[date] == 1:

                        if date > 0:
                            # Sell last portfolio

                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                        # Buy new portfolio

                        # Allocate 80% of cash in risky asset

                        n_risky_assets_held = (current_cash * .8)/self.risky_assets['price'].iloc[date]

                        # Allocate 20% of cash in risky asset

                        current_cash = .2 * current_cash

                    else:

                        # Sell last portfolio
                        if date>0:
                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                        # Allocate 60% of cash in risky asset
                        n_risky_assets_held = (current_cash * .4) / self.risky_assets['price'].iloc[date]

                        # Allocate 40% of cash in risky asset
                        current_cash = .6 * current_cash


                # Evaluate and store portfolio value each month



                self.portfolio_history["portfolio_value"].iloc[date] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].iloc[date] \
                                                                       + current_cash

            # Compute portfolio returns and returns in %
            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change()

        elif self.strategy == "120/80_equity":
            # Number of months we were in acceleration phase
            # To calculate cost of leverage
            n_months = 0

            # To track leverage cost
            borrowed_cash = 0


            for date in range(1,len(self.y_pred)-1):

                borrowing_cost = ((1+self.rf_rate['price'].iloc[date])**(1/12)-1 )/ 100

                # Model predicts acceleration
                if self.y_pred.iloc[date] == 1:

                    n_months += 1

                    # Sell last portfolio
                    if date>0:
                        current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                    # Buy new portfolio

                    # Allocate 120% of cash in risky asset
                    n_risky_assets_held = (current_cash * 1.2)/self.risky_assets['price'].iloc[date]

                    #Borrow 20%
                    borrowed_cash = 0.2 * current_cash
                    current_cash = - borrowed_cash


                else:
                    n_months=0

                    if date>0:
                        if current_cash!=0 and borrowed_cash>0:
                            current_cash -= borrowed_cash * n_months * borrowing_cost

                    # Sell last portfolio

                    current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                    # Allocate 80% of cash in risky asset

                    n_risky_assets_held = (current_cash * .6) / self.risky_assets['price'].iloc[date]

                    # Allocate 20% of cash in riskfree asset

                    current_cash = .4 * current_cash




                # Evaluate and store portfolio value in each iteration


                self.portfolio_history["portfolio_value"].iloc[date] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].iloc[date] \
                                                                       + current_cash

            # Compute portfolio returns in %

            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change()

        return self.portfolio_history["portfolio_value"], self.portfolio_history["return_pct"]

    def backtest_report_(self,portfolios_history: list = None, names: list = None):
        if self.strategy == "dynamic":
            print("************* Descriptive Statistics for the Dynamic Strategy *************")
        else:
            print("************* Descriptive Statistics for the 120/80  Strategy *************")
        print("Period", len(self.y_pred), "days")
        print("Max Monthly Drawdown", 100 * round(self.portfolio_history["return_pct"].min(), 2), "%")
        print("Max Monthly Drawdown BH", 100 * round(self.risky_assets['price'].pct_change().min(), 2), "%")

        print("Highest Monthly Return ", 100 * round(self.portfolio_history["return_pct"].max(), 2), "%")
        print("Average  Returns ", 100 * self.portfolio_history["return_pct"].mean(), "%")
        print("Average  Returns of the becnhmark ", 100 * self.risky_assets['price'].pct_change().mean(), "%")

        print("Volatility", 100 * round(self.portfolio_history["return_pct"].std(), 2), "%")
        print("Total Potential Return ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"]> 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Total Potential Loss ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"] < 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Net Return ", 100 * self.portfolio_history["return_pct"].sum().round(2), "%")
        print("Sharpe ratio", compute_sharpe_ratio(self.portfolio_history["return_pct"]))
        print("**************************************************")

    import pandas as pd
    from tabulate import tabulate

    import tabulate

    import pandas as pd
    from tabulate import tabulate

    import pandas as pd
    from tabulate import tabulate

    import pandas as pd
    from tabulate import tabulate

    def backtest_report(self, portfolios_history: list = None, names: list = None):
        """
        Generate a report for each portfolio in the input list.

        Args:
        portfolios_history (list): a list of dictionaries containing the history of each portfolio
        names (list): a list of names corresponding to each portfolio

        Returns:
        None
        """

        reports = []
        for portfolio_history, name in zip(portfolios_history, names):
            report = {}
            report['Period'] = len(portfolio_history)
            report['Max Monthly Drawdown in %'] = 100 * round(portfolio_history["return_pct"].min(), 2)
            report['Highest Monthly Return in %'] = 100 * round(portfolio_history["return_pct"].max(), 2)
            report['Average Returns in %'] = 100 * portfolio_history["return_pct"].mean()
            report['Volatility'] = 100 * round(portfolio_history["return_pct"].std(), 2)
            report['Net Return in %'] = 100 * portfolio_history["return_pct"].sum().round(2)
            report['Sharpe ratio'] = compute_sharpe_ratio(portfolio_history["return_pct"])
            reports.append(report)


        df = pd.DataFrame(reports)
        df.set_index('Period', inplace=True)
        df = df.transpose()
        df.columns = names
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))



    def plots(self, portfolios_history: list = None, y_preds: list = None, names: list = None):
        # Plots

        f, axarr = plt.subplots(2, figsize=(12, 7))
        if self.strategy == "120/80_equity":
            f.suptitle('Portfolio Value and Return with the 120/80 Equity strategy', fontsize=20)
        else:
            f.suptitle('Portfolio Value and Return with the dynamic strategy', fontsize=20)

        model_names = ["rf_", "gb_","logit_"]
        for (portfolio_history, y_pred, name, model_name) in zip(portfolios_history, y_preds, names, model_names):
            df = pd.concat([portfolio_history['portfolio_value'], y_pred], axis=1)

            axarr[0].plot(portfolio_history["portfolio_value"], label=f"{name} Portfolio", linewidth=2.5)
            axarr[0].scatter(df[df[model_name + "label"] == 1].index,
                             df["portfolio_value"][df[model_name + "label"] == 1], color='green', marker='.', s=100,
                             label=f"{name} Predicted Acc.")
            axarr[0].scatter(df[df[model_name + "label"] == 0].index,
                             df["portfolio_value"][df[model_name + "label"] == 0], color='red', marker='.', s=100,
                             label=f"{name} Predicted Slo.")
            axarr[1].plot(100 * portfolio_history["return_pct"], label=f"{name} Portfolio returns")
            axarr[1].grid(True)

        axarr[0].plot(self.risky_assets['price'], color='black', label='B&H SP500')
        axarr[0].grid(True)

        axarr[0].legend(loc='best')
        axarr[1].legend(loc='best')

        plt.show()


        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------


import pandas as pd

#pd.set_option('display.max_columns',None)

# Change path here

BDD_path = "BDD_SummerSchool_BENOIT.xlsx"

# Change sheet name here

BDD_sheet = "raw_data"

# Process data
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X = data.covariates()
Y = data.target()

# Change parameters here

tuning = {"normalize": True, "resample": True, "threshold_tuning": False, "pca": False, "params_tuning": False}
imputation_method = {"method_1":True, "method_2":False}

# Creates model
model = Models(Y=data.target(), X=X, date_split=204, step_ahead=12, tuning=tuning, method=imputation_method)

model_ = model.models()
models , y_hat, importance, acc = model_[0], model_[1], model_[-1], model_[-2]

# Store predictions
store_predictions(y_hat,"normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200")



        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------


from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet ="raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X=data.covariates()
Y = data.target()

# If available reads stored predictions

f = open("normalisation_meth1_with_lag_c1_rf200_gb200.txt",'r')
g = open("normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200.txt",'r')
y_hat = read_predictions(f)
y_hat_ = read_predictions(g)

plot, compare= True, True

if plot:
    pred_probs = [y_hat["logit_probs"].iloc[204:430],y_hat["rf_probs"].iloc[204:430],y_hat["gb_probs"].iloc[204:430]]
    pred_labels = [y_hat["logit_label"].iloc[204:430],y_hat["rf_label"].iloc[204:430],y_hat["gb_label"].iloc[204:430]]
    opt_threshold = [y_hat["logit_opt_threshold_f1_score"].iloc[204:430],y_hat["rf_opt_threshold_f1_score"].iloc[204:430],y_hat["gb_opt_threshold_f1_score"].iloc[204:430]]

    pred_probs_ = [y_hat_["logit_probs"].iloc[204:430], y_hat_["rf_probs"].iloc[204:430], y_hat_["gb_probs"].iloc[204:430]]
    pred_labels_ = [y_hat_["logit_label"].iloc[204:430], y_hat_["rf_label"].iloc[204:430], y_hat_["gb_label"].iloc[204:430]]
    opt_threshold_ = [y_hat_["logit_opt_threshold_f1_score"].iloc[204:430],
                     y_hat_["rf_opt_threshold_f1_score"].iloc[204:430], y_hat_["gb_opt_threshold_f1_score"].iloc[204:430]]
    names = ["Logistic Regression","Random Forest Classifier","XGradientBoosting Classifier"]

    plot_predictions(y_label=data.target().iloc[204:],y_probs=pred_probs,names=names)
    show_confusion_matrix(labels=data.target().iloc[204:430],preds=pred_labels, names=names)

    plot_predictions(y_label=data.target().iloc[204:], y_probs=pred_probs_, names=names)
    show_confusion_matrix(labels=data.target().iloc[204:430], preds=pred_labels_, names=names)

#compare models
if compare:
    models = ["logit_","rf_","gb_"]
    for model_name in models:
        print(model_name)
        compare_models(y_true=data.target().iloc[204:430],y_model2=y_hat_[model_name + "label"].iloc[204:430],
                   y_model1=y_hat[model_name + "label"].iloc[204:430])





        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------


from pprint import pprint
import pandas as pd
from os.path import exists

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet ="raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X=data.covariates()
Y = data.target()

f = open("normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200.txt", "r")
g = open("normalisation_resample_over_under_with_lag_c1_rf200_gb200.txt","r")
h = open("normalisation_with_lag_c1_rf200_gb200.txt","r")
y_hat= read_predictions(f)
y_hat_ = read_predictions(g)
y_hat__ = read_predictions(h)


pred_labels = [y_hat["rf_label"].iloc[204:430], y_hat_["gb_label"].iloc[204:430], y_hat__["logit_label"].iloc[204:430]]
names = ["Random Forest Classifier", "XGradientBoosting Classifier","Logistic Regression"]

# Create portfolio
portfolio = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[0],strategy="dynamic")

portfolio_ = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[1],strategy="dynamic")

portfolio__ = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[2],strategy="dynamic")

# Simulation of the strategy and plots
portfolio.simulation()
portfolio_.simulation()
portfolio__.simulation()

portfolios_history = [portfolio.portfolio_history,portfolio_.portfolio_history,portfolio__.portfolio_history]

portfolio.plots(portfolios_history=portfolios_history,y_preds=pred_labels,names=names)

# Backtest report
portfolio.backtest_report(portfolios_history=portfolios_history,names=names)



        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------





