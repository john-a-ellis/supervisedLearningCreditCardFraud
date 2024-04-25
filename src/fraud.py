#importing Dependencies
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.metrics.pairwise import haversine_distances
from math import radians


## function to genearte Confusion Matrix and Classification Report
def gen_cm_cr(modelName, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, index=["Legitimate 0", "Fraudulent 1"], columns=["Predicted Legitimate 0", "Predicted Fraudulent 1"]
    )
    
    # Calculating the accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    bal_acc_score = balanced_accuracy_score(y_test, y_pred)
    print("Confusion Matrix: " + modelName)
    display(cm_df)
    print(f"Accuracy Score : {acc_score}")
    print(f"Balanced Accuracy Score: {bal_acc_score}")
    print("Classification Report")
    print(classification_report(y_test, y_pred))

## function to find the distance in kilometers between two geo-spatial co-ordinates.  Based on excample from Scikit Learn documentation
def get_distance(from_lat, from_lng, to_lat, to_lng):

    start =[]
    end =[]
    start.append(from_lat)
    start.append(to_lat)
    end.append(to_lat)
    end.append(to_lng)
    start_in_radians = [radians(_) for _ in start]
    end_in_radians = [radians(_) for _ in end]
    result = haversine_distances([start_in_radians, end_in_radians])
    result * 6371000/1000 
    return result

## function to assign the U.S. Bureau of Economic Analysis region to a state.
def get_region(state):
    
    states_to_bea_regions = {
        "AL": "Southeast",
        "AK": "Far West",
        "AZ": "Southwest",
        "AR": "Southeast",
        "CA": "Far West",
        "CO": "Rocky Mountain",
        "CT": "New England",
        "DE": "Mideast",
        "DC": "Mideast",
        "FL": "Southeast",
        "GA": "Southeast",
        "HI": "Far West",
        "ID": "Rocky Mountain",
        "IL": "Great Lakes",
        "IN": "Great Lakes",
        "IA": "Plains",
        "KS": "Plains",
        "KY": "Southeast",
        "LA": "Southeast",
        "ME": "New England",
        "MD": "Mideast",
        "MA": "New England",
        "MI": "Great Lakes",
        "MN": "Great Lakes",
        "MS": "Southeast",
        "MO": "Great Lakes",
        "MT": "Rocky Mountain",
        "NE": "Plains",
        "NV": "Southwest",
        "NH": "New England",
        "NJ": "Mideast",
        "NM": "Southwest",
        "NY": "Mideast",
        "NC": "Southeast",
        "ND": "Plains",
        "OH": "Great Lakes",
        "OK": "Southwest",
        "OR": "Far West",
        "PA": "Mideast",
        "RI": "New England",
        "SC": "Southeast",
        "SD": "Plains",
        "TN": "Southeast",
        "TX": "Southwest",
        "UT": "Rocky Mountain",
        "VT": "New England",
        "VA": "Mideast",
        "WA": "Far West",
        "WV": "Southeast",
        "WI": "Great Lakes",
        "WY": "Rocky Mountain"
    }
    region = states_to_bea_regions.get(state)
    return region

## taken from SciKit learn prints the outcome of a tuning pipline
def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_balanced_accuracy, std_balanced_accuracy, mean_precision, std_precision, params in zip(
        filtered_cv_results["mean_test_balanced_accuracy"],
        filtered_cv_results["std_test_balanced_accuracy"],
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["params"],
    ):
        print(
            f"balanced accuracy: {mean_balanced_accuracy:0.3f} (±{std_balanced_accuracy:0.03f}),"
            f" precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" for {params}"
        )
    print()

##Tuning strategy framework taken from Sci-Kit Learn and adjusted.

def refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a balanced_accuracy score threshold
    of 0.965, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # print the info about the grid-search for the different scores
    # precision_threshold = 0.75
    balanced_accuracy_threshold = 0.965

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    # high_precision_cv_results = cv_results_[
        # cv_results_["mean_test_precision"] > precision_threshold
    # ]

    high_balanced_accuracy_cv_results = cv_results_[
        cv_results_["mean_test_balanced_accuracy"] > balanced_accuracy_threshold
    ]

    # print(f"Models with a precision higher than {precision_threshold}:")
    print(f"Models with a balanced accuracy score higher than {balanced_accuracy_threshold}:")
    # print_dataframe(high_precision_cv_results)
    print_dataframe(high_balanced_accuracy_cv_results)

    # high_precision_cv_results = high_precision_cv_results[
    #     [
    #         "mean_score_time",
    #         "mean_test_recall",
    #         "std_test_recall",
    #         "mean_test_precision",
    #         "std_test_precision",
    #         "rank_test_recall",
    #         "rank_test_precision",
    #         "params",
    #     ]
    # 
    high_balanced_accuracy_cv_results = high_balanced_accuracy_cv_results[
        [
            "mean_score_time",
            "mean_test_precision",
            "std_test_precision",
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "rank_test_precision",
            "rank_test_balanced_accuracy",
            "params",
        ]
    ]
    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_precision_std = high_balanced_accuracy_cv_results["mean_test_precision"].std()
    best_precision = high_balanced_accuracy_cv_results["mean_test_precision"].max()
    best_precision_threshold = best_precision - best_precision_std

    # high_recall_cv_results = high_precision_cv_results[
    #     high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    # ]
    high_precision_cv_results = high_balanced_accuracy_cv_results[
        high_balanced_accuracy_cv_results["mean_test_precision"] > best_precision_threshold
    ]
    print(
        "Out of the previously selected high balanced accuracy models, we keep all\n"
        "the models within one standard deviation of the highest precision model:"
    )
    print_dataframe(high_precision_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_precision_high_balanced_accuracy_index = high_precision_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on balanced acurracy and precision.\n"
        "Its scoring time is:\n\n"
        f"{high_precision_cv_results.loc[fastest_top_precision_high_balanced_accuracy_index]}"
    )

    return fastest_top_precision_high_balanced_accuracy_index