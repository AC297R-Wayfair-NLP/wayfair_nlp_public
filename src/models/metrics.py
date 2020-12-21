import tensorflow as tf
import numpy as np

# Asymmetric mean squared error
def asymmetric_mse(y_actual,y_pred):
    y_true = tf.cast(y_actual, y_pred.dtype)
    asymmetric_mse=tf.square(y_pred-y_actual)*tf.square(tf.sign(y_pred-y_actual)+0.3)
    return asymmetric_mse


def bootstrap(full_model, non_nlp_model, X_full, y, score_func, n_boot=100, nlp_cols=None):
    """Resamples X to calculate `n_boot` pairs of full and non-nlp model scores
    
    Args:
        full_model (model): must have .predict method
        non_nlp_model (model): must have .predict method
        X_full (pd.DataFrame): full X dataframe including NLP columns
        y (array-like): target variables
        score_func (function): must have argument `score_func(y_true, y_pred)`
        n_boot (int): number of bootstrap iterations
        nlp_cols (list): list of NLP columns. See code for default value
    """
    if nlp_cols is None:
        nlp_cols = ['compound', 'emb1', 'emb10', 'emb11', 'emb12', 'emb13', 'emb14',
                    'emb15', 'emb16', 'emb2', 'emb3', 'emb4', 'emb5', 'emb6', 'emb7',
                    'emb8', 'emb9', 'neg', 'neu', 'pos', 'subjectivity', 'topic_18',
                    'topic_6']
    # get predictions
    X_non_nlp = X_full.drop(nlp_cols, axis=1)
    y_pred_full = full_model.predict(X_full)
    y_pred_non_nlp = non_nlp_model.predict(X_non_nlp)
    X_non_nlp = np.array(X_non_nlp)

    # resample test set
    full_scores = []
    non_nlp_scores = []
    for i in range(n_boot):
        boot_idxs = np.random.choice(X_full.shape[0], size=X_full.shape[0], replace=True)
        X_boot = X_full.iloc[boot_idxs]
        y_true_boot = y.iloc[boot_idxs]
        y_pred_full_boot = y_pred_full[boot_idxs]
        y_pred_non_nlp_boot = y_pred_non_nlp[boot_idxs]

        full_scores.append(score_func(y_true_boot, y_pred_full_boot))
        non_nlp_scores.append(score_func(y_true_boot, y_pred_non_nlp_boot))

    return np.array(full_scores), np.array(non_nlp_scores)