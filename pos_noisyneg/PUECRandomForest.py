"""Bagging meta-estimator for PU learning.
Any scikit-learn estimator should work as the base estimator.
This implementation is fully compatible with scikit-learn, and is in fact based
on the code of the sklearn.ensemble.BaggingClassifier class with very minor
changes.
"""

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause
#
#
# Adapted for PU learning by Roy Wright <roy.w.wright@gmail.com>
# (work in progress)
#
# A better idea: instead of a separate PU class, modify the original
# sklearn BaggingClassifier so that the parameters `max_samples`
# and `bootstrap` may be lists or dicts...
# e.g. for a PU problem with 500 positives and 10000 unlabeled, we might set
# max_samples = [500, 500]     (to balance P and U in each bag)
# bootstrap = [True, False]    (to only bootstrap the unlabeled)


from __future__ import division

import itertools
import numbers
from warnings import warn
from abc import ABCMeta, abstractmethod

import numpy as np
# we can assume joblib is present because it's required by sklearn anyway
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from six import with_metaclass
from six.moves import zip
from six import string_types

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import (
    check_random_state, check_X_y, check_array, column_or_1d,
)
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils import indices_to_mask, check_consistent_length
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.ensemble import BaseEnsemble
from sklearn.base import BaseEstimator

from sklearn.ensemble._base import _partition_estimators

__all__ = ["ECDecisionTreeClassifier",
           "PUECBaggingClassifier"]

MAX_INT = np.iinfo(np.int32).max

def cost_loss(y_true, y_pred, cost_mat):
    #TODO: update description
    """ Cost classification loss
    This function calculates the cost of using y_pred on y_true with
    cost-matrix cost-mat. It differ from traditional classification evaluation
    measures since measures such as accuracy asing the same cost to different
    errors, but that is not the real case in several real-world classification
    problems as they are example-dependent cost-sensitive in nature, where the
    costs due to misclassification vary between examples.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    Returns
    -------
    loss : float
        Cost of a using y_pred on y_true with cost-matrix cost-mat
    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.
    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.
    See also
    --------
    savings_score
    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> cost_loss(y_true, y_pred, cost_mat)
    3
    """

    #TODO: Check consistency of cost_mat

    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(np.float)
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(np.float)
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)

class ECDecisionTreeClassifier(BaseEstimator):
    """A example-dependent cost-sensitive binary decision tree classifier.
    Parameters
    ----------
    criterion : string, optional (default="direct_cost")
        The function to measure the quality of a split. Supported criteria are
        "direct_cost" for the Direct Cost impurity measure, "pi_cost", "gini_cost",
        and "entropy_cost".
    criterion_weight : bool, optional (default=False)
        Whenever or not to weight the gain according to the population distribution.
    num_pct : int, optional (default=100)
        Number of percentiles to evaluate the splits for each feature.
    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_samples_leaf`` is not None.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.
    min_gain : float, optional (default=0.001)
        The minimum gain that a split must produce in order to be taken into account.

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.
    See also
    --------
    sklearn.tree.DecisionTreeClassifier
    References
    ----------
    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications" <http://albahnsen.com/files/Example-Dependent%20Cost-Sensitive%20Decision%20Trees.pdf>`__,
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveDecisionTreeClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveDecisionTreeClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CSDecisionTree
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.481916135529
    """

    def __init__(self,
                 criterion='direct_cost',
                 criterion_weight=False,
                 num_pct=100,
                 max_features=None,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_gain=0.001,
                 ):

        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

        self.n_features_ = None
        self.max_features_ = None

        self.tree_ = []

    def set_param(self, attribute, value):
        setattr(self, attribute, value)
        
        
    class _tree_class():
        def __init__(self):
            self.n_nodes = 0
            self.tree = dict()
            self.nodes = []

    def _node_cost(self, y_true, cost_mat):
        """ Private function to calculate the cost of a node.
        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.
        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.
        Returns
        -------
        tuple(cost_loss : float, node prediction : int, node predicted probability : float)
        """
        n_samples = len(y_true)

        # Evaluates the cost by predicting the node as positive and negative
        costs = np.zeros(2)
        costs[0] = cost_loss(y_true, np.zeros(y_true.shape), cost_mat)
        costs[1] = cost_loss(y_true, np.ones(y_true.shape), cost_mat)

        if self.criterion == 'direct_cost':
            costs = costs

        y_pred = np.argmin(costs)

        # Calculate the predicted probability of a node using laplace correction.
        n_positives = y_true.sum()
        y_prob = (n_positives + 1.0) / (n_samples + 2.0)

        return costs[y_pred], y_pred, y_prob

    def _calculate_gain(self, cost_base, y_true, X, cost_mat, split):
        """ Private function to calculate the gain in cost of using split in the
         current node.
        Parameters
        ----------
        cost_base : float
            Cost of the naive prediction
        y_true : array indicator matrix
            Ground truth (correct) labels.
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.
        split : tuple of len = 2
            split[0] = feature to split = j
            split[1] = where to split = l
        Returns
        -------
        tuple(gain : float, left node prediction : int)
        """

        # Check if cost_base == 0, then no gain is possible
        # TODO: This must be check in _best_split
        if cost_base == 0.0:
            # In case cost_b==0 and pi_1!=(0,1)
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)

        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples, n_features = X.shape

        # Check if one of the leafs is empty
        # TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in [0, n_samples]:  # One leaft is empty
            return 0.0, 0.0

        # Split X in Xl and Xr according to rule split
        Xl_cost, Xl_pred, _ = self._node_cost(
            y_true[filter_Xl], cost_mat[filter_Xl, :])
        Xr_cost, _, _ = self._node_cost(
            y_true[filter_Xr], cost_mat[filter_Xr, :])

        if self.criterion_weight:
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            Xl_w = n_samples_Xl * 1.0 / n_samples
            Xr_w = 1 - Xl_w
            gain = round(
                (cost_base - (Xl_w * Xl_cost + Xr_w * Xr_cost)) / cost_base, 6)
        else:
            gain = round((cost_base - (Xl_cost + Xr_cost)) / cost_base, 6)

        return gain, Xl_pred

    def _best_split(self, y_true, X, cost_mat):
        """ Private function to calculate the split that gives the best gain.
        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.
        Returns
        -------
        tuple(split : tuple(j, l), gain : float, left node prediction : int,
              y_pred : int, y_prob : float)
        """

        n_samples, n_features = X.shape
        num_pct = self.num_pct

        cost_base, y_pred, y_prob = self._node_cost(y_true, cost_mat)

        # Calculate the gain of all features each split in num_pct
        gains = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        # Add random state
        np.random.shuffle(selected_features)
        selected_features = selected_features[:self.max_features_]
        selected_features.sort()

        # TODO:  # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        # For each feature test all possible splits
        for j in selected_features:
            splits[j, :] = np.percentile(
                X[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())

            for l in range(num_pct):
                # Avoid repeated values, since np.percentile may return repeated values
                if l == 0 or (l > 0 and splits[j, l] != splits[j, l - 1]):
                    split = (j, splits[j, l])
                    gains[j, l], pred[j, l] = self._calculate_gain(
                        cost_base, y_true, X, cost_mat, split)

        best_split = np.unravel_index(gains.argmax(), gains.shape)

        return (best_split[0], splits[best_split]), gains.max(), pred[best_split], y_pred, y_prob

    def _tree_grow(self, y_true, X, cost_mat, level=0):
        """ Private recursive function to grow the decision tree.
        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.
        Returns
        -------
        Tree : Object
            Container of the decision tree
            NOTE: it is not the same structure as the sklearn.tree.tree object
        """

        # TODO: Find error, add min_samples_split
        if len(X.shape) == 1:
            tree = dict(y_pred=y_true, y_prob=0.5, level=level,
                        split=-1, n_samples=1, gain=0)
            return tree

        # Calculate the best split of the current node
        split, gain, Xl_pred, y_pred, y_prob = self._best_split(
            y_true, X, cost_mat)

        n_samples, n_features = X.shape

        # Construct the tree object as a dictionary

        # TODO: Convert tree to be equal to sklearn.tree.tree object
        tree = dict(y_pred=y_pred, y_prob=y_prob, level=level,
                    split=-1, n_samples=n_samples, gain=gain)

        # Check the stopping criteria
        if gain < self.min_gain:
            return tree
        if self.max_depth is not None:
            if level >= self.max_depth:
                return tree
        if n_samples <= self.min_samples_split:
            return tree

        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if min(n_samples_Xl, n_samples_Xr) <= self.min_samples_leaf:
            return tree

        # No stooping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(
            y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(
            y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    def fit(self, X, y, cost_mat, check_input=False):
        """ Build a example-dependent cost-sensitive decision tree from the training set (X, y, cost_mat)
        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        self : object
            Returns self.
        """

        # TODO: Check input
        # TODO: Add random state
        n_samples, self.n_features_ = X.shape

        self.tree_ = self._tree_class()

        # Maximum number of features to be taken into account per split
        if isinstance(self.max_features, string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_))
            else:
                max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        self.classes_ = np.array([0, 1])

        return self

    def _nodes(self, tree):
        """ Private function that find the number of nodes in a tree.
        Parameters
        ----------
        tree : object
        Returns
        -------
        nodes : array like of shape [n_nodes]
        """
        def recourse(temp_tree_, nodes):
            if isinstance(temp_tree_, dict):
                if temp_tree_['split'] != -1:
                    nodes.append(temp_tree_['node'])
                    if temp_tree_['split'] != -1:
                        for k in ['sl', 'sr']:
                            recourse(temp_tree_[k], nodes)
            return None

        nodes_ = []
        recourse(tree, nodes_)
        return nodes_

    def _classify(self, X, tree, proba=False):
        """ Private function that classify a dataset using tree.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        tree : object
        proba : bool, optional (default=False)
            If True then return probabilities else return class
        Returns
        -------
        prediction : array of shape = [n_samples]
            If proba then return the predicted positive probabilities, else return
            the predicted class for each example in X
        """

        n_samples, n_features = X.shape
        predicted = np.ones(n_samples)

        # Check if final node
        if tree['split'] == -1:
            if not proba:
                predicted = predicted * tree['y_pred']
            else:
                predicted = predicted * tree['y_prob']
        else:
            j, l = tree['split']
            filter_Xl = (X[:, j] <= l)
            filter_Xr = ~filter_Xl
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

            if n_samples_Xl == 0:  # If left node is empty only continue with right
                predicted[filter_Xr] = self._classify(
                    X[filter_Xr, :], tree['sr'], proba)
            elif n_samples_Xr == 0:  # If right node is empty only continue with left
                predicted[filter_Xl] = self._classify(
                    X[filter_Xl, :], tree['sl'], proba)
            else:
                predicted[filter_Xl] = self._classify(
                    X[filter_Xl, :], tree['sl'], proba)
                predicted[filter_Xr] = self._classify(
                    X[filter_Xr, :], tree['sr'], proba)

        return predicted

    def predict(self, X):
        """ Predict class of X.
        The predicted class for each sample in X is returned.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes,
        """
        # TODO: Check consistency of X
    
        tree_ = self.tree_.tree

        return self._classify(X, tree_, proba=False)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        prob : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        # TODO: Check consistency of X
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, 2))

        tree_ = self.tree_.tree

        prob[:, 1] = self._classify(X, tree_, proba=True)
        prob[:, 0] = 1 - prob[:, 1]

        return prob

def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features)
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, cost_mat, 
                               sample_weight, seeds, total_n_estimators, 
                               verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # # ============ MAIN MODIFICATION FOR PU LEARNING =============
        # iP = [pair[0] for pair in enumerate(y) if pair[1] == 1]
        # iU = [pair[0] for pair in enumerate(y) if pair[1] < 1]
        # features, indices = _generate_bagging_indices(random_state,
        #                                               bootstrap_features,
        #                                               bootstrap, n_features,
        #                                               len(iU), max_features,
        #                                               max_samples)
        # indices = [iU[i] for i in indices] + iP
        # # ============================================================
        
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples)
        
        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, cost_mat, sample_weight=curr_sample_weight)

        # Draw samples, using a mask, and then fit
        else:
            estimator.fit((X[indices])[:, features], y[indices], cost_mat[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:  # pragma: no cover
                proba[:, estimator.classes_] += \
                    proba_estimator[:, range(len(estimator.classes_))]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba


def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute log probabilities within a job."""
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    all_classes = np.arange(n_classes, dtype=np.int)

    for estimator, features in zip(estimators, estimators_features):
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:  # pragma: no cover
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))])

            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing],
                                                 -np.inf)

    return log_proba


def _parallel_decision_function(estimators, estimators_features, X):
    """Private function used to compute decisions within a job."""
    return sum(estimator.decision_function(X[:, features])
               for estimator, features in zip(estimators,
                                              estimators_features))


class BaseBaggingPUEC(with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for Bagging PU meta-estimator.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=True,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaseBaggingPUEC, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, cost_mat, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (1 for positive, 0 for unlabeled).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Returns self.
        """
        return self._fit(X, y, cost_mat, max_samples=self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, cost_mat, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (1 for positive, 0 for unlabeled).
        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        self.y = y

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc'])
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:  # pragma: no cover
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:  # pragma: no cover
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * sum(y < 1))

        if not (0 < max_samples <= sum(y < 1)):
            raise ValueError(
                "max_samples must be positive"
                " and no larger than the number of unlabeled points")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:  # pragma: no cover
            del self.oob_score_  # pragma: no covr

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:  # pragma: no cover
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        if n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:  # pragma: no cover
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                cost_mat,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):  # pragma: no cover
        # Default implementation
        return column_or_1d(y, warn=True)

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            random_state = np.random.RandomState(seed)

            # # ============ MAIN MODIFICATION FOR PU LEARNING =============
            # iP = [pair[0] for pair in enumerate(self.y) if pair[1] == 1]
            # iU = [pair[0] for pair in enumerate(self.y) if pair[1] < 1]

            # feature_indices, sample_indices = _generate_bagging_indices(
            #     random_state, self.bootstrap_features, self.bootstrap,
            #     self.n_features_, len(iU), self._max_features,
            #     self._max_samples)

            # sample_indices = [iU[i] for i in sample_indices] + iP
            # # ============================================================
            feature_indices, sample_indices = _generate_bagging_indices(
                random_state, self.bootstrap_features, self.bootstrap,
                self.n_features_, self._n_samples, self._max_features,
                self._max_samples)

            
            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        """The subset of drawn samples for each base estimator.
        Returns a dynamically generated list of boolean masks identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.
        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        sample_masks = []
        for _, sample_indices in self._get_estimators_indices():
            mask = indices_to_mask(sample_indices, self._n_samples)
            sample_masks.append(mask)

        return sample_masks


class PUECBaggingClassifier(BaseBaggingPUEC, ClassifierMixin):
    """A Bagging PU classifier.
    Adapted from sklearn.ensemble.BaggingClassifier, based on
    A bagging SVM to learn from positive and unlabeled examples (2013)
    by Mordelet and Vert
    http://dx.doi.org/10.1016/j.patrec.2013.06.010
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Mordelet2013bagging.pdf
    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.
    max_samples : int or float, optional (default=1.0)
        The number of unlabeled samples to draw to train each base estimator.
    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.
    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.
    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.
    oob_score : bool, optional (default=True)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.
    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.
    classes_ : array of shape = [n_classes]
        The classes labels.
    n_classes_ : int or list
        The number of classes.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. Positive data points, and perhaps some of the unlabeled,
        are left out during the bootstrap. In these cases,
        `oob_decision_function_` contains NaN.
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=True,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super(PUECBaggingClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(PUECBaggingClassifier, self)._validate_estimator(
            default=ECDecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        # Modified: no warnings about non-OOB points (i.e. positives)
        with np.errstate(invalid='ignore'):
            denominator = predictions.sum(axis=1)[:, np.newaxis]
            oob_decision_function = predictions / denominator
            oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, "classes_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, "classes_")
        if hasattr(self.base_estimator_, "predict_log_proba"):
            # Check data
            X = check_array(X, accept_sparse=['csr', 'csc'])

            if self.n_features_ != X.shape[1]:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is {0} "
                                 "and input n_features is {1} "
                                 "".format(self.n_features_, X.shape[1]))

            # Parallel loop
            n_jobs, n_estimators, starts = _partition_estimators(
                self.n_estimators, self.n_jobs)

            all_log_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_log_proba)(
                    self.estimators_[starts[i]:starts[i + 1]],
                    self.estimators_features_[starts[i]:starts[i + 1]],
                    X,
                    self.n_classes_)
                for i in range(n_jobs))

            # Reduce
            log_proba = all_log_proba[0]

            for j in range(1, len(all_log_proba)):  # pragma: no cover
                log_proba = np.logaddexp(log_proba, all_log_proba[j])

            log_proba -= np.log(self.n_estimators)

            return log_proba
        # else, the base estimator has no predict_log_proba, so...
        return np.log(self.predict_proba(X))

    @if_delegate_has_method(delegate='base_estimator')
    def decision_function(self, X):
        """Average of the decision functions of the base classifiers.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.
        """
        check_is_fitted(self, "classes_")

        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1} "
                             "".format(self.n_features_, X.shape[1]))

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # Reduce
        decisions = sum(all_decisions) / self.n_estimators

        return decisions