import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import inspect
from collections import defaultdict
from matal.utils import cache_return, obj_to_hash, load_cache, save_cache, auto_log

import lz4.frame
from tqdm.auto import tqdm
from scipy.spatial.distance import pdist, cdist, squareform
import copy

try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings



def generate_random_compositions(*args, **kwargs):
    raise NotImplementedError()
    
    
class UniformCompositionGenerator:
    """
    Generating spaced-out compositions within design space while maximizing
    predicted performance using Monte Carlo method.
    """

    def __init__(self, cols=None, n_comps=30, n_iters=500000, comp_gen_func=None,
                 random_state=None, perf_model=None, perf_func=None, existing_comps=None,
                 boltzmann_const=1e5, checkpoint_freq=10, perf_coeff=1.0, dist_coeff=1.0, 
                 intra_dist_coeff=None, inter_dist_coeff=None, dist_col_weights=None, dist_col_scales=None,
                 **kwargs):
        self.cols = cols
        self.n_comps = n_comps
        self.n_iters = n_iters
        self.random_state = random_state
        self.comp_gen_func = comp_gen_func if comp_gen_func else generate_random_compositions
        
        if perf_func:
            self.perf_func = perf_func
        elif perf_model:
            self.perf_func = perf_model.predict
        else:
            self.perf_func = None
            
        self.existing_comps = existing_comps
        self.boltzmann_const = boltzmann_const
        self.checkpoint_freq = checkpoint_freq
        self.perf_coeff = perf_coeff
        self.dist_coeff = dist_coeff
        self.dist_col_weights = dist_col_weights if dist_col_weights is not None else np.ones(len(cols))
        self.dist_col_scales = dist_col_scales if dist_col_scales is not None else np.ones(len(cols))
        self.intra_dist_coeff = intra_dist_coeff if intra_dist_coeff is not None else dist_coeff
        self.inter_dist_coeff = inter_dist_coeff if inter_dist_coeff is not None else dist_coeff
        self.kwargs = kwargs

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                    key == "base_estimator"
                    and valid_params[key] == "deprecated"
                    and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                    " deprecated in favor of 'estimator'. See"
                    f" {self.__class__.__name__}'s docstring for more details.",
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self

    def _get_rand_comp(self, with_pred=False):
        ''' return a random composition '''

        # If the cache does not exist, or the cache is empty, rebuild cache
        if len(getattr(self, '_comp_cache', [])) == 0:
            self._build_comp_cache()

        # pop one composition, and with its prediction if existed, from cache
        comp = self._comp_cache.pop()
        pred = self._pred_cache.pop() if self.perf_func else None
        if with_pred:
            return comp, pred
        else:
            return comp

    def _build_comp_cache(self):
        ''' build cache containing a list of random composition '''
        
        comps, preds = None, None
        if comps is None:
            seed = self.rng_.integers(0, np.iinfo(np.int64).max)
            comps = self.comp_gen_func(n=self.n_comps, random_state=seed, **self.kwargs)
            preds = None
        self._comp_cache = [c for c in comps]
        
        if self.perf_func:
            if preds is None:
                auto_log(f'Calculating preds using {self.perf_func}', level='debug')
                self._pred_cache = [c for c in self.perf_func(self._comp_cache)]
            else:
                self._pred_cache = [c for c in preds]

    
    def optimize(self):
        pbar = tqdm(total=self.n_iters)

        # Initialize comps and calculate its score
        if not hasattr(self, 'iter_'):
            self.best_score_ = -np.inf
            self.opt_history_ = []
            self.iter_ = 0
            self.rng_ = np.random.default_rng(self.random_state)
            [self._get_rand_comp(with_pred=True) for _ in range(self.n_comps)]
            
            init_camps_preds = [self._get_rand_comp(with_pred=True) for _ in range(self.n_comps)]
            self.current_comps_ = [c for c, _ in init_camps_preds]
            self.current_predictions_ = [p for _, p in init_camps_preds]
            self.current_score_ = self.score_comps(self.current_comps_, 
                                                   perf_func=self.perf_func,
                                                   predictions=None if (self.perf_func is None) else self.current_predictions_)
        else:
            pbar.update(self.iter_)

        if self.perf_func and (not hasattr(self, 'current_predictions_')):
            # Keep a list of prediction, so we don't need to re-predict all of them
            # each time we changed one composition
            self.current_predictions_ = self.perf_func(self.current_comps_)

        checkpoint_i = 0
        while self.iter_ < self.n_iters:
            self.iter_ += 1
            pbar.update(1)

            if self.perf_func:
                # Randomly select a composition, compositions with lower
                # predicted performance are more likely to be selected.
                sorted_indices = np.argsort(self.current_predictions_)
                p = np.e ** (-0.1 * np.arange(len(sorted_indices)))
                pivot = self.rng_.choice(sorted_indices, p=p / p.sum())
            else:
                # Randomly select a composition
                pivot = self.rng_.integers(0, len(self.current_comps_))

            # Replace the composition with a new random one
            comps = copy.deepcopy(self.current_comps_)
            new_comp, new_pred = self._get_rand_comp(with_pred=True)
            comps[pivot] = new_comp

            # Update predictions
            if self.perf_func:
                predictions = copy.deepcopy(self.current_predictions_)
                predictions[pivot] = new_pred
            else:
                predictions = None

            # Update score
            score = self.score_comps(comps, predictions=predictions)

            if score > self.best_score_:
                self.current_score_ = score
                self.current_comps_ = comps
                if self.perf_func:
                    self.current_predictions_ = predictions
                self.opt_history_.append([self.iter_, pivot, score])

                if score > self.best_score_:
                    self.best_comps_ = comps
                    self.best_score_ = score
                    if self.perf_func:
                        self.best_predictions_ = predictions
                    checkpoint_i += 1
                    # if checkpoint_i % self.checkpoint_freq == 0:
                    #     self.save_checkpoint()

        # self.save_checkpoint()
        pbar.close()
        return pd.DataFrame(self.best_comps_, columns=self.cols)

    def score_comps(self, comps, perf_func=None, predictions=None, ):
        if predictions is None:
            if perf_func is None:
                predictions = [1.0]
            else:
                predictions = perf_func(comps)
        perf_score = np.mean(predictions)
        
        if (self.intra_dist_coeff > 0.0):
            min_intra_distance = pdist(comps * self.dist_col_scales, 
                                       'minkowski', p=2, w=self.dist_col_weights).min()
        else:
            min_intra_distance = 1.0
            
        if (self.inter_dist_coeff > 0.0) and (self.existing_comps is not None):
            min_inter_distance = cdist(self.existing_comps * self.dist_col_scales, 
                                       comps * self.dist_col_scales, 
                                       'minkowski',p=2, w=self.dist_col_weights).min()
        else:
            min_inter_distance = 1.0

        # score of a list of comps = (min paired-distance) * mean(predicted performance)
        score = (min_intra_distance ** self.intra_dist_coeff) * \
                (min_inter_distance ** self.inter_dist_coeff) * \
                (perf_score ** self.perf_coeff)
        return score

X_LIMS = np.array([35.0, 6.0, 1.0])


@ignore_warnings(category=UserWarning)
def filter_comp(comp, svm_models=None, scale=X_LIMS, kdtree=None, min_dist=0, min_comp=0, her_cutoff=-0.86, oer_cutoff=1.74, max_comp=1.0):
    comp = comp / scale
    if svm_models is not None:
        sol = svm_models['solubility'].predict(comp)
        her = svm_models['HER'].predict(comp)
        oer = svm_models['OER'].predict(comp)
        mask = (sol & (her >= her_cutoff) & (oer >= oer_cutoff)).astype(bool)
    else:
        mask = np.ones(len(comp)).astype(True)

    if (kdtree is not None) and (min_dist > 0.0):
        dist, min_idx = kdtree.query(comp)
        mask = mask & (dist > min_dist)

    if min_comp > 0.0:
        mask = mask & (comp.min(axis=1) > min_comp)

    if max_comp < 1.0:
        mask = mask & (comp.max(axis=1) < max_comp)

    return mask


def get_rand_comp(n=100, func=filter_comp, func_kw=dict(), batch_factor=8.0, max_iter=10, scale=X_LIMS,
                  random_state=None, rng=None,
                  extra_filter=None):
    if rng is None:
        rng = np.random.default_rng(random_state)
    comp = []
    for _ in range(max_iter):
        comp_i = (rng.random((round(n * batch_factor), len(scale))) * scale).round(2)
        if func is not None:
            mask = func(comp_i, **func_kw)
            comp_i = comp_i[mask]
        if extra_filter is not None:
            mask = extra_filter(comp_i)
            comp_i = comp_i[mask]
        if len(comp_i) > 0:
            comp = np.vstack((comp, comp_i)) if len(comp) > 0 else comp_i
        if len(comp) >= n:
            break
    return comp[:n]
