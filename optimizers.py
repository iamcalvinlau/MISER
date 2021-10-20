import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression, Ridge, Lasso, ElasticNet
from sklearn.utils.validation import check_is_fitted

from pysindy import BaseOptimizer, STLSQ


class STLSQ_mod(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.
    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight array w that are below a given threshold.
    See the following reference for more details:
        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.
    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.
    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.
    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.
    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.
    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.
    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.
    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.
    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        target_number_terms=3,
        alpha=0.05,
        max_iter=20,
        ridge_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
    ):
        super(STLSQ_mod, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.target_n_terms = target_number_terms
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.initial_guess = initial_guess

    def _sparse_coefficients(self, dim, ind, coef, threshold, drop_terms=True):
        """Perform thresholding of the weight vector(s)"""
        # print(dim,coef.shape,ind.shape)
        c = np.zeros(dim)
        c[ind] = coef
        if(drop_terms):
            # remove lowest value coefficient
            big_ind = np.abs(c) > np.abs(coef).min()
            c[~big_ind] = 0
        else:
            big_ind = np.abs(c) > 0 
        return c, big_ind

    def _regress(self, x, y, iterate=True):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        if(iterate):self.iters += 1
        return coef
    
    def _sparse_regress(self, x, y, iterate=True):
        """Perform ridge regression for every combination and choose best"""
        kw = self.ridge_kw or {}
        # regr_ = Ridge(self.alpha, **kw)
        # regr_.fit(x, y);
        # # This uses an R2 best fit
        # score_orig = regr_.score(x, y);
        
        score = np.zeros_like(x[0,:])
        coef_total = np.zeros_like(score)
        coefs = []
        # Calculate the score after every possible elimination
        for i in range(score.shape[0]):
            # remove slice along second axis
            x_test = np.delete(x, i, 1)
            regr_ = Ridge(self.alpha, **kw)
            # regr_ = Lasso(self.alpha, **kw)
            # regr_ = ElasticNet(self.alpha, **kw)
            regr_.fit(x_test, y);
            coefs.append(regr_.coef_);
            # This uses an R2 best fit
            score[i] = regr_.score(x_test, y);
            # This finds the average coef (normalized so that 1 is max);
            coef_total[i] = np.sum(
                # np.square(
                    abs(regr_.coef_) / abs(regr_.coef_).max()
                # )
            )
        # print(coef_total)
        # Choose the elimination that has the highest score
        # and highest coef total (so it doesn't want badly fitting terms)
        i_best = np.argmax(score * coef_total);
        # Use the coefficients of that best elimination,
        # making sure to stick a zero in place of the eliminated coefficient
        coef = np.asarray(coefs[i_best]); 
        coef = np.insert(coef, i_best, 0);
        if(iterate):self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.
        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """

        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)
        
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess
        else:
            self.coef_ = (np.random.rand(n_targets, n_features) - 0.5) * 2.0

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                
                coef_i = self._sparse_regress(x[:, ind[i]], y[:, i])
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )                
                coef[i] = coef_i
                ind[i] = ind_i
                
            self.history_.append(coef)
            if( 
#                 (np.sum(ind) == n_features_selected) or 
#                 (self._no_change()) or 
                (np.count_nonzero(ind) == self.target_n_terms)
            ):
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )