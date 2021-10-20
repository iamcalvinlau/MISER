import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

###############################################################
# Functions for creating smooth polynomial fits
###############################################################
def get_degrees_per_feature(poly_in, dimensions=2):
    # Make the names based on easy to use
    if dimensions == 2:
        variables = ["a", "b"]
    elif dimensions == 2:
        variables = ["a", "b", "c"]
    feat_names = poly_in.get_feature_names(variables)
    degrees = np.zeros((len(feat_names), len(variables)), dtype=int)
    for i, feat_name in enumerate(feat_names):
        for j, dim in enumerate(variables):
            # Check for first dimension
            if variables[j] not in feat_name:
                degrees[i, j] = 0
            elif (variables[j] in feat_name) and (variables[j] + "^" not in feat_name):
                degrees[i, j] = 1
            elif variables[j] + "^" in feat_name:
                degrees[i, j] = feat_name[
                    feat_name.find(variables[j] + "^") + len(variables[j] + "^")
                ]
    return degrees

def weighted_multivariate_polyfit(X, y, w, poly_degree=1, thresh=1e-10):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Set up multivariate polynomial base from input X
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    # fit WLS using weights
    WLS = LinearRegression()
    WLS.fit(X_poly, y, sample_weight=w)
    # kill any coefficients below threshold
    WLS.coef_[abs(WLS.coef_) < thresh] = 0.0
    # dictionary will contain coefficients, feature names, score(?)
    WMP = dict(
        coefs=np.copy(WLS.coef_),
        degrees=get_degrees_per_feature(poly, dimensions=X.shape[1]),
        X=np.copy(X),
    )
    # not sure why, but the coefficient seems to be wrong for the bias
    WLS_mean = np.sum(weighted_multivariate_polypredict(WMP, dx=[0, 0]) * w) / np.sum(w)
    f_mean = np.sum(y * w) / np.sum(w)
    WMP['coefs'][0] += f_mean - WLS_mean
    return WMP

def weighted_multivariate_polypredict(WMP, dx=[0, 0]):
    coefs = np.copy(WMP["coefs"])
    degrees = np.copy(WMP["degrees"])
    for i in range(degrees.shape[1]):
        for j in range(dx[i]):
            coefs[:] *= degrees[:, i]
            degrees[:, i] -= 1
            degrees[degrees[:, i] < 0, i] = 0
    pred = np.zeros_like(WMP["X"][:, 0])
    for i in range(degrees.shape[0]):
        this_term = np.ones_like(pred)
        for j in range(degrees.shape[1]):
            this_term[:] *= WMP["X"][:, j] ** degrees[i, j]
        pred[:] += coefs[i] * this_term
    return pred

###############################################################
# Class for sparse identification using polynomial fit approach
###############################################################
class sparse_identifier_polyfit:    
    def __init__(
        self,
        data, data_coords,
        N_iterations = 10,
        N_terms = [1,2,3,4,5],
        N_samples = 100,
        poly_degree = 3,
        i_pow = 3,
        j_pow = 3,
        i_frac = 0.1,
        j_frac = 0.1,
        RNG_seed = 1234
    ):
        # set random seed
        np.random.seed(RNG_seed)
    
        # Data to be modeled
        self.data = data
        self.data_coords = data_coords
        
        # Number of iterations used to sample
        self.N_iterations = N_iterations
        
        # Number of terms desired
        self.N_terms = N_terms
        
        # Number of samples used for identification
        # (Note that this is roughly what the number of samples are
        # since there is a uniqueness call due to the way SINDy is used)
        self.N_samples = N_samples
        
        # The degree of the polynomial used for fitting
        self.poly_degree = poly_degree
        
        # Integral weighting set up
        self.i_pow = i_pow
        self.j_pow = j_pow
        self.i_frac = i_frac
        self.j_frac = j_frac
        self.i_int_size = int(self.data.shape[0]*self.i_frac)
        self.j_int_size = int(self.data.shape[1]*self.j_frac)
        if(self.i_int_size%2==0):self.i_int_size+=1
        if(self.j_int_size%2==0):self.j_int_size+=1
        self._integral_weight()
        
    def _integral_weight(self):
        j_int_mesh, i_int_mesh = np.meshgrid(
            np.arange(self.j_int_size) - (self.j_int_size//2),
            np.arange(self.i_int_size) - (self.i_int_size//2),
        )
        i_int_weight = abs( i_int_mesh ).astype(float)
        i_int_weight /= np.max(i_int_weight)
        i_int_weight = np.power(1 - i_int_weight, self.i_pow)

        j_int_weight = abs( j_int_mesh ).astype(float)
        j_int_weight /= np.max(j_int_weight)
        j_int_weight = np.power(1 - j_int_weight, self.j_pow)

        integral_weight = i_int_weight * j_int_weight
        self.integral_weight = integral_weight / np.sum(integral_weight)
        
    def _select_eval_batch(self):
        i0 = 0; i1 = self.data.shape[0]
        j0 = 0; j1 = self.data.shape[1]
        # Note the "unique" call due to the "t" dimension requiring
        # uniqueness in SINDy
        eval_j0 = (
            np.unique(
                np.random.choice(
                    np.arange(j0, j1-self.j_int_size, 1), 
                    size=self.N_samples
                )
            )
        )
        eval_i0 = (
            (
                np.random.choice(
                    np.arange(i0, i1-self.i_int_size, 1), 
                    size=eval_j0.size
                )
            )
        )
        return eval_i0, eval_j0
    
    def _calculate_weighted_terms(self):
        inputs = dict({
            "f" : [], 
            "f_t" : [], 
            "f_tt" : [], 
            "f_x" : [], 
            "f_xx" : [],
            "f f" : [],
            "f f_t" : [],
            "f f_x" : [],
            "f f_tt" : [],
            "f f_xx" : [],
            "f_t f_t" : [],
            "f_t f_x" : [],
            "f_x f_x" : [],
            "t" : [], 
            "x" : []
        })
        eval_i0, eval_j0=self._select_eval_batch()
        for _i, _j in zip(eval_i0[:], eval_j0[:]):
            # The base data position
            inputs["x"].append(self.data_coords[0][_i]);
            inputs["t"].append(self.data_coords[1][_j]);
            # The data positions that are being averaged over
            _x_1d = self.data_coords[0][_i:_i+self.i_int_size]
            _t_1d = self.data_coords[1][_j:_j+self.j_int_size]
            _tt, _xx = np.meshgrid(
                 _t_1d, _x_1d,
            )
            _X = np.vstack((_xx.flatten(), _tt.flatten())).T

            # Find the weighted polyfit 
            WMP = weighted_multivariate_polyfit(
                _X, 
                self.data[_i:_i+self.i_int_size,_j:_j+self.j_int_size].flatten(), 
                self.integral_weight.flatten(), 
                poly_degree=self.poly_degree
            )

            # Calculate the derivatives and values from the polyfit
            _f = weighted_multivariate_polypredict(WMP, dx=[0, 0]).reshape(_xx.shape)
            _f_x = weighted_multivariate_polypredict(WMP, dx=[1, 0]).reshape(_xx.shape)
            _f_xx = weighted_multivariate_polypredict(WMP, dx=[2, 0]).reshape(_xx.shape)
            _f_t = weighted_multivariate_polypredict(WMP, dx=[0, 1]).reshape(_xx.shape)
            _f_tt = weighted_multivariate_polypredict(WMP, dx=[0, 2]).reshape(_xx.shape)

            # Calculate the weighted values
            inputs["f"].append( np.sum(_f * self.integral_weight) )
            inputs["f_t"].append( np.sum(_f_t * self.integral_weight) )
            inputs["f_tt"].append( np.sum(_f_tt * self.integral_weight) )
            inputs["f_x"].append( np.sum(_f_x * self.integral_weight) )
            inputs["f_xx"].append( np.sum(_f_xx * self.integral_weight) )

            inputs["f f"].append( np.sum(_f * _f * self.integral_weight) )
            inputs["f f_t"].append( np.sum(_f * _f_t * self.integral_weight) )
            inputs["f f_tt"].append( np.sum(_f * _f_tt * self.integral_weight) )
            inputs["f f_x"].append( np.sum(_f * _f_x * self.integral_weight) )
            inputs["f f_xx"].append( np.sum(_f * _f_xx * self.integral_weight) )

            inputs["f_t f_t"].append( np.sum(_f_t * _f_t * self.integral_weight) )
            inputs["f_t f_x"].append( np.sum(_f_t * _f_x * self.integral_weight) )
            inputs["f_x f_x"].append( np.sum(_f_x * _f_x * self.integral_weight) )
            
        return inputs
        
    def _setup_for_SINDy(self, inputs):
        U = []
        feat_names = []
        # Don't want x, t, f_t 
        for key, val in inputs.items():
            if("f" in key):
                if(
                    ("f_t" == key)
                ):
                    continue
                else:
                    U.append(val)
                    feat_names.append(key)
        # Since we are using SINDy in a silly way,
        # the inputs are all "control" and not state vectors
        feat_names = ['f_t'] + feat_names
        U = np.array(U).T
        X_dot = np.array([inputs['f_t']]).T
        X = np.array([inputs['x']]).T
        T = np.array([inputs['t']]).T
        
        return X, X_dot, U, T, feat_names
    
    def _get_label(self, model):
        ind = np.argsort(abs(model.coefficients().flatten()[1:]))[::-1]
        feature_names = np.array(model.feature_names[1:])[ind].flatten()
        coefs = model.coefficients().flatten()[1:][ind].flatten()
        label = "u_t = "
        for ifig, (feat_name, coef) in enumerate(zip(
            feature_names, coefs
        )):  
            if(abs(coef)>0):
                if(ifig > 0 ):
                    label += " + "
                label += "{:.3f} ".format(coef) + feat_name
        return label
    
    def _sparse_identification(self):
        from optimizers import STLSQ_mod
        # Set up with the necessary values
        inputs = self._calculate_weighted_terms()
        X, X_dot, U, T, feat_names = self._setup_for_SINDy(inputs)
        
        # Calculate the coefficients for desired number of terms
        score = []
        coefs = []
        eqns = []
        for num_terms in self.N_terms:
            lib = ps.IdentityLibrary()
            opt = STLSQ_mod(
                threshold=np.power(10,0.0), 
                alpha=1e-10, 
                ridge_kw=dict(
                    tol=1e-10, 
                #     solver='auto', 
                    max_iter=100000
                ), 
                target_number_terms=num_terms, 
            )
            model = ps.SINDy(
                feature_names=feat_names,
                optimizer=opt,
                feature_library=lib
            )
            model.fit(X, t=T, x_dot=X_dot, u=U)
            score.append(model.score(X, t=T, x_dot=X_dot, u=U))
            coefs.append(model.coefficients().flatten().flatten())
            eqns.append(self._get_label(model))
        score = np.array(score)
        coefs = np.array(coefs)
        
        return score, coefs, eqns, feat_names, X.shape[0]
    
    def iterate_sparse_identification(self):
        scores = []
        coefficients = []
        N_samples_actual = []
        equations = []
        for i in range(self.N_iterations):
            score, coefs, eqns, feat_names, N_samples = self._sparse_identification()
            scores.append(score)
            coefficients.append(coefs)
            equations.append(eqns)
            N_samples_actual.append(N_samples)
        self.scores = np.array(scores)
        self.coefficients = np.array(coefficients)
        self.feature_names = feat_names
        self.N_samples_actual = np.array(N_samples_actual)
        self.equations = equations
