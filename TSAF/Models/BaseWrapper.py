"""

@author: viniciussantino
"""


class BaseWrapper(object):
    """Base class for the Model TSAF wrapper.

      Warning: This class should not be used directly.
      Use descendant classes instead.
    """
    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.check_params(sk_params)

  def check_params(self, params):
    """Checks for user typos in `params`.

    Arguments:
        params: dictionary; the parameters to be checked

    Raises:
        ValueError: if any member of `params` is not a valid argument.
    """
    legal_params_fns = [
        Sequential.fit, Sequential.predict, Sequential.predict_classes,
        Sequential.evaluate
    ]
    if self.build_fn is None:
      legal_params_fns.append(self.__call__)
    elif (not isinstance(self.build_fn, types.FunctionType) and
          not isinstance(self.build_fn, types.MethodType)):
      legal_params_fns.append(self.build_fn.__call__)
    else:
      legal_params_fns.append(self.build_fn)

    for params_name in params:
      for fn in legal_params_fns:
        if has_arg(fn, params_name):
          break
      else:
        if params_name != 'nb_epoch':
          raise ValueError('{} is not a legal parameter'.format(params_name))

  def get_params(self, **params):  # pylint: disable=unused-argument
    """Gets parameters for this estimator.

    Arguments:
        **params: ignored (exists for API compatibility).

    Returns:
        Dictionary of parameter names mapped to their values.
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

  def set_params(self, **params):
    """Sets the parameters of this estimator.

    Arguments:
        **params: Dictionary of parameter names mapped to their values.

    Returns:
        self
    """
    self.check_params(params)
    self.sk_params.update(params)
    return self

  def fit(self, x, y, **kwargs):
    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

    Arguments:
        x : array-like, shape `(n_samples, n_features)`
            Training samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.fit`

    Returns:
        history : object
            details about the training history at each epoch.
    """
    if self.build_fn is None:
      self.model = self.__call__(**self.filter_sk_params(self.__call__))
    elif (not isinstance(self.build_fn, types.FunctionType) and
          not isinstance(self.build_fn, types.MethodType)):
      self.model = self.build_fn(
          **self.filter_sk_params(self.build_fn.__call__))
    else:
      self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

    if (losses.is_categorical_crossentropy(self.model.loss) and
        len(y.shape) != 2):
      y = to_categorical(y)

    fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
    fit_args.update(kwargs)

    history = self.model.fit(x, y, **fit_args)

    return history

  def filter_sk_params(self, fn, override=None):
    """Filters `sk_params` and returns those in `fn`'s arguments.

    Arguments:
        fn : arbitrary function
        override: dictionary, values to override `sk_params`

    Returns:
        res : dictionary containing variables
            in both `sk_params` and `fn`'s arguments.
    """
    override = override or {}
    res = {}
    for name, value in self.sk_params.items():
      if has_arg(fn, name):
        res.update({name: value})
    res.update(override)
    return res
