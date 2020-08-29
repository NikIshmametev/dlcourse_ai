import numpy as np
import linear_classifer

def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        delta_arr = np.zeros_like(x)
        delta_arr[ix] = delta

        # TODO compute value of numeric gradient of f to idx
        fx_pd, _ = f(x+delta_arr)
        fx_md, _ = f(x-delta_arr)
        numeric_grad = (fx_pd - fx_md)/(2*delta)

        numeric_grad_at_ix = np.array([numeric_grad]) if isinstance(numeric_grad, float) else numeric_grad[ix]
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


if __name__=='__main__':
  num_classes = 4
  batch_size = 3
  predictions = np.random.randint(-1, 3, size=(batch_size, num_classes)).astype(np.float)
  target_index = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int)
  check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, target_index), predictions)

        
