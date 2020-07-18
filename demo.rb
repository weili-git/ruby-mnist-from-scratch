require 'nmatrix/nmatrix'

module NnFunctions
  def sigmoid(x)
    N::ones_like(x) / ((-x).exp + 1.0)
  end

  def softmax(x)
    x2 = x - x.max(1)[0]
    x2.exp / x2.exp.sum(1)[0]
  end

  def argmax(x)
    list = x.shape[0] == 1 ? [x.to_a] : x.to_a
    res = list.map do |row|
      max = -Float::MAX
      idx = nil
      row.each_with_index do |n, i|
        if n > max
          idx = i
          max = n
        end
      end
      idx
    end
    N[*res]
  end
  ##############################################
  def step_function(x)  # NMatrix
    (x>0).cast(:list, :int32)
  end

  def sigmoid2(x)
      ((-x).exp + 1)**(-1)
  end

  def relu(x)
      x.map {|e| e = e > 0 ? e : 0}
  end

  def identity_function(x)
      x
  end

  def softmax2(x)
      c = x.max
      exp_a = (x - c).exp
      exp_sum = exp_a.sum
      exp_a/exp_sum[0]
  end

  def mean_squared_error(x, y)
    if x.dim==1
      x = x.reshape([1] + x.shape)
      y = y.reshape([1] + y.shape)
    end
    batch_size = x.shape[0]
    ((y-x)**2).sum.sum(1)[0, 0] * 0.5 / batch_size
  end
  
  def cross_entropy_error(x, y)
    if x.dim==1
      x = x.reshape([1] + x.shape)
      y = y.reshape([1] + y.shape)
    end
    batch_size = x.shape[0]
    -(y*(x+1e-7).log).sum.sum(1)[0, 0] / batch_size
  end

  def numerical_diff(f, x)  # numerical_diff(method(:f), x)  # https://www.jianshu.com/p/139037ac0b31
    h = 1e-4
    if x.dtype == :int32
      x = x.cast(:list, :float64)
    end
    grad = N::zeros_like(x)
    x.size.times do |idx|
      tmp_val = x[idx]
      x[idx] = tmp_val + h
      fxh1 = f.call(x)
      x[idx] = tmp_val - h
      fxh2 = f.call(x)
      grad[idx] = (fxh1-fxh2)/(2*h)
      x[idx] = tmp_val
    end
    return grad
  end

  def gradient_descent(f, init_x, lr=1e-2, step_num=100)
    if init_x.dtype == :int32
      init_x = init_x.cast(:list, :float64)
    end
    x = init_x
    step_num.times do |i|
      grad = numerical_diff(f, x)
      x -= grad*lr
    end
    return x
  end
end
