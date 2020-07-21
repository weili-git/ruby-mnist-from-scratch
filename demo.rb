require 'nmatrix/nmatrix'

module NnFunctions
  def sigmoid(x)
    N::ones_like(x) / ((-x).exp + 1.0)
  end

  def softmax(x)
    x2 = x - x.max(1)[0]
    x2.exp / x2.exp.sum(1)[0]
  end

  def argmax(x)  # built-in NMatrix
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

  def numerical_diff(f, x)
    (f.call(x+1e-4) - f.call(x-1e-4))/2e-4
  end

  def numerical_gradient_1d(f, x)  # numerical_gradient(method(:f), x)  # https://www.jianshu.com/p/139037ac0b31
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

  def numerical_gradient_2d(f, x)
    if x.dim == 1
      numerical_gradient_1d(f, x)
    else
      grad = N::zeros_like(x)
      x.each_with_index do |xx, i|
        x[i] = numerical_gradient_1d(f, xx)
      end
    end
  end

  def numerical_gradient(f, x)
    h = 1e-4
    # grad = M::zeros_like(x)
    len = 1
    x.shape.each do |s|
      len = len*s
    end
    grad = NMatrix.new([len], 0)
    #
    x.each_with_index do |xx, i|
      fx1 = f.call(xx+h)
      fx2 = f.call(xx-h)
      grad[i] = (fx1-fx2)/(2*h)  # 如何赋值对应
    end
    return grad.reshape(x.shape)
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

class SimpleNet
  def initialize(in_dim, out_dim)
    @w = NMatrix.random([in_dim, out_dim])
  end
  def predict(x)
    x.dot @w
  end
  def loss(x, t)
    z = predict(x)
    y = softmax(z)
    mean_squared_error(y, t)
  end
end


class TwoLayerNet
  attr_reader :params
  def initialize(in_dim, hid_dim, out_dim, std=0.01)
    @params = Hash.new
    @params['W1'] = NMatrix.random([in_dim, hid_dim]) * std
    @params['b1'] = NMatrix.new([1, hid_dim], 0.0)
    @params['W2'] = NMatrix.random([hid_dim, out_dim]) * std
    @params['b2'] = NMatrix.new([1, out_dim], 0.0)
  end

  def predict(x)
    w1, w2 = @params['W1'], @params['W2']
    b1, b2 = @params['b1'], @params['b2']
    a1 = x.dot(w1) + b1.repeat(x.shape[0], 0)  # batch
    z1 = sigmoid(a1)
    a2 = z1.dot(w2)+b2.repeat(x.shape[0], 0)
    y = softmax(a2)
    return y
  end

  def loss(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = predict(x)
    y = argmax(y)
    t = argmax(t)
    # accuracy = (y==t)
    total = y.shape[0]
    accurate = 0.0
    total.times do |idx|
      if y[idx] == t[idx]
        accurate += 1.0
      end
    end
    return accurate / total.to_f
  end

  def num_grad(x, t)  # error!
    loss_w = lambda {|w| loss(x, t)}
    grads = Hash.new
    grads['W1'] = numerical_gradient(method(:loss_w), @params['W1'])
    grads['b1'] = numerical_gradient(method(:loss_w), @params['b1'])
    grads['W2'] = numerical_gradient(method(:loss_w), @params['W2'])
    grads['b2'] = numerical_gradient(method(:loss_w), @params['b2'])
    return grads
  end
end
# ######################################################
include NnFunctions
# t = N[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = N[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# t = N[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
# y = N[[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
# p "mse:" , mean_squared_error(y, t)
# p "cee:" , cross_entropy_error(y, t)
# gets
# ######################################################
# ######################################################
# def f(x)
#   return x[0]**2 + x[1]**2
# end
# x = N[3.0, 4.0]
# p numerical_gradient_1d(method(:f), x).to_s
# p gradient_descent(method(:f), x, 0.1).to_s
# ######################################################
# ######################################################
# w = SimpleNet.new(2, 3)
# x = N[[0.6, 0.9]]
# p argmax(w.predict(x)).to_s  # argmax
# p w.loss(x, N[[1.0, 0.0, 0.0]])
# ######################################################
# ######################################################
net = TwoLayerNet.new(28*28, 100, 10)  # net
x = NMatrix.random([100, 28*28])  # image
t = NMatrix.random([100, 10])  # label
pp net.predict(x).shape
pp net.loss(x, t)
# pp net.num_grad(x, t).shape
