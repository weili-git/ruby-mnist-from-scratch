require 'NMatrix'

def step_function(x)  # NMatrix
    (x>0).cast(:list, :int32)
end

def sigmoid(x)
    ((-x).exp + 1)**(-1)
end

def relu(x)
    x.map {|e| e = e>0 ? e : 0}
end

def identity_function(x)
    x
end

def softmax(x)
    c = x.max
    exp_a = (x - c).exp
    exp_sum = exp_a.sum
    exp_a/exp_sum[0]
end

m = N[[-1, 1], [2, 3]]
pp m.dtype
pp step_function(m)
pp sigmoid(m)
pp relu(m)
n = N[0.3, 2.9, 4.0]
pp softmax(n)

