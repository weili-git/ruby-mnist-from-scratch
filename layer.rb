# Encoding: utf-8

require_relative 'nn_functions'
include NnFunctions


class MulLayer
    def initialize
        @x, @y = nil, nil
    end

    def forward(x, y)
        @x, @y = x, y
        out = x*y
        return out
    end

    def backward(dout)
        dx = dout*@y  # reverse x and y
        dy = dout*@x
        return dx, dy
    end
end

class AddLayer
    def initialize
        # do nothing
    end

    def forward(x, y)
        return x + y
    end

    def backward(dout)
        dx = dout*1
        dy = dout*1
        return dx, dy
    end
end

class Relu
    def initialize
        @mask = nil
    end

    def forward(x)
        @mask = (x<=0)
        out = x.dup
        @mask.each_with_index {|mask, idx| out[idx] = 0 if mask}
        # out[@mask] = 0  # syntactic sugar
        return out
    end

    def backward(dout)
        @mask.each_with_index {|mask, idx| dout[idx] = 0 if mask}
        dx = dout
        return dx
    end
end

class Sigmoid
    def initialize
        @out = nil
    end

    def forward(x)
        out = sigmoid(x)
        @out = out
        return out
    end

    def backward(dout)
        dx = dout * (-@out + 1.0) * @out  # vector * vector => u*v.t
        return dx
    end
end

class Affine
    def initialize(W, b)
        @W = W
        @b = b

        @x = nil
        ## @original_x_shape = nil
        @dW = nil
        @db = nil
    end

    def forward(x)
        ## @original_x_shape = x.shape
        ## x = x.reshape([ x.shape[0] , x.shape.inject(1, :*) / x.shape[0] ])  # why?
        # x = x.reshape(x.shape[0], -1) # python
        @x = x

        out = @x.dot(@W) + @b.repeat(@x.shape[0], 0)
        return out
    end

    def backward(dout)
        dx = dout.dot(@W.transpose)
        @dW = @x.transpose.dot(dout)
        @db = dout.sum  # (axis=0)

        ## dx = dx.reshape(*@original_x_shape)  # *可变参数 python
        return dx
    end
end

class SoftmaxWithLoss  # ???
    def initialize
        @loss = nil
        @y = nil # softmax的输出
        @t = nil # 监督数据
    end

    def forward(x, t)
        @t = t
        @y = softmax(x)
        @loss = cross_entropy_error(@y, @t)
        
        return @loss
    end

    def backward(dout=1)
        batch_size = @t.shape[0]
        if @t.size == @y.size # 监督数据是one-hot-vector的情况
            dx = (@y - @t) / batch_size
        else
            # ??? numpy syntax
            # dx = @y.dup  # batch_size
            # dx[np.arange(batch_size), @t] -= 1
            # dx = dx / batch_size
            # python
        end
        return dx
    end
end
###########################################################
=begin
class Dropout
    """
    http://arxiv.org/abs/1207.0580
    """
    def initialize(dropout_ratio=0.5)
        @dropout_ratio = dropout_ratio
        @mask = nil
    end

    def forward(x, train_flg=True)
        if train_flg
            @mask = np.random.rand(*x.shape) > @dropout_ratio
            return x * @mask
        else
            return x * (1.0 - @dropout_ratio)
        end
    end

    def backward(dout):
        return dout * @mask
    end
end


class BatchNormalization
    """
    http://arxiv.org/abs/1502.03167
    """
    def initialize(gamma, beta, momentum=0.9, running_mean=nil, running_var=nil)
        @gamma = gamma
        @beta = beta
        @momentum = momentum
        @input_shape = nil # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        @running_mean = running_mean
        @running_var = running_var  
        
        # backward时使用的中间数据
        @batch_size = nil
        @xc = nil
        @std = nil
        @dgamma = nil
        @dbeta = nil
    end

    def forward(x, train_flg=True)
        @input_shape = x.shape
        if x.ndim != 2
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        end

        out = @__forward(x, train_flg)
        
        return out.reshape(*@input_shape)
    end
            
    def __forward(x, train_flg)
        if @running_mean is nil
            N, D = x.shape
            @running_mean = np.zeros(D)
            @running_var = np.zeros(D)
        end
                        
        if train_flg
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            @batch_size = x.shape[0]
            @xc = xc
            @xn = xn
            @std = std
            @running_mean = @momentum * @running_mean + (1-@momentum) * mu
            @running_var = @momentum * @running_var + (1-@momentum) * var            
        else
            xc = x - @running_mean
            xn = xc / ((np.sqrt(@running_var + 10e-7)))
        end
            
        out = @gamma * xn + @beta 
        return out
    end

    def backward(dout)
        if dout.ndim != 2
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        end

        dx = @__backward(dout)

        dx = dx.reshape(*@input_shape)
        return dx
    end

    def __backward(dout)
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(@xn * dout, axis=0)
        dxn = @gamma * dout
        dxc = dxn / @std
        dstd = -np.sum((dxn * @xc) / (@std * @std), axis=0)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / @batch_size) * @xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / @batch_size
        
        @dgamma = dgamma
        @dbeta = dbeta
        
        return dx
    end
end


class Convolution
    def initialize(W, b, stride=1, pad=0):
        @W = W
        @b = b
        @stride = stride
        @pad = pad
        
        # 中间数据（backward时使用）
        @x = nil   
        @col = nil
        @col_W = nil
        
        # 权重和偏置参数的梯度
        @dW = nil
        @db = nil
    end

    def forward(x)
        FN, C, FH, FW = @W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*@pad - FH) / @stride)
        out_w = 1 + int((W + 2*@pad - FW) / @stride)

        col = im2col(x, FH, FW, @stride, @pad)
        col_W = @W.reshape(FN, -1).T

        out = np.dot(col, col_W) + @b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        @x = x
        @col = col
        @col_W = col_W

        return out
    end

    def backward(dout)
        FN, C, FH, FW = @W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        @db = np.sum(dout, axis=0)
        @dW = np.dot(@col.T, dout)
        @dW = @dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, @col_W.T)
        dx = col2im(dcol, @x.shape, FH, FW, @stride, @pad)

        return dx
    end
end


class Pooling
    def initialize(pool_h, pool_w, stride=1, pad=0):
        @pool_h = pool_h
        @pool_w = pool_w
        @stride = stride
        @pad = pad
        
        @x = nil
        @arg_max = nil
    end

    def forward(x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - @pool_h) / @stride)
        out_w = int(1 + (W - @pool_w) / @stride)

        col = im2col(x, @pool_h, @pool_w, @stride, @pad)
        col = col.reshape(-1, @pool_h*@pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        @x = x
        @arg_max = arg_max

        return out
    end

    def backward(dout)
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = @pool_h * @pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(@arg_max.size), @arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, @x.shape, @pool_h, @pool_w, @stride, @pad)
        
        return dx
    end
end
=end
