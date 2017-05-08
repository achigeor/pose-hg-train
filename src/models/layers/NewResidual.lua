local conv = nnlib.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Second branch of new bottlneck module
local function secondInnerConvBlock(numIn)
    return nn.Sequential()
        :add(batchnorm(numIn /4))
        :add(relu(true))
        :add(conv(numIn /4, numIn /4,3,3,1,1,1,1):noBias())
end

-- First branch of new bottleneck module
local function firstInnerConvBlock(numIn)
    return nn.Sequential()
        :add(batchnorm(numIn /2))
        :add(relu(true))
        :add(conv(numIn /2, numIn /4,3,3,1,1,1,1):noBias())
end


-- Main Bottleneck - New based on https://arxiv.org/abs/1703.00862
local function bottleneckBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,3,3,1,1,1,1):noBias())
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(firstInnerConvBlock(numOut))
                :add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(secondInnerConvBlock(numOut))
                )
                :add(nn.JoinTable(2,4))
            )
        )
        :add(nn.JoinTable(2,4)) -- concatanate along the channels dimension

end


-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1):noBias())
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(bottleneckBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

