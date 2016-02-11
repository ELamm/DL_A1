----------------------------------------------------------------------
-- This script preprocesses the (MNIST) Handwritten Digit 
-- training data. It checks the size of each class, augments the data
-- to create balanced classes, and divides the training data into 10 folds
-- for cross-validation.
--
-- Elizabeth Lamm
----------------------------------------------------------------------
require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
----------------------------------------------------------------------
data_path = 'mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')
trsize = 60000
tesize = 10000

----------------------------------------------------------------------

loaded = torch.load(train_file, 'ascii')
trainData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
-- check class distributions in training set
x = trainData.data
y = trainData.labels

classes = {}
for i=1, 10 do 
	classes[i] = 0
end


for i=1, trsize do
	classes[y[i]] = classes[y[i]]+1
end
print(classes)

for k=1, 10 do
	print(classes[k]/trsize)
end

-------------------------------------------------------------------
-- observe that the class sizes range from 5421-6742
-------------------------------------------------------------------
augmentedTrsize = trsize

-------------------------------------------------------------------
-- create and save ten cross-validation folds
-------------------------------------------------------------------

--[[
foldSize = augmentedTrsize/10
shuffle = torch.randperm(augmentedTrsize)

print(x:size())
foldStart = 1
foldEnd = foldStart + foldSize
foldiX = torch.Tensor(foldSize, x:size(2), x:size(3), x:size(4))
foldiY = torch.Tensor(foldSize)
for j=0, 9 do
	for k=1, foldSize do
		foldiX[k] = x[shuffle[j*foldSize+k]]
--		foldiY[k] = y[shuffle[j*foldSize+k]]
--[[	end
	foldj = {
		data = foldiX,
		labels = foldiY
	}
	file = torch.DiskFile('fold' .. j+1 .. 'test.t7', 'w')
	file:writeObject(foldj)
	file:close()
end

trainjX = torch.Tensor(foldSize*9, x:size(2), x:size(3), x:size(4))
trainjY = torch.Tensor(foldSize*9)
for j=1, 10 do
	curIndex = 1
	if j>=2 then
		for k = 1, j-1 do
			fileIn = torch.DiskFile('fold' .. k .. 'test.t7', 'r')
			foldk = fileIn:readObject()
			for l=1, foldSize do
				trainjX[curIndex]=foldk.data[l]
				trainjY[curIndex]=foldk.labels[l]
				curIndex = curIndex+1
			end
			fileIn:close()
		end
	end	
	if j<=9 then
		for k = j+1, 10 do
			fileIn = torch.DiskFile('fold' .. k .. 'test.t7', 'r')
			foldk = fileIn:readObject()
			for l=1, foldSize do
				trainjX[curIndex]=foldk.data[l]
				trainjY[curIndex]=foldk.labels[l]
				curIndex = curIndex + 1
			end
			fileIn:close()
		end
	end
	foldj = {
		data = trainjX,
		labels = trainjY
	}
	fileOut = torch.DiskFile('fold' .. j .. 'train.t7', 'w')
	fileOut:writeObject(foldj)
	fileOut:close()
end
]]

