--------------------------------------------------------------------------------
-- ImproveTripletEmbeddingCriterion by Yemt
--------------------------------------------------------------------------------

local ImproveTripletEmbeddingCriterion, parent = torch.class('nn.ImproveTripletEmbeddingCriterion', 'nn.Criterion')

function ImproveTripletEmbeddingCriterion:__init(Tau1, Tau2, weight)
   parent.__init(self)
   self.alpha = alpha or 0.8
   self.Tau1 = Tau1 or -1
   self.Tau2 = Tau2 or 0.01
   self.weight = weight or 0.002
   self.Li = torch.Tensor()
   self.gradInput = {}
   print(self.Tau1)
end

function ImproveTripletEmbeddingCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   --
   
   -- max{d^n(I0, I+, I-), T1} + max{d^p(I0, I+), T2}

   local loss = torch.max(torch.cat(torch.Tensor(N):type(torch.type(a)):fill(self.Tau1), 
         (a - p):norm(2,2):pow(2) - (a - n):norm(2,2):pow(2), 2), 2) + 
            self.weight * torch.max(torch.cat(torch.Tensor(N):type(torch.type(a)):fill(self.Tau2), (a-p):norm(2,2):pow(2), 2), 2)
                - torch.Tensor(N):type(torch.type(a)):fill(self.Tau1+self.weight*self.Tau2)
   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), loss,2), 2)
   -- max{d^p(I0, I+), T2} d^p(I0, I+)值越大说明两者距离越远,max{d^p(I0, I+), T2}*beta的意思是两者的距离至少是0.01*beita
   -- |T1| 大于 |T2| 的意思是类间
   -- 最外层的max意义是，如果max{d^n(I0, I+, I-), T1} + max{d^p(I0, I+), T2} > 0, 表示有损失,因为类内的距离大于类间的距离了
   self.output = self.Li:sum() / N
   
   return self.output
end

function ImproveTripletEmbeddingCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   local dn = ((a - p):norm(2,2):pow(2) - (a - n):norm(2,2):pow(2))
   local dp = ((a - p):norm(2,2):pow(2))
   local dn_case = torch.expand(dn:gt(self.Tau1), dn:size(1), a:size(2)):type(a:type())
   local dp_case = torch.expand(dp:gt(self.Tau2), dp:size(1), a:size(2)):type(a:type())
   self.gradInput[1] = torch.cmul(dn_case, (n-p):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N))
                            + torch.cmul(dp_case, (a-p):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)):mul(self.weight)
    
   self.gradInput[2] = torch.cmul(dn_case, (p-a):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N))
                            + torch.cmul(dp_case, (p-a):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)):mul(self.weight)
   
   self.gradInput[3] = torch.cmul(dn_case, (a-n):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N))
   
   return self.gradInput
end

