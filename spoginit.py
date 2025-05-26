
import torch
import math
import numpy as np
from models.GCN_layer import MP
import torch.nn.functional as F
import copy
import time

def find_parameters(model,to_copy=False):
    """Find all weight parameters in a model.
    
    Args:
        model: The neural network model
        to_copy: Whether to create a deep copy of the model first
        
    Returns:
        List of weight parameters
    """
    z = []
    if to_copy:
        md2=copy.deepcopy(model)
    else:
        md2=model
    for a,grad in md2.named_parameters():
        if "weight" in a:
            z.append(grad)
    return z

def find_parameters_gat(model,to_copy=False):
    z = []
    if to_copy:
        md2=copy.deepcopy(model)
    else:
        md2=model
    for a,grad in md2.named_parameters():
        if len(grad.size())>1:
            z.append(grad)
    return z
    

def print_weightnorm(model):
    """Calculate and print the average norm of weight parameters.
    
    Args:
        model: The neural network model
        
    Returns:
        Average norm of weight parameters as numpy array
    """
    z = []
    times = 0
    #coe = 1
    for name, i in model.named_parameters():
        if "weight" in name:
            #temp = torch.linalg.norm(i.lin.weight.detach(),ord=2)**2
            temp = torch.norm(i.data.detach())
            z.append(temp.detach())
            #coe = coe*(sig.max())*s
        times = times+1
    z = torch.tensor(z)
    return torch.mean(z).cpu().detach().numpy()



class SpogInit:
    def __init__(self, model, edge, mask, num_nodes, x_dim, y_dim,device,metric_way):
        """Initialize the SPOG initialization class.
        
        Args:
            model: The GNN model to initialize
            edge: edge_index for the graph
            mask: training mask for training set
            num_nodes: Number of nodes in the graph
            x_dim: Dimension of input features
            y_dim: Dimension of output (number of classes)
            device: Computation device (cpu/gpu)
            metric_way: Method for calculating metrics ('divide_old', 'divide_stable'.)
        """
        super(SpogInit,self).__init__()
        self.model = model
        self.edge = edge
        self.num_nodes = num_nodes
        self.mask = mask
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device
        self.messagepa = MP(cached = True).to(device)
        ## 这个参数定义metric 如何计算： 除还是mse？
        self.metric_way=metric_way
        
    def new_smoothness_gpu(self,x):
        ## 计算dirichelet energy
        smooth = torch.zeros(1)
        smooth = smooth.to(self.device)
        for k in range(x.shape[1]):
            u = x[:,k].to(self.device)
            time3 = time.time()
            if torch.matmul(u.T,u) == 0:
                smooth += 0
            else:
                smooth += torch.matmul(u.T,u.reshape(-1,1)-self.messagepa(u.reshape(-1,1),self.edge))/torch.matmul(u.T,u)
            time4 = time.time()
        final =0.5 * smooth / x.shape[1] 
        return final

    def new_smoothness_gpu_fast(self,x):
        """Faster version of Dirichlet energy calculation using matrix operations.
        
        Args:
            x: Node features
            
        Returns:
            Average smoothness across feature dimensions
        """
        
        time3 = time.time()
        smooth= torch.matmul(x.T,x-self.messagepa(x,self.edge))/torch.matmul(x.T,x)
        smooth = torch.nan_to_num(smooth, nan=0.0, posinf=1e6, neginf=-1e6)
        smooth = torch.trace(smooth)                                       
        time4 = time.time()
        #print(f"run {time4-time3}")
        #break
        final =0.5 * smooth / x.shape[1] 
        return final

    
    def forward_metric_divide_old(self,forward_norm):
        """Forward metric: ratio of first to last layer norms."""
        return (torch.norm(forward_norm[0])/torch.norm(forward_norm[-1])-1)**2
    
    
    def forward_metric_divide_stable(self,forward_norm):
        """Forward metric: ratio of maximum and minimal layer norms."""
        a = torch.tensor([torch.norm(i)/math.sqrt(i.shape[1]) for i in forward_norm]) 
        return (a.max()/a.min()-1)**2
    def forward_metric_divide_stable_avoidinf(self,forward_norm):
        """Stable backward metric avoiding division by zero."""
        a = torch.tensor([torch.norm(i)/math.sqrt(i.shape[1]) for i in forward_norm]) 
        print("forward: ",a)
        return (a.max()/(a[torch.where(a>1e-8)].min())-1)**2
    

    def bakcward_metric_divide_old(self,gradients):
        """Backward metric: ratio of second to second-to-last layer gradients."""
        return (torch.norm(gradients[1])/torch.norm(gradients[-2])-1)**2
    
    def bakcward_metric_divide_stable(self,gradients):
        """Stable backward metric using max/min gradient norms."""
        a = torch.tensor([torch.norm(i) for i in gradients]) 
        return (a.max()/a.min()-1)**2

    def bakcward_metric_divide_stable_avoidinf(self,gradients):
        """Stable backward metric avoiding division by zero."""
        a = torch.tensor([torch.norm(i) for i in gradients]) 
        print("backward: ",a)
        return (a.max()/(a[torch.where(a>1e-8)].min())-1)**2
    
    
    
    def output_diversity(self,model_print):
        return self.new_smoothness_gpu(model_print)

    def output_diversity_fast(self,model_print):
        """Faster output diversity calculation."""
        return self.new_smoothness_gpu_fast(model_print)
    
    
    def generate_metrics(self,x,y):
        """Generate forward, backward and diversity metrics.
        
        Args:
            x: Input features
            y: Target labels
            
        Returns:
            Tuple of (forward_metric, backward_metric, diversity_metric)
        """
        params = find_parameters(self.model.convs)
        forward_norm = self.model.print_all_x(x, self.edge)
        out = self.model(x,self.edge)
        if len(y.size())==1:
            loss = F.nll_loss(out[self.mask], y[self.mask]).to(self.device)
        else:
            loss = F.nll_loss(out[self.mask], y.squeeze(1)[self.mask]).to(self.device)
        grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
        #time2 = time.time()
        if self.metric_way == "divide_old":
            forward = self.forward_metric_divide_old(forward_norm)
            backward = self.bakcward_metric_divide_old(grad)
            od = self.output_diversity_fast(forward_norm[-1])#/self.output_diversity_fast(x)
        elif self.metric_way == "divide_stable":
            #forward = self.forward_metric_divide(forward_norm)
            forward = self.forward_metric_divide_stable(forward_norm)
            backward = self.bakcward_metric_divide_stable(grad)
            od = self.output_diversity(forward_norm[-1])

        elif self.metric_way == "divide_stable_avoidinf":
            
            forward = self.forward_metric_divide_stable_avoidinf(forward_norm)
            backward = self.bakcward_metric_divide_stable_avoidinf(grad)
            od = self.output_diversity(forward_norm[-1])

            
        return forward.cpu().detach().float().item(), backward.cpu().detach().float().item(), od.cpu().detach().float().item()


    
    def generate_random_data(self):
        """Generate random input features and labels."""
        x = torch.normal(mean=torch.zeros(self.num_nodes,self.x_dim), std=1).to(self.device)
        y = torch.randint(0,self.y_dim,(self.num_nodes,)).to(self.device)
        return x,y
        
        
        
    def zeroincrease_initialization(self,x,y,lr=0.05,steps=100,max_pati=10,w1=1,w2=1,w3=1,generate_data=False):
        """Zero-order optimization with search directions across all layers .
        
        Args:
            x: Input features
            y: Target labels
            lr: Learning rate
            steps: Maximum optimization steps
            max_pati: Patience for early stopping
            w1, w2, w3: Metric component weights
            generate_data: Whether to generate random data each iteration
        """
        T1 = time.time()
        params = find_parameters(self.model.convs)
        init_weight = torch.ones(len(params))
        updated_init_weight = torch.ones(len(params))
        memory = torch.zeros(len(params))
        metric_memory = []
        scale_meory = []
        best_metric =1000
        patience = 0

        if generate_data == True:
            x,y = self.generate_random_data()
        
        data_diver = self.new_smoothness_gpu(x)
        self.model.eval()
        
        for i in range(steps):
            if generate_data == True:
                x,y = self.generate_random_data()

            delta = 1e-5
            f,b, diversity = self.generate_metrics(x,y)
            old_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
            if old_metric<best_metric:
                best_metric = old_metric
                patience = 0
            else:
                patience = patience+1
            
            if patience>=max_pati:
                break
            metric_memory.append(old_metric)
            scale_meory.append(init_weight.clone())
            #print(torch.norm(params[0]))
            #for random in range(10):
            for random in range(3):
                z = torch.randn(len(params))
                init_grad = torch.zeros(len(params))
                for layer in range(len(params)):
                    
                    params[layer].data.mul_((init_weight[layer] + delta*z[layer])/init_weight[layer])
                f,b, diversity = self.generate_metrics(x,y)
                new_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
                
                
                init_grad = init_grad + (new_metric.item()-old_metric.item())/(delta * z*3)
                for layer in range(len(params)):
                    params[layer].data.mul_(init_weight[layer]/(init_weight[layer] + delta*z[layer]))
            for layer in range(len(params)):
                updated_init_weight[layer] = init_weight[layer] - lr*torch.sign(init_grad[layer])
                params[layer].data.mul_(updated_init_weight[layer]/init_weight[layer])
            if i%20==0:
                lr = 0.7*lr
            init_weight = updated_init_weight.clone()

            print(str(i)+"-the iteration total metric "+ str(old_metric.float())+ f" sigmas: {init_weight}")
    
        for layer in range(len(params)):
            params[layer].data.mul_(scale_meory[torch.tensor(metric_memory).argmin()][layer]/init_weight[layer])
        T2 = time.time()
        print("total times", T2-T1)



    def zerosingle_initialization(self,x,y,lr=0.05,steps=100,max_pati=10,w1=1,w2=1,w3=1,decay=0.7,generate_data=False):
        """
        
        Similar to zeroincrease_initialization but uses same direction for all parameters.
        """
        T1 = time.time()
        params = find_parameters(self.model.convs)
        init_weight = torch.ones(len(params))
        updated_init_weight = torch.ones(len(params))
        memory = torch.zeros(len(params))
        metric_memory = []
        scale_meory = []
        if len(params)<=3:
            print("we need model with more that 4 layers")
            raise RuntimeError('Error')
        best_metric =1000
        patience = 0

        if generate_data == True:
            x,y = self.generate_random_data()

        data_diver = self.new_smoothness_gpu(x)
        self.model.eval()
        
        for i in range(steps):
            if generate_data == True:
                x,y = self.generate_random_data()
            delta = 1e-5
            f,b, diversity = self.generate_metrics(x,y)
            old_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
            metric_memory.append(old_metric)
            scale_meory.append(init_weight.clone())
            #print(torch.norm(params[0]))
            #for random in range(10):
            for random in range(2):
                a = torch.randn(1)
                z = torch.tensor([a]*len(params))
                init_grad = torch.zeros(len(params))
                for layer in range(len(params)):
                    
                    params[layer].data.mul_((init_weight[layer] + delta*z[layer])/init_weight[layer])
                
                f,b, diversity = self.generate_metrics(x,y)
                new_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
                
                
                init_grad = init_grad + (new_metric.item()-old_metric.item())/delta * z
                for layer in range(len(params)):
                    params[layer].data.mul_(init_weight[layer]/(init_weight[layer] + delta*z[layer]))
            for layer in range(len(params)):
                updated_init_weight[layer] = init_weight[layer] - lr*torch.sign(init_grad[layer])
                params[layer].data.mul_(updated_init_weight[layer]/init_weight[layer])
            if (i+1)%20==0:
                lr = decay*lr
            init_weight = updated_init_weight.clone()

            print(str(i)+"-the iteration total metric "+ str(old_metric.float())+ f" sigmas: {init_weight}")
    
        for layer in range(len(params)):
            params[layer].data.mul_(scale_meory[torch.tensor(metric_memory).argmin()][layer]/init_weight[layer])
        T2 = time.time()
        print("total times", T2-T1)

    def gate_zeroincrease_initialization(self,x,y,lr=0.05,steps=100,max_pati=10,w1=1,w2=1,w3=1,generate_data=False):
        """Zero-order optimization for gatResGCN models, by optimizing the alpha across all layers."""
        T1 = time.time()
        params = find_parameters_gat(self.model.convs)
        init_weight = torch.ones(len(params))
        updated_init_weight = torch.ones(len(params))
        memory = torch.zeros(len(params))
        metric_memory = []
        scale_meory = []
        #if len(params)<=3:
        #    print("we need model with more that 4 layers")
        #    raise RuntimeError('Error')
        best_metric =1000
        patience = 0

        if generate_data == True:
            x,y = self.generate_random_data()
        
        data_diver = self.new_smoothness_gpu(x)
        self.model.eval()
        
        for i in range(steps):
            if generate_data == True:
                x,y = self.generate_random_data()

            delta = 1e-5
            f,b, diversity = self.generate_metrics(x,y)
            old_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
            if old_metric<best_metric:
                best_metric = old_metric
                patience = 0
            else:
                patience = patience+1
            
            if patience>=max_pati:
                break
            metric_memory.append(old_metric)
            scale_meory.append(init_weight.clone())
            for random in range(3):
                z = torch.randn(len(params))
                init_grad = torch.zeros(len(params))
                for layer in range(len(params)):
                    
                    params[layer].data.mul_((init_weight[layer] + delta*z[layer])/init_weight[layer])
                f,b, diversity = self.generate_metrics(x,y)
                new_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
                
                
                init_grad = init_grad + (new_metric.item()-old_metric.item())/(delta * z*3)
                for layer in range(len(params)):
                    params[layer].data.mul_(init_weight[layer]/(init_weight[layer] + delta*z[layer]))
            for layer in range(len(params)):
                updated_init_weight[layer] = init_weight[layer] - lr*torch.sign(init_grad[layer])
                params[layer].data.mul_(updated_init_weight[layer]/init_weight[layer])
            if i%20==0:
                lr = 0.7*lr
            init_weight = updated_init_weight.clone()

            print(str(i)+"-the iteration total metric "+ str(old_metric.float())+ f" sigmas: {init_weight}")
    
        for layer in range(len(params)):
            params[layer].data.mul_(scale_meory[torch.tensor(metric_memory).argmin()][layer]/init_weight[layer])
        T2 = time.time()
        print("total times", T2-T1)

    def zerosingle_initialization_gate(self,x,y,lr=0.05,steps=100,max_pati=10,w1=1,w2=1,w3=1,decay=0.7):
        """
        
        Similar to gate_zeroincrease_initialization but uses same direction for all parameters.
        """
        T1 = time.time()
        params = find_parameters(self.model.convs)
        init_weight = torch.ones(len(params))
        updated_init_weight = torch.ones(len(params))
        memory = torch.zeros(len(params))
        metric_memory = []
        scale_meory = []
        if len(params)<=3:
            print("we need model with more that 4 layers")
            raise RuntimeError('Error')
        best_metric =1000
        patience = 0
        data_diver = self.new_smoothness_gpu(x)
        self.model.eval()
        
        for i in range(steps):
            delta = 1e-5
            f,b, diversity = self.generate_metrics(x,y)
            old_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
            metric_memory.append(old_metric)
            scale_meory.append(init_weight.clone())
            #print(torch.norm(params[0]))
            #for random in range(10):
            for random in range(2):
                a = torch.randn(1)
                z = torch.tensor([a]*len(params))
                init_grad = torch.zeros(len(params))
                for layer in range(len(params)):
                    
                    self.model.alpha[layer].data.mul_((init_weight[layer] + delta*z[layer])/init_weight[layer])
                
                f,b, diversity = self.generate_metrics(x,y)
                new_metric = w1*f + w2*b + w3*(data_diver-diversity)**2
                
                
                init_grad = init_grad + (new_metric.item()-old_metric.item())/delta * z
                for layer in range(len(params)):
                    self.model.alpha[layer].data.mul_(init_weight[layer]/(init_weight[layer] + delta*z[layer]))
            for layer in range(len(params)):
                updated_init_weight[layer] = init_weight[layer] - lr*torch.sign(init_grad[layer])
                self.model.alpha[layer].data.mul_(updated_init_weight[layer]/init_weight[layer])
            if i+1%20==0:
                lr = decay*lr
            init_weight = updated_init_weight.clone()

            print(str(i)+"-the iteration total metric "+ str(old_metric.float())+ f" sigmas: {init_weight}")
    
        for layer in range(len(params)):
            self.model.alpha[layer].data.mul_(scale_meory[torch.tensor(metric_memory).argmin()][layer]/init_weight[layer])
        T2 = time.time()
        print("total times", T2-T1)


    def secondorder_initialization(self,x,y,lr=0.05,steps=100,max_pati=10,w1=1,w2=1,w3=1,decay=0.7,generate_data=False):
        """Second-order initialization searching method."""
        params = find_parameters(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.1, betas=(0.9,0.9995),weight_decay = 0)
        best_params = params
        origin_norm = print_weightnorm(self.model)
        if len(params)<=3:
            print("we need model with more that 4 layers")
            raise RuntimeError('Error')
        best_metric =1000
        patience = 0
        origin_diri = self.new_smoothness_gpu(x)
        for i in range(steps):
            x = torch.normal(mean=torch.zeros(self.num_nodes,self.x_dim), std=1).to(self.device)
            y = torch.randint(0,self.y_dim,(self.num_nodes,)).to(self.device)
            self.model.eval()
            forward_norm = self.model.print_all_x(x, self.edge)
            out = self.model(x,self.edge)
            loss = F.nll_loss(out[self.mask], y[self.mask]).to(self.device)
            grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
            metric = (torch.norm(forward_norm[0])/torch.norm(forward_norm[-1])-1)**2 + 10*(torch.norm([g for g in grad if len(g.size())>=2][1])/torch.norm([g for g in grad if len(g.size())>=2][-2])-1)**2
            metric = metric - (self.new_smoothness_gpu(self.model.print_x(x,self.edge))/origin_diri)
            optimizer.zero_grad()
            if metric<best_metric:
                best_metric = metric
                best_params = params
                patience = 0
            else:
                patience = patience+1
            if patience==max_pati:
                break
            grad = torch.autograd.grad(metric, params,allow_unused=True)
            for j, (p, g_all) in enumerate(zip(params, grad)):
                #print(g_all)
                norm = p.data.norm().item()
                if g_all is not None:
                    g = torch.sign((p.data * g_all).sum() / norm)
                    new_norm = norm - lr* g.item()
                    if new_norm / norm<1e-6:
                        new_norm = 1e-6*norm
                    p.data.mul_(new_norm / norm)
            print(str(i+1)+"-th iterations： "+ " metric:, " + str(metric.item()))
            optimizer.zero_grad()
        
        index = 0
        for a,pa in self.model.named_parameters():
            if "weight" in a:
                #print(grad.grad)
                pa.data = best_params[index]
                index = index+1

