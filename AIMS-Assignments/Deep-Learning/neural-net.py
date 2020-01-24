import torch
# TO LOAD MNIST DATASET
from dlc_practical_prologue import *


data = load_data(one_hot_labels=True,normalize=True)


# HELPER FUNCTIONS
def sigma(inputs):
    return torch.tanh(inputs)

def dsigma(inputs):
    return 1-torch.pow(sigma(inputs),2)

def loss(v,t):
    return torch.pow((v-t),2).sum()

def dloss(v,t):
    return 2*(v-t)
  
  
  
def get_params(train_images,hidden_dim=50,out_dim=10,std = 1e-6):
    w1 = torch.empty((hidden_dim,train_images.size(1))).normal_(0,std)
    b1 = torch.empty((hidden_dim,1)).normal_(0,std)

    w2 = torch.empty((out_dim,hidden_dim)).normal_(0,std)
    b2 = torch.empty((out_dim,1)).normal_(0,std)

    dl_w1 = torch.empty_like(w1)
    dl_b1 = torch.empty_like(b1)

    dl_w2 = torch.empty_like(w2)
    dl_b2 = torch.empty_like(b2)

    return w1,b1,w2,b2,dl_w1,dl_b1,dl_w2,dl_b2
  
  
def forward_pass(w1,b1,w2,b2,x):
    s1 = torch.add(torch.mm(w1,x),b1)
    x1 = sigma(s1)

    s2 = torch.add(torch.mm(w2,x1),b2)
    x2 = sigma(s2)

    return x,s1,x1,s2,x2

    
    
def backward_pass(w1,b1,w2,b2,t,x,s1,x1,s2,x2,dl_dw1,dl_db1,dl_dw2,dl_db2):
    dl_dx2 = dloss(x2,t)
    dl_ds2 = dl_dx2*dsigma(s2)

    dl_dw2.add_(torch.mm(dl_ds2,x1.view(1,-1)))
    dl_db2.add_(dl_ds2)

    dl_dx1 = torch.mm(w2.T,dl_ds2)
    dl_ds1 = dl_dx1*dsigma(s1)

    dl_dw1.add_(torch.mm(dl_ds1,x.view(1,-1)))
    dl_db1.add_(dl_ds1)
    
    
  
train_images,train_labels,test_images,test_labels = data

w1,b1,w2,b2,dl_dw1,dl_db1,dl_dw2,dl_db2 = get_params(train_images,hidden_dim=50,out_dim=10)

train_labels,test_labels = train_labels * 0.9,test_labels * 0.9
train_images,train_labels = train_images * 0.9,train_labels * 0.9

train_size = len(train_images)
test_size = len(test_images)

lr = 1e-1 / train_size
epoch = 1000

for i in range(epoch):
    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    correct_count = 0
    test_correct_count = 0
    running_loss = 0

    for image,labels in zip(train_images,train_labels):
        x,s1,x1,s2,x2 = forward_pass(w1,b1,w2,b2,image.view(-1,1))
        running_loss += loss(x2,labels.view(-1,1))
        backward_pass(w1,b1,w2,b2,labels.view(-1,1),image.view(-1,1),s1,x1,s2,x2,dl_dw1,dl_db1,dl_dw2,dl_db2)

    w1 = w1 -  lr * dl_dw1
    b1 = b1 -  lr * dl_db1
    w2 = w2 -  lr * dl_dw2
    b2 = b2 -  lr * dl_db2

    for img,lbl in zip(train_images,train_labels):
        _,_,_,_,x2 = forward_pass(w1,b1,w2,b2,img.view(-1,1))
        if x2.max(0)[1].item()==lbl.max(0)[1].item():
            correct_count +=1

    for t_img,t_lbl in zip(test_images,test_labels):
        _,_,_,_,t_x2 = forward_pass(w1,b1,w2,b2,t_img.view(-1,1))
        if t_x2.max(0)[1].item()==t_lbl.view(-1,1).max(0)[1].item():
            test_correct_count +=1
    
    if i%100 == 0:
        print('Epoch: {}, Loss: {:.02f} Train Accuracy(Error Rate): {:.02f}({:.02f}) , Test Accuracy(Error Rate): {:.02f}({:.02f})'.format(i,running_loss,\
            correct_count/train_size,1-(correct_count/train_size),test_correct_count/test_size,1-(test_correct_count/test_size)))