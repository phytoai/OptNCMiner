import os, logging
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class OptNPMiner(nn.Module):
    def __init__(self, Xshape, headshape, bodyshape, dr=0.1, combine_mode="subtract"):
        super(OptNPMiner, self).__init__()
        self.input_parameters={'Xshape':Xshape,'headshape':headshape,'bodyshape':bodyshape,
                               'dr':dr,'combine_mode':combine_mode}
        self.combine_mode=combine_mode
        self.heads=nn.ModuleList()
        self.head_drs=nn.ModuleList()
        self.bodies=nn.ModuleList()
        self.body_drs=nn.ModuleList()

        inshape=Xshape
        for i in range(len(headshape)):
            self.heads.append(nn.Linear(inshape,headshape[i]))
            self.head_drs.append(nn.Dropout(dr))
            inshape=headshape[i]
        if combine_mode== 'subtract' or combine_mode== 'add':
            inshape = inshape
        elif combine_mode== "cos":
            inshape=1
        elif combine_mode== 'concat':
            inshape = inshape*2
        for i in range(len(bodyshape)):
            self.bodies.append(nn.Linear(inshape,bodyshape[i]))
            self.body_drs.append(nn.Dropout(dr))
            inshape=bodyshape[i]
        self.out=nn.Linear(inshape,1)
        self.cos= nn.CosineSimilarity(dim=1, eps=1e-6)

        self.support_pos=None
        self.support_neg=None

    def network(self, left, right):
        for i in range(len(self.heads)):
            layer=self.heads[i]
            dr=self.head_drs[i]
            left = dr(t.relu(layer(left)))
            right = dr(t.relu(layer(right)))

        if self.combine_mode== 'subtract':
            res=left-right
        elif self.combine_mode== 'add':
            res=left+right
        elif self.combine_mode == 'cos':
            res = t.clamp(self.cos(left, right).reshape([-1, 1]),0,1)
            return res
        elif self.combine_mode== 'concat':
            res=t.cat([left,right],dim=1)
        # body
        for i in range(len(self.bodies)):
            layer=self.bodies[i]
            dr=self.body_drs[i]
            res=dr(t.relu(layer(res)))
        # output layer
        res=t.sigmoid(self.out(res))
        return res

    def forward(self, x):
        # heads
        inputs=t.split(x,x.shape[1]//2,dim=1)
        left=inputs[0]
        right=inputs[1]
        return self.network(left,right)

    def fit(self, x, y, valset=None, support=None,batch_size=128, epochs=40, printiter=25, use_gpu=False, optimizer=None,
            lr=0.00005, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False,
            lossfn=nn.BCELoss(), earlystop=True,es_thresh=20):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=lr, betas=betas, eps=eps,
                                   weight_decay=weight_decay, amsgrad=amsgrad)
        xtr=t.from_numpy(x).float()
        ytr=t.from_numpy(y).float()
        if support is not None:
            if type(support)==list:
                self.support_pos=support[0]
                self.support_neg=support[1]
            elif type(support)==dict:
                self.support_pos=support

        if valset is not None:
            xv, yv = valset
            xv = t.from_numpy(xv).float()
            yv = t.from_numpy(yv).float()
            if use_gpu:
                xv=xv.cuda()
                yv=yv.cuda()
        if use_gpu:
            self.cuda()
        minvloss=None
        dataset=TensorDataset(xtr,ytr)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        losses=pd.DataFrame(columns=['total','validation'])
        self.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                if use_gpu:
                    x=x.cuda()
                    y=y.cuda()
                optimizer.zero_grad()
                ypred = self(x)
                err = lossfn(ypred,y)
                err.backward()
                optimizer.step()
            with t.no_grad():
                self.eval()
                if use_gpu:
                    tloss = lossfn(self(xtr.cuda()),ytr.cuda())
                else:
                    tloss = lossfn(self(xtr), ytr)
                if use_gpu:
                    tloss=tloss.cpu()
                tloss=tloss.detach().numpy()
                text='epoch:%d\t\ttraining loss:%f'%(epoch+1,tloss)
                losses.loc[epoch+1, 'total'] = float(tloss)
                if valset is not None:
                    vloss = lossfn(self(xv), yv)
                    if use_gpu:
                        vloss = vloss.cpu()
                    vloss = vloss.detach().numpy()
                    losses.loc[epoch+1, 'validation'] = float(vloss)
                    text += '\tvalloss: %f' % vloss
                    if earlystop and epoch > 0:
                        if minvloss is None or vloss <= minvloss:
                            bestep = epoch
                            minvloss = vloss
                            beststate = self.state_dict()
                            logging.getLogger(__name__).info('[earlystop checkpoint] ep%d' % (bestep + 1))
                        elif epoch > bestep + es_thresh:
                            break
                logging.getLogger(__name__).info(text)

                self.train()

        if earlystop and minvloss is not None:
            logging.getLogger(__name__).info('early stopping at epoch %d' % (bestep + 1))
            self.load_state_dict(beststate)

        if use_gpu:
            self.cpu()
        return losses, optimizer

    def predict(self, x):
        self.eval()
        return self(t.from_numpy(x).float()).detach().numpy()

    def predict2(self,left, right):
        self.eval()
        return self.network(t.from_numpy(left).float(),t.from_numpy(right).float()).detach().numpy()

    def predict_support(self,x, n_support=None):
        if n_support is not None:
            px=self.support_pos.sample(n=min(len(self.support_pos),n_support)).reset_index(drop=True)
            nx = self.support_neg.sample(n=min(len(self.support_neg), n_support)).reset_index(drop=True)
        else:
            px = self.support_pos.copy().reset_index(drop=True)
            nx = self.support_neg.copy().reset_index(drop=True)

        X = list(px.columns)
        X2 = ['X%d' % (i + 1) for i in range(len(X), len(X) * 2)]

        res=pd.DataFrame(columns=['px','nx'],index=list(range(len(x))))
        for i in range(len(x)):
            px[X2] = None
            px.loc[0, X2] = x[i]
            px = px.fillna(method='ffill')
            xx = px.values.astype(np.float64)
            ppred = self.predict(xx).mean()

            nx[X2] = None
            nx.loc[0, X2] = x[i]
            nx = nx.fillna(method='ffill')
            xx = nx[X2 + X].values.astype(np.float64)
            npred = self.predict(xx).mean()

            res.loc[i] = [ppred,npred]
            if i%(len(x)//4) in [0,1]:
                logging.getLogger(__name__).info("predicting from support network %d/%d %.2f, %.2f"%(
                    i+1,len(x),ppred,npred))

        return res


def saveModel(model,fn):
    t.save({'model_state_dict':model.state_dict(),
            'support_pos':model.support_pos,
            'support_neg':model.support_neg,
            'params':model.input_parameters

    },fn)


def loadModel(fn):
    checkpoint=t.load(fn)
    input_params=checkpoint['params']
    model = OptNPMiner(**input_params)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.support_pos=checkpoint['support_pos']
    model.support_neg=checkpoint['support_neg']
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
