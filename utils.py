import logging
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix, multilabel_confusion_matrix, roc_auc_score,average_precision_score
from model import OptNCMiner, saveModel
X=['X%d' % i for i in range(1, 1025)]
Y='potency'
label = "SMILES"
xcols = ['X%d' % (i + 1) for i in range(2 * len(X))]
combine_modes = ['subtract', 'add', 'cos', 'concat']
pd.set_option("display.max_columns",100)
pd.set_option("display.width",1000)


def metrics(y, y_pred):
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc=roc_auc_score(y,y_pred)
    conf = confusion_matrix(y, y_pred)
    tn = conf[0, 0]
    fp = conf[0, 1]
    fn = conf[1, 0]
    tp = conf[1, 1]

    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return acc,precision,recall,f1,roc,conf, mcc


def evaluation(y_matrix, y_test, thresh=0.5, noneg=False):
    proteins=list(y_matrix.columns[:-2])
    res=pd.DataFrame(columns=['protein','accuracy','precision','recall','f1','auc','ap','tp','fp','tn','fn','count'])
    for p in proteins:
        temp=y_matrix[[p]]
        ev=pd.DataFrame()
        ev['pred'] = temp[p]>=thresh
        ev['true'] = y_test==p

        acc=accuracy_score(ev['true'],ev['pred'])
        pre=precision_score(ev['true'],ev['pred'])
        rec=recall_score(ev['true'],ev['pred'])
        f1=f1_score(ev['true'],ev['pred'])
        try:
            auc=roc_auc_score(ev['true'],ev['pred'])
        except ValueError:
            auc=-1
        try:
            ap = average_precision_score(ev['true'],ev['pred'])
        except ValueError:
            ap=-1

        conf = confusion_matrix(ev['true'], ev['pred'])
        tn = conf[0, 0]
        fp = conf[0, 1]
        fn = conf[1, 0]
        tp = conf[1, 1]

        res.loc[len(res)]=[p,acc,pre,rec,f1,auc,ap,tp,fp,tn,fn,ev['true'].sum()]
    if not noneg:
        ev['pred'] = y_matrix[proteins].max(axis=1)<0.5
        ev['true'] = y_test=='-'
        acc = accuracy_score(ev['true'], ev['pred'])
        pre = precision_score(ev['true'], ev['pred'])
        rec = recall_score(ev['true'], ev['pred'])
        f1 = f1_score(ev['true'], ev['pred'])
        try:
            auc = roc_auc_score(ev['true'], ev['pred'])
        except ValueError:
            auc=-1
        try:
            ap = average_precision_score(ev['true'], ev['pred'])
        except ValueError:
            ap=-1

        conf = confusion_matrix(ev['true'], ev['pred'])
        if len(conf)!=2:
            tn=-1
            fp=-1
            fn=-1
            tp=-1
        else:
            tn = conf[0, 0]
            fp = conf[0, 1]
            fn = conf[1, 0]
            tp = conf[1, 1]

        res.loc[len(res)] = ['-', acc, pre, rec, f1,auc,ap,tp,fp,tn,fn, ev.true.sum()]
    count=max(res['count'].sum(),1)
    avg = ['average',
           res.accuracy.mean(),
           res.precision.mean(),
           res.recall.mean(),
           res.f1.mean(),
           res.auc.mean(),
           res.ap.mean(),
           -1,
           -1,
           -1,
           -1,
           count
           ]
    wavg=['weighted average',
           (res.accuracy*res['count']).sum(axis=0)/count,
           (res.precision*res['count']).sum(axis=0)/count,
           (res.recall*res['count']).sum(axis=0)/count,
           (res.f1*res['count']).sum(axis=0)/count,
           (res.auc * res['count']).sum(axis=0) / count,
           (res.ap * res['count']).sum(axis=0) / count,
          -1,
          -1,
          -1,
          -1,
           count]
    res.loc[len(res)] = avg
    res.loc[len(res)] = wavg
    return res


def evaluation_multilabel(y_matrix,y_test, thresh=0.5):
    proteins=list(y_matrix.columns[:-2])
    labelcount=y_test[proteins].sum()
    tproteins=list(labelcount[labelcount>0].index)

    acc=accuracy_score(y_test[proteins], y_matrix[proteins] >= thresh)
    pre=precision_score(y_test[proteins], (y_matrix[proteins] >= thresh).astype(float),average='weighted')
    rec=recall_score(y_test[proteins], (y_matrix[proteins] >= thresh).astype(float),average='weighted')
    f1=f1_score(y_test[proteins], (y_matrix[proteins] >= thresh).astype(float),average='weighted')
    roc=roc_auc_score(y_test[tproteins], (y_matrix[tproteins] >= thresh).astype(float), average='weighted',multi_class='ovr')
    ap=average_precision_score(y_test[tproteins], (y_matrix[tproteins] >= thresh).astype(float),average='weighted')
    cf=multilabel_confusion_matrix(y_test[proteins], (y_matrix[proteins] >= thresh).astype(float))

    return acc,pre,rec,f1,roc,ap,cf


def predict_cos(model,x,n_support=100,iter=100, support = None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    final=pd.DataFrame()
    if support is None:
        ss=model.support_pos
    else:
        ss=support
    keys = list(ss.keys())
    support = pd.DataFrame()
    for key in keys:
        temp = ss[key]
        temp = temp[temp[Y] == 1].sample(min(n_support, len(temp[temp[Y] == 1])))
        support = pd.concat([support, temp], axis=0)
    sup = support[X].values.astype(np.float64)

    for i in range(0,len(x),iter):
        logging.getLogger(__name__).info("predicting %i/%d"%(i+1,len(x)))
        xt=x[i:min(i+iter,len(x))]

        left=np.tile(sup,(len(xt),1))
        right=np.repeat(xt,repeats=len(support),axis=0)

        p=t.cosine_similarity(t.from_numpy(left).float(),t.from_numpy(right).float()).detach().numpy()
        r=p.reshape([-1,len(support)])
        res = pd.DataFrame(r,columns=support.fn.values)

        final=pd.concat([final,res],axis=0)
    return final


def predict_general(model, x, n_support=100,iter=500, support = None,random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if support is None:
        ss = model.support_pos
    else:
        ss=support
    keys = list(ss.keys())
    support = pd.DataFrame()
    for key in keys:
        temp = ss[key]
        temp = temp[temp[Y] == 1].sample(min(n_support, len(temp[temp[Y] == 1])))
        support = pd.concat([support, temp], axis=0)
    sup = support[X].values.astype(np.float64)
    ress=[]
    oldxt=0
    for i in range(0,len(x),iter):
        logging.getLogger(__name__).info("predicting %i/%d"%(i+1,len(x)))
        xt=x[i:min(i+iter,len(x))]
        if len(xt)!=oldxt:
            left=np.tile(sup,(len(xt),1))
        right=np.repeat(xt,repeats=len(support),axis=0)

        p=model.predict2(left,right)
        r=p.reshape([-1,len(support)])
        res = pd.DataFrame(r,columns=support.fn.values)
        ress.append(res)
        oldxt=len(xt)
    final=pd.concat(ress,axis=0)
    return final


def trainCycle(params, save=True):
    name = params['name']
    headshape = params['headshape']
    bodyshape = params['bodyshape']
    dr = params['dr']
    lr=params['lr']
    combine_mode = params['combine_mode']

    modelname = "%s_%s_%.2f_%s_%s" % (name,combine_mode, dr,
                                 '-'.join([str(i) for i in headshape]),
                                 '-'.join([str(i) for i in bodyshape]))

    # Load data
    train_raw = pd.read_csv(name+"_sim_train.csv")
    data = pd.read_csv(name+"_sim_data.csv")

    train, valid = train_test_split(data, test_size=0.1)
    x_tr = train[xcols].values.astype(np.float64)
    y_tr = train['Y'].values.astype(np.float64).reshape([-1, 1])
    x_val = valid[xcols].values.astype(np.float64)
    y_val = valid['Y'].values.astype(np.float64).reshape([-1, 1])

    support_set=dict(tuple(train_raw.groupby('fn')))

    model = OptNCMiner(len(X), headshape, bodyshape, dr=dr, combine_mode=combine_mode)
    losses, optimizer = model.fit(x_tr, y_tr, valset=(x_val, y_val), support=support_set,
                                  epochs=500, lr=lr, es_thresh = 50)
    if save:
        saveModel(model, "model_%s.pt" % (modelname))

    return model, losses, optimizer


def transferTrainCycle(model,params, save=True):
    name = params['name']
    tfname= params['tfname']
    headshape = params['headshape']
    bodyshape = params['bodyshape']
    dr = params['dr']
    lr=params['lr']
    combine_mode = params['combine_mode']

    modelname = "%s_%s_%.2f_%s_%s_tf%s" % (name,combine_mode, dr,
                                 '-'.join([str(i) for i in headshape]),
                                 '-'.join([str(i) for i in bodyshape]),tfname)

    # Load data
    train_raw = pd.read_csv(tfname+"_sim_train.csv")
    data = pd.read_csv(tfname+"_sim_data.csv")

    train, valid = train_test_split(data, test_size=0.1)
    x_tr = train[xcols].values.astype(np.float64)
    y_tr = train['Y'].values.astype(np.float64).reshape([-1, 1])
    x_val = valid[xcols].values.astype(np.float64)
    y_val = valid['Y'].values.astype(np.float64).reshape([-1, 1])

    support_set=dict(tuple(train_raw.groupby('fn')))

    losses, optimizer = model.fit(x_tr, y_tr, valset=(x_val, y_val), support=support_set,
                                  epochs=100, lr=lr, es_thresh = 10)
    if save:
        saveModel(model, "model_%s.pt" % (modelname))

    return model, losses, optimizer


def testCycle(model, params, saveName=None, test_raw=None,thresh=0.5,seed=None,noneg=False):
    name=params['name']
    nsupport=params['nsupport']
    niter=params['niter']

    if test_raw is None:
        test_raw = pd.read_csv(name+"_sim_test.csv")

    x_test = test_raw.loc[:, X].values.astype(np.float64)
    y_test=test_raw.loc[:,'class']

    # get probability matrix
    y_proba = predict_general(model,x_test,n_support=nsupport,iter=niter,random_seed=seed)

    # turn to labels
    y_pred = y_proba.idxmax(axis=1)
    y_pred.loc[y_proba.max(axis=1) < 0.5] = '-'

    y_matrix = y_proba.groupby(axis=1, level=0).max()
    y_matrix['LABEL'] = y_test
    y_matrix[label] = test_raw[label]
    y_matrix=y_matrix.reset_index(drop=True)
    res = evaluation(y_matrix, y_test, thresh=thresh, noneg=noneg)
    print(res)

    if saveName is not None:
        y_matrix.to_csv("%s_yproba-matrix.csv" % saveName, index=False)
        y_proba.to_csv("%s_yproba.csv" % saveName, index=False)
        print(saveName)
        with open("%s_results.txt"%saveName,'w') as fi:
            fi.writelines(str(res))
    return y_proba, y_matrix


def tfTestCycle(model, params, saveName=None, test_raw=None,thresh=0.5,seed=None):
    name=params['tfname']
    nsupport=params['nsupport']
    niter=params['niter']

    if test_raw is None:
        test_raw = pd.read_csv(name+"_sim_test.csv")

    x_test = test_raw.loc[:, X].values.astype(np.float64)
    y_test=test_raw.loc[:,'class']

    # get probability matrix
    y_proba = predict_general(model,x_test,n_support=nsupport,iter=niter,random_seed=seed)

    # turn to labels
    y_pred = y_proba.idxmax(axis=1)
    y_pred.loc[y_proba.max(axis=1) < 0.5] = '-'

    y_matrix = y_proba.groupby(axis=1, level=0).max()
    y_matrix['LABEL'] = y_test
    y_matrix[label] = test_raw[label]
    y_matrix=y_matrix.reset_index(drop=True)
    res = evaluation(y_matrix, y_test, thresh=thresh)
    print(res)

    if saveName is not None:
        y_matrix.to_csv("%s_yproba-matrix.csv" % saveName, index=False)
        y_proba.to_csv("%s_yproba.csv" % saveName, index=False)
        print(saveName)
        with open("%s_results.txt"%saveName,'w') as fi:
            fi.writelines(str(res))
    return y_proba, y_matrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
