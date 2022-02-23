import logging
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def preprocess_multilabel(path, X, Y, label, n_pos, n_neg, exname=''):
    df = pd.read_csv(path+"/train.csv")
    train,test=train_test_split(df,test_size=0.1)
    train.to_csv(path+exname+"ml_sim_train.csv",index=False)
    test.to_csv(path +exname+ "ml_sim_test.csv", index=False)

    toSingleLabel(path+exname)
    train=pd.read_csv("%s_sim_train.csv"%(path+exname))

    xcols = ['X%d' % (i+1) for i in range(2 * len(X))]
    data=pd.DataFrame(columns=['left','right','Y']+xcols, index=range(n_pos+n_neg))
    for i in range(n_pos):
        left=train.sample(1).iloc[0]
        rightr = train[train['class']==left['class']].sample(1)
        while len(rightr)<=0:
            left = train.sample(1).iloc[0]
            rightr = train[train['class'] == left['class']].sample(1)
        right=rightr.iloc[0]
        l = list(left[X].values)
        r = list(right[X].values)
        x = l + r
        data.loc[i] = [left[label], right[label], 1] + x
        logging.getLogger(__name__).info("%d/%d" % (i+1, n_pos + n_neg))
    for i in range(n_neg):
        left=train.sample(1).iloc[0]
        rightr = train[train['class']!=left['class']].sample(1)
        while len(rightr)<=0:
            left = train.sample(1).iloc[0]
            rightr = train[train['class'] != left['class']].sample(1)
        right=rightr.iloc[0]

        l = list(left[X].values)
        r = list(right[X].values)
        x = l + r
        data.loc[i+n_pos] = [left[label], right[label], 0] + x
        logging.getLogger(__name__).info("%d/%d" % (i+1+n_pos, n_pos + n_neg))
    data.sample(frac=1).to_csv(path + exname + '_sim_data.csv', index=False)


def toSingleLabel(name):
    for task in ["_sim_test.csv","_sim_train.csv"]:

        file = pd.read_csv('%sml%s'%(name,task))
        pro_m = file.iloc[:, 1026:]

        cl = []
        for i in range(0, len(pro_m)):
            pro = pro_m.iloc[i, :]
            pos = pro[pro == 1]
            cl.append(pos.sample(1).index[0])

        data = file.iloc[:, :1026]
        data['class']=cl
        data['fn']=cl
        data.to_csv('%s%s'%(name,task), index=False)


def preprocess_directory(path, X, Y, label, n_pos, n_neg, nnpairs=0, exceptions=(), exname='', testsize=0.1, maxrows=40000):
    fns=glob.glob(path+'/*.csv')
    dfs=[]
    pp=0
    nn=0
    for i,fn in enumerate(fns):
        proname=fn.split('\\')[1].split('.')[0]
        if proname not in list(exceptions):
            logging.getLogger(__name__).info("reading %s"%fn)
            try:
                temp=pd.read_csv(fn,sep=',', nrows=maxrows)
                pp += len(temp[temp[Y] == 1])
                nn += len(temp[temp[Y] == 0])

            except KeyError:
                temp=pd.read_csv(fn,sep='\t',nrows=maxrows)
                pp += len(temp[temp[Y] == 1])
                nn += len(temp[temp[Y] == 0])
            temp['fn']=proname
        dfs.append(temp)
    df=pd.concat(dfs,axis=0)
    df=df.sample(frac=1).reset_index(drop=True)
    df['class']=df.fn
    df.loc[df[Y]==0,'class']='-'
    cnt=df.groupby(label).count()[Y]
    cnt.name='count'
    df2 = df.join(cnt, on=label)


    ss=StratifiedShuffleSplit(n_splits=1,test_size=testsize)
    (tridx,teidx) = list(ss.split(df[df2['count']==1][X], df2[df2['count']==1]['class']))[0]
    train=df.drop(teidx,axis=0)
    test=df.loc[teidx]
    train.to_csv(path+exname+"_sim_train.csv",index=False)
    test.to_csv(path +exname+ "_sim_test.csv", index=False)
    xcols = ['X%d' % (i+1) for i in range(2 * len(X))]
    overlaps={}
    data=pd.DataFrame(columns=['left','right','Y']+xcols, index=range(n_pos+n_neg))
    proteins=df.fn.unique()
    for i in range(n_pos):
        p=random.choice(proteins)
        temp=train[(train['fn']==p) & (train[Y]==1)].sample(2,axis=0)
        l = list(temp.iloc[0][X].values)
        r = list(temp.iloc[1][X].values)
        x = l + r
        data.loc[i] = [temp.iloc[0][label], temp.iloc[1][label], 1] + x
        logging.getLogger(__name__).info("%d/%d" % (i+1, n_pos + n_neg+nnpairs))
    for i in range(n_neg):
        p1,p2=random.sample(list(proteins),2)
        if (p1,p2) not in overlaps:
            overlaps[(p1,p2)]=pd.merge(train[train['fn'] == p1], train[train['fn'] == p2], how='inner')
        m = overlaps[(p1,p2)]

        left=train[(train['fn']==p1) &(~train[label].isin(m[label].unique()))& (train[Y]==1)].sample(1,axis=0).iloc[0]
        right=train[(train.fn==p2)&(~train[label].isin(m[label].unique()))].sample(1,axis=0).iloc[0]

        x = list(left[X]) + list(right[X])
        data.loc[i + n_pos] = [left[label], right[label], 0] + x
        logging.getLogger(__name__).info("%d/%d" % (i + n_pos+1, n_pos + n_neg+nnpairs))
    for i in range(nnpairs):
        p = random.choice(proteins)
        temp = train[(train['fn'] == p) & (train[Y] == 0)].sample(2, axis=0)
        l = list(temp.iloc[0][X].values)
        r = list(temp.iloc[1][X].values)
        x = l + r
        data.loc[i+n_pos+n_neg] = [temp.iloc[0][label], temp.iloc[1][label], 1] + x
        logging.getLogger(__name__).info("%d/%d" % (i+n_pos+n_neg+1, n_pos + n_neg+nnpairs))
    data.sample(frac=1).to_csv(path+exname+'_sim_data.csv',index=False)


def preprocess(df, X, Y, label, n_pos, n_neg):
    xcols=['X%d'%i for i in range(2*len(X))]
    data = pd.DataFrame(columns=['left', 'right', 'Y']+xcols, index=range(n_pos+n_neg))

    for i in range(n_pos // 2):
        # pos pos
        temp = df[df[Y] == 1].sample(2, axis=0)
        l=list(temp.iloc[0][X].values)
        r=list(temp.iloc[1][X].values)
        x=l+r
        data.loc[i] = [temp.iloc[0][label], temp.iloc[1][label], 1]+x
        logging.getLogger(__name__).info("%d/%d"%(i,n_pos+n_neg))

    for i in range(n_pos // 2):
        # neg neg
        temp = df[df[Y] == 0].sample(2, axis=0)
        l=list(temp.iloc[0][X].values)
        r=list(temp.iloc[1][X].values)
        x=l+r
        data.loc[i+n_pos//2] = [temp.iloc[0][label], temp.iloc[1][label], 1]+x
        logging.getLogger(__name__).info("%d/%d"%(i+n_pos//2,n_pos+n_neg))

    for i in range(n_neg):
        # pos neg
        left = df[df[Y] == 1].sample(1, axis=0).iloc[0]
        right = df[df[Y] == 0].sample(1, axis=0).iloc[0]
        x= list(left[X])+list(right[X])
        data.loc[i+n_pos] = [left[label], right[label], 0]+x
        logging.getLogger(__name__).info("%d/%d"%(i+n_pos,n_pos+n_neg))

    return data.sample(frac=1)


def addSupport(Y,path,support,test_raw, keepTest=5, testPerc=0.1,include=()):
    fns=glob.glob(path+'/*.csv')
    for i,fn in enumerate(fns):
        try:
            temp=pd.read_csv(fn)
            temp[Y]
        except KeyError:
            temp=pd.read_csv(fn,sep='\t')

        name=fn.split('\\')[1].split('.')[0]
        if len(include)>0:
            if name not in include:
                continue
        temp['fn']=name
        temp['class'] = temp.fn
        temp=temp[temp[Y]==1]
        ntest = max([int(len(temp)*testPerc),keepTest])
        if len(temp)<keepTest *2:
            ntest = 1
        tetemp = temp.sample(ntest)
        sutemp = temp.drop(tetemp.index)

        support[name]=sutemp
        test_raw=pd.concat([test_raw,tetemp],axis=0)
    test_raw=test_raw.reset_index(drop=True)
    return support, test_raw


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
