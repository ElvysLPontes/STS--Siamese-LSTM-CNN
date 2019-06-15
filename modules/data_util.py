# -*- coding: utf-8 -*-
""" Functions file
We reused the approach of Mueller and Thyagarajan to expand the SICK dataset. https://github.com/aditya1503/Siamese-LSTM
"""

from nltk.corpus import stopwords
import pickle, gensim, random
import numpy as np

# Load data
cachedStopWords = stopwords.words("english")
d2              = pickle.load(open("data/synsem.p",'rb'))
f               = open("data/dwords.p",'rb')
dtr             = pickle.load(f, encoding='latin1')

print ("Loading Word2Vec...")
model = gensim.models.KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz",binary=True)
print ("Word2Vec loaded!")

def embed(stmx,training):
    """
    From https://github.com/aditya1503/Siamese-LSTM
    """
    dmtr=np.zeros((stmx.shape[0],300), dtype=np.float32)
    count=0
    while(count<len(stmx)):
        if stmx[count]=='<end>':
            count+=1
            continue
        if stmx[count] in dtr:
            dmtr[count]=model[dtr[stmx[count]]]
            count+=1
        elif stmx[count] in model:
            dmtr[count]=model[stmx[count]]
            count+=1
        else:
            dmtr[count]=dmtr[count]
            count+=1
    return dmtr

def getmtr(xa, maxlen):
    ls=[]
    for i in range(0,len(xa)):
        q=xa[i].split()
        while(len(q)<maxlen):
            q.append('<end>')
        else:
            q = q[:maxlen]
        ls.append(q)
    xa=np.array(ls)
    return xa

def prepare_data(data, maxlen, training):
    xa1 = [ data[i][0] for i in range(0,len(data)) ]
    xb1 = [ data[i][1] for i in range(0,len(data)) ]
    y   = [ data[i][2] for i in range(0,len(data)) ]

    lengths1, lengths2 =[],[]
    for i in xa1:
        if len(i.split()) > maxlen:
            lengths1.append(maxlen)
        else:
            lengths1.append(len(i.split()))
    for i in xb1:
        if len(i.split()) > maxlen:
            lengths2.append(maxlen)
        else:
            lengths2.append(len(i.split()))

    words1 = getmtr(xa1, maxlen)
    emb1   = [ embed(words,training) for words in words1]
    words2 = getmtr(xb1, maxlen)
    emb2   = [ embed(words,training) for words in words2]

    y = np.array(y, dtype=np.float32)

    return [ emb1, lengths1, emb2, lengths2, y ]

def chsyn(s,trn):
    """
    From https://github.com/aditya1503/Siamese-LSTM
    """
    cnt=0
    global flg
    x2=s.split()
    x=[]

    for i in x2:
        x.append(i)
    for i in range(0,len(x)):
        q=x[i]
        mst=''
        if q not in d2:
            continue

        if q in cachedStopWords or q.title() in cachedStopWords or q.lower() in cachedStopWords:
            #print q,"skipped"
            continue
        if q in d2 or q.lower() in d2:
            if q in d2:
                mst=findsim(q)
            #print q,mst
            elif q.lower() in d2:
                mst=findsim(q)
            if q not in model:
                mst=''
                continue

        if mst in model:
            if q==mst:
                mst=''

                continue
            if model.similarity(q,mst)<0.6:
                continue
            #print x[i],mst
            x[i]=mst
            if q.find('ing')!=-1:
                if x[i]+'ing' in model:
                    x[i]+='ing'
                if x[i][:-1]+'ing' in model:
                    x[i]=x[i][:-1]+'ing'
            if q.find('ed')!=-1:
                if x[i]+'ed' in model:
                    x[i]+='ed'
                if x[i][:-1]+'ed' in model:
                    x[i]=x[i][:-1]+'ed'
            cnt+=1
            mst=''
    return ' '.join(x),cnt

def findsim(wd):
    """
    From https://github.com/aditya1503/Siamese-LSTM
    """
    syns=d2[wd]
    x=random.randint(0,len(syns)-1)
    return syns[x]

def check(sa,sb,dat):
    """
    From https://github.com/aditya1503/Siamese-LSTM
    """
    for i in dat:
        if sa==i[0] and sb==i[1]:
            return False
        if sa==i[1] and sb==i[0]:
            return False
    return True

def expand(data):
    """
    From https://github.com/aditya1503/Siamese-LSTM
    """
    n=[]
    for m in range(0,10):
        for i in data:
            sa,cnt1=chsyn(i[0],data)
            sb,cnt2=chsyn(i[1],data)
            if cnt1>0 and cnt2>0:
                l1=[sa,sb,i[2]]
                n.append(l1)
    for i in n:
        if check(i[0],i[1],data):
            data.append(i)
    return data
