import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from statistics import pvariance,pstdev,mode,geometric_mean
from PIL import Image,ImageFont,ImageDraw

def buildPolygon(arrV,mbins,outfile):
    a, bins, c = plt.hist(arrV, bins=mbins,color='c',histtype='step')
    l = list(bins)
    l.insert(0, 0)
    l.insert(len(bins) + 1, bins[len(bins) - 1])
    mid = []
    for i in range(len(l) - 1):
        ele = (l[i] + l[i + 1]) / 2
        mid.append(ele)
    x = list(a)
    x.insert(0, 0)
    x.insert(len(a) + 1, 0)
    plt.plot(mid, x, 'bo-')
    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.title("Polygon")
    plt.savefig(outfile)
    plt.close()
def lower_upper_bound(x,operator:str):
    operator=operator.lower().strip()
    if operator.__eq__('lb'):
        for i in range(0, len(x)):
            x[i] = x[i] -0.5
    elif operator.__eq__('up'):
        for i in range(0,len(x)):
            x[i]=x[i]+0.5
    return x

def relative_frequency(x,totalObservation):
    for i in range(0,len(x)):
        x[i]=round(x[i]/totalObservation,4)
    return x
def findNumberClass(arrV):
    return 1+(3.3*math.log(len(arrV),10))

def staster(ls=[]):
   s = pd.Series(ls)
   staText: str = '\nCount: ' + str(s.count()) + '\nMean(Average): ' + str(
      s.mean()) + '\nMedian : ' + str(s.median()) + '\nMode:' + str(mode(s)) + '\nMin Value : ' + str(
      s.min()) + '\nMax Value : ' + str(s.max()) + '\nGeometric Mean: ' + str(
      geometric_mean(s)) + '\nStandard Deviation: ' + str(pstdev(s)) + '\nVariance:' + str(pvariance(s)) \
                  + '\nSample Standard Deviation: ' + str(s.std()) + '\nSample Variance: ' + str(s.var())
   return staText


def drawSolution(mdict: dict = {},outpath: str = ''):

    x_a = [20, 120, 220, 320, 420, 520, 620, 750, 850, 1000, 1200]
    img = Image.open("d1.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'Roboto-Regular.ttf', 12)
    liny: int = 0
    for i in range(0, len(mdict['lower bound'])):
        liny = liny + 30
        t: str = ''

        for j, k in enumerate(mdict.keys()):
            print(x_a[j])
            if liny <= 30:
                draw.text((x_a[j], liny), k, (0, 0, 0), font=font)
            else:
                t: str = str(mdict[k][i])
                draw.text((x_a[j], liny), t, (0, 0, 0), font=font)
        print("***************")

    img.save(f'descriptive_{outpath}.png')

def defineText(t:str):
    t=t.lower().strip()
    t=t.replace(' ','')
    t=t.replace('(','[')
    t=t.replace(')',']')
    if t.count('[')==1 and t.count(']')==1 and t[len(t)-1]==']' and len(t)<300:
        return t
    return 'This bot focus with Statistics Only!\nPlease input with correct format  !\nExample : histogram[1,3,4,5,6,]'
def toCammaText(text: str) -> str:
    text = text.lower().strip()
    text=text.replace('   ','')
    text = text.replace('  ', ' ')
    text = text.replace(' ', ',')
    return text
def textToList(textJeaList:str)->list:
    p = textJeaList.replace('[', ' ').replace(']', ' ').strip()
    p = '[' + p + ']'
    t = toCammaText(p)
    return t

def mSort(c:list)->list:
    for i in range(0, len(c)):
        for j in range(i + 1, len(c)):
            if c[i] >= c[j]:
                c[i], c[j] = c[j], c[i]
    return c
def manualSolution(arrV:dict,secretKey,numberOfClass=0)->dict:
    c:float=0.0
    if numberOfClass==0:
        print("Don't have argument")
        c=findNumberClass(arrV)
    else:
        c=numberOfClass
    c1=round(c)
    n=min(arrV)
    m=max(arrV)
    rang=(m-n)
    classWidth=rang/c
    classWidth=round(classWidth)
    lb = []
    for i in range(0,c1):
        lb.append(n)
        # print(n)
        n=n+classWidth

    up=[]
    for i in range(0,c1):
        if i<=len(lb)-2:
            up.append(lb[i+1]-1)
            # print(up[i])
        else:
            up.append(m)
            # print(up[i])

    midpoint=[]
    for j in range(0,c1):
        k=(lb[j]+up[j])/2
        midpoint.append(k)

    values, bins, bars = plt.hist(arrV, bins=c1, edgecolor='white')
    # plt.xlabel("Age")
    plt.xticks(midpoint)
    plt.ylabel("frequency")
    plt.title('Histogram')
    plt.bar_label(bars, fontsize=20, color='navy')
    plt.margins(x=0.01, y=0.1)
    plt.savefig(f'1_Histogram_{secretKey}.png')
    plt.close()


    totalclass=len(arrV)
    rt:list=[]
    perc:list=[]
    cummulativeFrequency:list=[]

    for i in range(0,len(values)):
        t=values[i]/totalclass
        perc.append(round(t*100,2))
        t=round(t,4)
        rt.append(t)
        if i==0:
            cummulativeFrequency.append(values[i])
        else:
            cf=cummulativeFrequency[i-1]+values[i]
            cummulativeFrequency.append(cf)
    cumrefre:list=[]
    cumrefrepercent:list= []

    for k in range(0,len(values)):
        co=len(cummulativeFrequency)-1
        lastCummulative=cummulativeFrequency[co]
        crf=cummulativeFrequency[k]/lastCummulative
        cumrefre.append(round(crf,4))
        cumrefrepercent.append(round(crf*100,4))

    mlowerbound=lower_upper_bound(lb,'lb')
    mupperbound=lower_upper_bound(up,'up')

    tb="No. of Observation (n) : "+str(len(arrV))
    tb=tb+"\nNo. of Classes C : "+str(c)+" = "+str(c1)
    tb=tb+"\nMin : "+str(n)
    tb=tb+"\nMax : "+str(m)
    tb=tb+"\nRange : "+str(rang)+"\nClass width/size : "+str(round(classWidth))
    tb=tb+staster(arrV)
    print("================================================")
    print(tb)
    print("================================================")
    print("Lower Bound")
    print(lb)
    print("Upper Bound")
    print(up)
    print("================================================")
    print("Lower Bound")
    print(mlowerbound)
    print("Upper Bound")
    print(mupperbound)
    print("================================================")
    print("Frequency")
    print(values)
    print("================================================")
    print("Mid point")
    print(midpoint)
    print("================================================")
    print("Relative Frequency")
    print(rt)
    print("================================================")
    print("Percent")
    print(perc)
    # print("================================================")
    print("Cummulative Frequency")
    print(cummulativeFrequency)
    # print("================================================")
    print("Cummulative Relative Frequency")
    print(cumrefre)
    # print("================================================")
    print("Cummulative Relative Frequency")
    print(cumrefrepercent)

    retriveDict={
            "lower bound": lb,
            "upper bound": up,
            "LowerBound": mlowerbound,
            "UpperBound": mupperbound,
            "Frequency": values,
            "Middlepoint": midpoint,
            "Relative Frequency": rt,
            "Percent": perc,
            "Cummulative Frequency": cummulativeFrequency,
            "Cummulative Relative Frequency": cumrefre,
            "Cummulative Relative Frequency Percent": cumrefrepercent
        }
    drawSolution(retriveDict,secretKey)

    df = pd.DataFrame(retriveDict)
    df.index = np.arange(1, len(df) + 1)
    mtable=tabulate(df, headers='keys', tablefmt='fancy_grid')
    with open(f'tableHistogram_{secretKey}.txt', 'w') as f:
         f.write(mtable)
    df.to_csv(f"csvHistogram_{secretKey}.csv", sep=',', index=False, encoding='utf-8')


    npR="========================Solutions========================\n"
    npR=npR+"No. of Observation (n) : "+str(len(arrV))+"\n"
    npR=npR+"No. of Classes C=1+log(n): "+str(c)+" = "+str(c1)+"\n"
    npR=npR+"Min : "+str(n)+"\n"
    npR=npR+"Max : "+str(m)+"\n"
    npR=npR+"Range : "+str(rang)+"\n"
    npR=npR+"class width/size : "+str(round(classWidth))+"\n"
    ms=npR
    ms =ms+ "{:>13}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}".format("lower bound", "upper bound", "LowerB",
                                                                          "UpperB", "frequency", "midpoint",
                                                                           "Re. Freqency", "Percent", "Cu. Re.","CumReFre","CumReFrePercent")
    
    np.savetxt(f'npHistogram_{secretKey}.txt', np.c_[lb, up,mlowerbound,mupperbound,values,midpoint,rt,perc,cummulativeFrequency,cumrefre,cumrefrepercent],fmt="%15s",header=ms)

    upOgive: list = []
    upOgive.append(lb[0])
    upOgive.extend(up)
    # calculating frequency and class interval
    values, base = np.histogram(arrV, bins=c1)
    # calculating cumulative sum
    cumsum = np.cumsum(values)
    # # plotting  the ogive graph
    plt.plot(base[1:],cumsum, color='blue', marker='o', linestyle='solid')
    plt.xticks(upOgive)
    plt.yticks(cumsum)
    plt.title('Ogive Graph')
    plt.xlabel('Marks in End-Term')
    plt.ylabel('Cumulative Frequency')
    plt.grid(axis='y')
    plt.savefig(f"4_Ogive02_{secretKey}.png")
    plt.close()

    w: list = []
    w.append(0)
    w.extend(cummulativeFrequency)
    plt.xticks(upOgive)
    plt.yticks(cummulativeFrequency)
    plt.title("Ogive")
    plt.ylabel('Cumulative Frequency')
    plt.plot(upOgive,w,color='blue', marker='o')
    plt.grid(axis='both')
    plt.axis([upOgive[0],upOgive[len(upOgive)-1]+5,0,w[len(w)-1]+5])
    plt.savefig(f"3_Ogive01_{secretKey}.png")
    plt.close()
    buildPolygon(arrV,c1,f"2_PolyganHistogram_{secretKey}.png")
    return retriveDict
