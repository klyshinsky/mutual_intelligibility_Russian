import pandas as pd
import numpy as np

from math import sqrt
from scipy.stats import pearsonr

import matplotlib as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text as plText
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.transforms as transforms

from ipywidgets import interact, interactive_output, Select, HBox, FloatSlider
from IPython.display import display

import copy

def drawCompleteness(test_id, users_answers):
    dat = users_answers[(~users_answers["answer"].isna())&(users_answers["test_id"]==test_id)]
    dat = dat[["id_user", "answer"]].groupby("id_user").count()
    dat.hist()

def calcCIFrame(data2: pd.DataFrame, field: str) -> pd.DataFrame:
    data = data2.groupby([field]).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.index = [i for i in data.index]
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    return data

def moveCollection(ax, coll_no, size, direction):
    offset = transforms.ScaledTranslation(size, 0, ax.figure.dpi_scale_trans)
    trans = ax.collections[coll_no].get_transform()
    if direction == 'left':
        ax.collections[coll_no].set_transform(trans - offset)
    elif direction == 'right':
        ax.collections[coll_no].set_transform(trans + offset)
    return offset

def moveLines(ax, offset, direction='left'):
    for line in ax.lines:
        trans = line.get_transform()
        if direction == 'left':
            line.set_transform(trans - offset)
        elif direction == 'right':
            line.set_transform(trans + offset)

def orderIndex(data: pd.DataFrame, order: list) -> pd.DataFrame:
    o2 = []
    for ind in order:
        if ind in data.index:
            o2.append(ind)
            
    data4 = pd.DataFrame()
    for ind in o2:
        for col in data.columns:
            data4.loc[ind, col] = data.loc[ind, col]
            
    return data4

# Drawing data on the number of partisipants.
def show_langs(test: str, user_data, axes):
    if test == 'All':
        data2 = user_data[['parallel_lang', 'mean_mark']]
    else:
        data2 = user_data[['parallel_lang', 'mean_mark']][user_data.test_id == int(test)]
    
    order = ['No Parallel Text', 'Ukranian', 'Belorussian', 'Bulgarian', 
             'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    order2= ['Контроль', 'Укр.', 'Бел.', 'Болг.', 
             'Пол.', 'Чеш.', 'Словацк.', 'Серб.', 'Словенск.']
    
    data = calcCIFrame(data2, 'parallel_lang')
    data4 = orderIndex(data, order)
#    display(data)
    display(data4.T)
    
    axes.clear()

    axes.set_ylim(0, 1)
    sns.pointplot(x='parallel_lang', y="mean_mark", data=data2, palette="Dark2",#color="#FF0000", 
                  markers="x", order=order, errwidth=2.5, capsize=0.1, 
                  join=False, legend=None, ax=axes) 

    offset = moveCollection(axes, 0, 12/72., "left")
    moveLines(axes, offset)
    #sns.boxplot(x="speciality", y="uavg", data=datas[(datas.ln1==lang)], order=["yes", "no"], notch=True, ax=axes[0])
    sns.swarmplot(x='parallel_lang', y="mean_mark", data=data2, order=order, 
                  palette="Set2", size=3, color=".3", linewidth=0, ax=axes, alpha=0.5) 
    
#    a = data2.iloc[0, 0]
    
    
    patches = [Rectangle((-0.5, data4.iloc[0, 2]), 10, data4.iloc[0, 3] - data4.iloc[0, 2], alpha = 0.3, edgecolor='#0011DD')]
    pc = PatchCollection(patches, alpha = 0.1, facecolor='#0011DD')

    axes.add_collection(pc)  
    plt.ylabel('Средняя понимаемость для информанта')
    axes.set_xticklabels(order2)
#    axes.set_xticklabels(axes.get_xticklabels(), rotation=15)#, ha='right')
    plt.xlabel('Язык параллельного текста')
#    axes.xaxis.set_label_coords(2, 1)
#    axes.set_title("")

def show_langs_by_tests(user_data, axes):
    data2 = user_data[['parallel_lang', 'mean_mark', 'test_id']]
    data = data2.groupby(['parallel_lang', 'test_id']).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    data['test_id'] = [i[1] for i in data.index]
    data['language'] = [i[0] for i in data.index]
    #data.index = [i[0]+' '+str(i[1]) for i in data.index]
    data.index = range(len(data.index))
    #data.columns
    data

    order = ['No Parallel Text', 'Ukranian', 'Belorussian', 'Bulgarian', 
             'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    order2= ['Контроль', 'Укр.', 'Бел.', 'Болг.', 
             'Пол.', 'Чеш.', 'Словацк.', 'Серб.', 'Словенск.']

    fig, axes = plt.subplots(1, 1, figsize=(7,4), num=101)
    axes.set_ylim(0, 1)

    markers = ['o', 'x', 'd', '^', '+', 's', 'p', '*', 'v']

    numl = len(order)
    #for i in range(6):
    for i in range(numl):
        #sns.pointplot(x='parallel_lang', y="uavg", data=data2[data2.test_id==6-i], palette="Dark2",#color="#FF0000", 
        sns.pointplot(x='test_id', y="mean_mark", data=data2[data2.parallel_lang==order[numl-i-1]], palette="Dark2",#color="#FF0000", 
                      markers=markers[i], errwidth=2.5, capsize=0.1, #order=order, 
                      join=False, legend=None, axes=axes) 
        #xs = axes.get_xticks()
        for j in range(i+1):
            offset = moveCollection(axes, j, 6/70., "right")
        moveLines(axes, offset, "right")
    axes.set_xlim(0, 6)
    plt.ylabel('Средняя понятность по тесту')
    plt.xlabel('Номер теста')
    axes.set_xticklabels([' '*20+str(i) for i in range(1, 7)])

    axes.grid()

    _ = axes.legend(handles=[Line2D([], [], marker=m, color='black') for m in markers[::-1]], labels=order2,
                    loc = 'lower right', ncol=2, title='Языки')

    fig.savefig('img_res/Fig_2_distribution_users_tests.png', dpi = 600)

def show_tests_by_langs(user_data, axes):
    data2 = user_data[['parallel_lang', 'mean_mark', 'test_id']]
    data = data2.groupby(['parallel_lang', 'test_id']).agg(['count', 'mean', 'std'])
    data['ci'] = 1.96 * data[('mean_mark', 'std')] / np.sqrt(data[('mean_mark', 'count')])
    data['ci_min'] = data[('mean_mark', 'mean')] - data['ci']
    data['ci_max'] = data[('mean_mark', 'mean')] + data['ci']
    data = data.drop([('mean_mark', 'std'), 'ci'], axis=1)
    data.columns = ['count', 'mean', 'ci_min', 'ci_max']
    data['test_id'] = [i[1] for i in data.index]
    data['language'] = [i[0] for i in data.index]
    #data.index = [i[0]+' '+str(i[1]) for i in data.index]
    data.index = range(len(data.index))
    #data.columns
    data

    order = ['No Parallel Text', 'Ukranian', 'Belorussian', 'Bulgarian', 
             'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    order2= ['Контроль', 'Укр.', 'Бел.', 'Болг.', 
             'Пол.', 'Чеш.', 'Словацк.', 'Серб.', 'Словенск.']

    fig, axes = plt.subplots(1, 1, figsize=(7,4), num=102)
    axes.set_ylim(0, 1)

    markers = ['o', 'x', 'd', '^', 's', '*']

    numl = len(order)
    for i in range(6):
    #for i in range(numl):
        sns.pointplot(x='parallel_lang', y="mean_mark", data=data2[data2.test_id==6-i], palette="Dark2",#color="#FF0000", 
        #sns.pointplot(x='test_id', y="mean_mark", data=data2[data2.parallel_lang==order[numl-i-1]], palette="Dark2",#color="#FF0000", 
                      markers=markers[i], errwidth=2.5, capsize=0.1, #order=order, 
                      join=False, legend=None, axes=axes) 
        #xs = axes.get_xticks()
        for j in range(i+1):
            offset = moveCollection(axes, j, 6/70., "right")
        moveLines(axes, offset, "right")
    axes.set_xlim(0, numl)
    plt.ylabel('Средняя понятность по тесту')
    plt.xlabel('Параллельный язык')
    axes.set_xticklabels([' '*15+i for i in order2])

    axes.grid()

    _ = axes.legend(handles=[Line2D([], [], marker=m, color='black') for m in markers[::-1]], 
                    labels=range(1, 7), loc = 'lower right', ncol=2, title='Тесты')

    fig.savefig('img_res/Fig_3_distribution_users_languagess.png', dpi = 600)

# Drawing data on the number of partisipants.
def show_age(field: str, test: str, user_data, axes):
    if test == 'All':
        data2 = user_data[[field, 'mean_mark']]
        data3 = user_data[[field, 'mean_mark', 'parallel_lang']]
    else:
        data2 = user_data[[field, 'mean_mark']][all_users.test_id == int(test)]
        data3 = user_data[[field, 'mean_mark', 'parallel_lang']][all_users.test_id == int(test)]

    if field == 'age':
        order = ['sch', 'bak', 'mag', 'asp', 'fin', 'unk', 'unk2']
    elif field == 'spec':
        order = ['yes', 'no', 'unk', 'unk2']
        
    data = calcCIFrame(data2, field)
    data = orderIndex(data, order)
    display(data.T)
  
    data3['no parallel text'] = data3['parallel_lang'] == 'No Parallel Text'
    
    axes.clear()

    axes.set_ylim(0, 1)
    sns.pointplot(x=field, y="mean_mark", data=data3, color="#FF0000", 
                  markers="x", order=order, errwidth=1.5, capsize=0.1, 
                  join=False, legend=None, ax=axes) 
    offset = moveCollection(axes, 0, 6/72., "left")
    moveLines(axes, offset)

    #sns.boxplot(x="speciality", y="uavg", data=datas[(datas.ln1==lang)], order=["yes", "no"], notch=True, ax=axes[0])
    sns.swarmplot(x=field, y="mean_mark", data=data3, order=order, 
                  palette="Set2", hue='no parallel text',
                    size=4, color=".3", linewidth=0, ax=axes, alpha=0.7)
    #axes.set_title(lang)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=15, ha='right')

    
    # If there is any difference between linguists and non-linguists?
    # It is for pupils and bachelors, but not for masters and alumni.
    if field == 'age':
        if test == 'All':
            datal = user_data[[field, 'mean_mark', 'parallel_lang']][user_data.speciality=='yes']
            sns.pointplot(x=field, y="mean_mark", data=datal, color="#00FF00", 
                          markers="x", order=order, errwidth=1.5, capsize=0.1, 
                          join=False, legend=None, ax=axes) 
            datan = user_data[[field, 'mean_mark', 'parallel_lang']][user_data.speciality=='no']
            sns.pointplot(x=field, y="mean_mark", data=datan, color="#0000FF", 
                          markers="x", order=order, errwidth=1.5, capsize=0.1, 
                          join=False, legend=None, ax=axes) 
        else:
            datal = user_data[[field, 'mean_mark', 'parallel_lang']] \
                             [(all_users.test_id == int(test)) & (user_data.speciality=='yes')]
            sns.pointplot(x=field, y="mean_mark", data=datal, color="#00FF00", 
                          markers="x", order=order, errwidth=1.5, capsize=0.1, 
                          join=False, legend=None, ax=axes) 
            datan = user_data[[field, 'mean_mark', 'parallel_lang']] \
                             [(all_users.test_id == int(test)) & (user_data.speciality=='no')]
            sns.pointplot(x=field, y="mean_mark", data=datan, color="#0000FF", 
                          markers="x", order=order, errwidth=1.5, capsize=0.1, 
                          join=False, legend=None, ax=axes) 
    
    offset = moveCollection(axes, 0, 12/72., "left")
    moveLines(axes, offset)
    _ = moveCollection(axes, -1, 12/72., "left")
    _ = moveCollection(axes, -2, 12/72., "left")
    axes.set_ylabel('Понимаемость')
    
def drawForeign(test_no, user_data, axes):
    if test_no == 'All':
        dat = user_data
    else:
        dat = user_data[user_data.test_id==int(test_no)]
    flangs = dat.groupby('known_langs').count()
#    flangs = list(flangs[flangs['native_lang']>2].index)
    flangs = list(flangs.index)
    dat = dat[dat['known_langs'].map(lambda x: x in flangs)].copy()
    #display(dat)
    axes.clear()
    
    sns.pointplot(x='known_langs', y='mean_mark', data = dat, markers='x', color='r', 
                  errwidth=1.5, capsize=0.2, alpha=0.5, join=False, axes=axes)
    sns.swarmplot(x='known_langs', y='mean_mark', data = dat, ax=axes)
    axes.set_ylabel('Понимаемость')
    
def makeWords(langs, texts, set_no):
    words=pd.DataFrame()
    for lang in langs:
        words0=[]
        for i, sent in enumerate(texts['sent'][(texts['id_set']==set_no) & (texts['lang_name']==lang)]):
            [words0.append(s) for j, s in enumerate(sent.split('_')) if j%2==1]
        words[lang]=words0
    return words

def makeStatistics(words, wo_frame, rf, sim, set_no):
    tmpres=[]
    langs3=[l for l in rf.columns if l in words.columns]
    count_pos = list(wo_frame.columns).index('mean_mark')
    for j, word in enumerate(words['Russian']):
        for lang in langs3:
            a=rf[lang][set_no, j]
            b=wo_frame[(wo_frame.answer_no == j) & (wo_frame['test_id'] == set_no)].iloc[0, count_pos]
            if b!=0:
                tmpres.append({'orig':word, 'lang':lang, 'fword':words[lang][j], 'type':sim[lang][j], 'wo_par':b, 'w_par':a, 'rel':a/b})
            else:
                tmpres.append({'orig':word, 'lang':lang, 'fword':words[lang][j], 'type':sim[lang][j], 'wo_par':b, 'w_par':a, 'rel':0})

    return pd.DataFrame(data=tmpres, columns=['orig', 'lang', 'fword', 'type', 'wo_par', 'w_par', 'rel'])

def showWords(test_no: str, text_frame):
    if test_no == '1' or test_no == '2':
        langs = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs, text_frame, test_no))
    elif test_no == '3':
        langs3 = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs3, text_frame, '3'))
    elif test_no == '4' or test_no == '5':
        langs4 = ['Russian', 'Ukranian', 'Belorussian', 'Polish', 'Bulgarian', 'Czech', 'Slovak', 'Serbian', 'Slovene']
        display(makeWords(langs4, text_frame, test_no))
    else:
        langs4 = ['Russian', 'Ukranian', 'Belorussian', 'Polish', 'Bulgarian', 'Czech', 'Serbian', 'Slovene']
        display(makeWords(langs4, text_frame, test_no))
        
# These data are for statistical image generation.
sim_rus1={#    0        10        20        30        40
"Belorussian":"101!121122201111112110111121021111011011100",
"Bulgarian"  :"200!!100201020111122201122012201!12112211!0",
"Czech"      :"010!00020021010011222!01000202!2!2221!21000",
"Polish"     :"021!01102022102111202201010002!000!01001002",
"Serbian"    :"01000100!011210111220!11220100!1!122122100!",
"Slovak"     :"220!00000022010101220!110002220210211020002",
"Ukranian"   :"1211121020121111012110111221010111001210100",   
"Slovene"    :"002!012!!020000011021!01222002!2!020102112!",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus2={#    0        10        20        30        40
"Belorussian":"110211112110001110210202!0!!111121201020",
"Bulgarian"  :"!10111200021!121100100010!1211120121!120",
"Czech"      :"!!02010201002!21000100200!2001102121!0!0",
"Polish"     :"10000001!120!02100010022!0101112!021!020",
"Serbian"    :"!0011010!020012110!100020!10111!11210120",
"Slovak"     :"!00201022100022100010000!!2011120121!120",
"Ukranian"   :"21011110!110201110010102!0221111!1201220",   
"Slovene"    :"!201100000!0202102!1000222!2111!!021!1!0",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus3={#    0        10        20        30
"Belorussian":"100220101222!122111!1!121!10!211121",
"Bulgarian"  :"12100111!0002100112!1!101!11!111101",
"Czech"      :"110220!002000!00!1!22!0!!!!2!1!102!",
"Polish"     :"100222!01202!!00!12!2!12!!!2!1!1121",
"Ukranian"   :"111022!012!2!!00!1211!12!!!2!1!1121",
"Slovak"     :"111220!0220!0!!0!1!!2!22!2!2!!2111!",
"Slovene"    :"102222!200020!!0!1122!12!1!2!!2110!",
"Serbian"    :"11!0211!012001101100201!1!10!101122",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus4={#    0        10        20        30        40
"Ukranian"   :"110!1!1!21212!22!1!2!01!10001!21000!1001!00!",
"Belorussian":"01!1212020212222!2!2!0121!001!2!!00210011000",
"Polish"     :"20!12!202!022022!2!2!01010!01!20!00221121000",
"Bulgarian"  :"02012212!1!2!121!!22111!211111010!020121110!",
"Czech"      :"02!11!!20!200021!122201010!!0!01202022111220",
"Slovak"     :"!2201!220!210021!2!02!1010!000012010121!1!!2",
"Slovene"    :"22000!0002220021!2122202!2202220!0!210121110",
"Serbian"    :"!20112000122221200222102!2121!02!00!0101100!",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus5={#    0        10        20        30        40
"Ukranian"   :"211101111021110111211110!10020102!010",
"Belorussian":"011001010001100!1102111!!000201022!10",
"Polish"     :"21100101100200201102111000!!101010010",
"Bulgarian"  :"222!00!!10221!2!12!01111!2111221121!1",
"Czech"      :"221!!0!110221!201002111!!000101111010",
"Slovak"     :"22!!1!!112221!!111021110!000111010210",
"Slovene"    :"2!20!20!!!2200!1!102111!02111!!010212",
"Serbian"    :"2!2!!!!!00221!!011!!1110001112!000002",
}

# 0 - nothing similar, 1 - same word root, 2 - has word with a similar sence, ! - false translator's friend.
sim_rus6={#    0        10        20        30        40
"Ukranian"   :"!1110211210012210001001111210111101",
"Belorussian":"!11112110101111!0011010111120111101",
"Polish"     :"!11122110100122!0022001121100122200",
"Bulgarian"  :"111110011101111112!10!0111000121112",
"Czech"      :"!11101200000100!00010!0!0!1!100001!",
"Serbian"    :"11111201211!1!!1!!!10!01!102!211112",
"Slovene"    :"!12120211100122!2202002221!22!22120",
}


# Generates an image for statistical analysis of dependency 
# between phonetical similarity and intelligibility.
def processIntelligibility(data, texts):
    all_frame = data[data['parallel_lang'] != 'No Parallel Text'].copy()
    wo_frame = data[data['parallel_lang'] == 'No Parallel Text'].copy()
    all_frame['test_id'] = all_frame['test_id'].astype("str")
    wo_frame['test_id'] = wo_frame['test_id'].astype("str")
    
    langs  = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs3 = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs4 = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Slovak', 'Serbian', 'Slovene']
    langs5 = ['Russian', 'Ukranian', 'Belorussian', 'Bulgarian', 'Polish', 'Czech', 'Serbian', 'Slovene']

    rf=all_frame.pivot_table(values="mean_mark", index=['test_id', 'answer_no'], columns='parallel_lang')

    # !!! If you will not use these wordsN, it will be faster. But if you suddenly will need the words list, you will reimplement this.
    words1=makeWords(langs, texts, '1')
    words2=makeWords(langs, texts, '2')
    words3=makeWords(langs3, texts, '3')  
    words4=makeWords(langs4, texts, '4')  
    words5=makeWords(langs4, texts, '5')  
    words6=makeWords(langs5, texts, '6')  
    changes={"Polish":[(16, 17), (23, 24), (36, 37)], "Czech":[(16, 17), (26, 27)], "Slovak":[(26, 27)], "Bulgarian":[(16, 17), (23, 24)], "Belorussian":[(40, 41)]}

    # In the first text some words was swapped during the translation.
    for c_lang, chang in changes.items():
        for ch in chang:
            words1[c_lang][ch[0]], words1[c_lang][ch[1]] = words1[c_lang][ch[1]], words1[c_lang][ch[0]]
            rf[c_lang]['1',ch[0]], rf[c_lang]['1',ch[1]] = rf[c_lang]['1',ch[1]], rf[c_lang]['1',ch[0]]

    all_resn = []
    all_resn.append(makeStatistics(words1, wo_frame, rf, sim_rus1, '1'))
    all_resn.append(makeStatistics(words2, wo_frame, rf, sim_rus2, '2'))
    all_resn.append(makeStatistics(words3, wo_frame, rf, sim_rus3, '3'))
    all_resn.append(makeStatistics(words4, wo_frame, rf, sim_rus4, '4'))
    all_resn.append(makeStatistics(words5, wo_frame, rf, sim_rus5, '5'))
    all_resn.append(makeStatistics(words6, wo_frame, rf, sim_rus6, '6'))
    all_tests=['test 1,', 'test 2,', 'test 3,', 'test 4,', 'test 5,', 'test 6,']
    return all_resn, all_tests

draw_shift = 2./72.

# Draws several results of test on the same figure.
def drawIntelligibility2(dats, nams, axes, fig):

    # Join data into one DataFrame.
    replacement=["same root", "similar word", "no analogues", "false friend"]
    replacement2=["1", "2", "0", "!"]
    ddd2 = []
    for n, dat2 in enumerate(dats):
        for r1,r2 in zip(replacement, replacement2):
            dat=dat2[dat2['type']==r2].copy()
            dat['type']=dat['type'].replace(r2, nams[n]+" "+r1)
            ddd2.append(dat)
    ddd=pd.concat(ddd2)
    
    order=[n+" "+r1 for r1 in replacement for n in nams]

    axes[0].clear()
    axes[1].clear()

    sns.pointplot(x=ddd['type'], y=ddd['wo_par'], color="#FF0000", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, legend=None, ax=axes[0]) 
    offset = moveCollection(axes[0], 0, draw_shift, "right")
    moveLines(axes[0], offset, "right")
    moveLines(axes[0], offset, "right")
    sns.pointplot(x=ddd['type'], y=ddd['w_par'], palette="Dark2", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, legend=None, ax=axes[0]) 
    offset = moveCollection(axes[0], 1, draw_shift, "left")
    moveLines(axes[0], offset, "left")

    sns.swarmplot(x='type', y='w_par', data=ddd, order=order,# hue='lang',
                  palette="Set2", size=2, color=".3", linewidth=0, ax=axes[0], alpha=0.7)
    
    sns.pointplot(x=ddd['type'], y=ddd['wo_par'], palette="Dark2", markers="x", order=order, errwidth=1.5, capsize=0.2, join=False, legend=None, ax=axes[1]) 
    offset = moveCollection(axes[1], 0, draw_shift, "left")
    moveLines(axes[1], offset)

    sns.swarmplot(x='type', y='wo_par', data=ddd, order=order,# hue='lang',
                  palette="Set2", size=2, color=".3", linewidth=0, ax=axes[1], alpha=0.7)

    axes[0].set_xticklabels([i  for j in range(4) for i in range(1,7)])
    axes[1].set_xticklabels([i  for j in range(4) for i in range(1,7)])
    axes[0].set(xlabel="Тест с параллельным текстом", ylabel="", ylim=(0,1))
    axes[1].set(xlabel="Контрольный тест", ylabel="", ylim=(0,1))
    axes[0].yaxis.grid(True)
    axes[1].yaxis.grid(True)
    

    patch_colors = ['#0011DD', '#FF0044', '#00FFDD', '#DD88DD']
    for i in range(4):
        patches = [Rectangle((-0.5+i*6, -0.5), 6, 2, alpha = 0.05, edgecolor=patch_colors[i])]
        pc = PatchCollection(patches, alpha = 0.05, facecolor=patch_colors[i])
        pc2 = copy.copy(pc)
        axes[0].add_collection(pc)  
        axes[1].add_collection(pc2)  
    
    plt.subplots_adjust(bottom = 0.3, wspace = 0.1)
    axes[0].set_ylabel('Понятность слов')
    
    axes[0].text(0.1, 0.02, "Однокор.")
    axes[0].text(7.1, 0.02, "Сходн.")
    axes[0].text(13.1, 0.02, "Разн.")
    axes[0].text(18.1, 0.02, "Ложн. др.")
    axes[1].text(0.1, 0.02, "Однокор.")
    axes[1].text(7.1, 0.02, "Сходн.")
    axes[1].text(13.1, 0.02, "Разн.")
    axes[1].text(18.1, 0.02, "Ложн. др.")
    
    fig.savefig('img_res/Fig_4_intelligibility_on_cognate.png', dpi = 600)    

def showIntelligibility(qu_data, text_frame, axes, fig):
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    drawIntelligibility2(all_resn, all_tests, axes, fig)

markers = ['', 'o', 'v', '^', 'x', '+', '*']
edge_colors = ['#000000','#FF0000', '#00FF00', '#0000FF', '#FF8800', '#8800FF', '#00FF88']

def drawAnIntel(user_data, all_resn, intellig, sameness, test_no, test_len, axes=None, show_legend = None):
    if sameness == 'Same':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='1')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'Same and Similar':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='1') | (all_resn[test_no-1]["type"]=='2')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'No analogues':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='0')][['lang', 'w_par']].groupby('lang').count()['w_par']
    elif sameness == 'False Friends':
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='!')][['lang', 'w_par']].groupby('lang').count()['w_par']
    else:
        intellig['cnt'+str(test_no)] = all_resn[test_no-1][(all_resn[test_no-1]["type"]=='2')][['lang', 'w_par']].groupby('lang').count()['w_par']
    intellig['intel'+str(test_no)] = all_resn[test_no-1][['lang', 'w_par']].groupby('lang').mean()
    intellig['cnt'+str(test_no)] = intellig['cnt'+str(test_no)] / test_len
    mean = user_data[['parallel_lang', 'mean_mark']] \
                     [(user_data.test_id == test_no) & \
                      (user_data.parallel_lang == 'No Parallel Text')].mean().iloc[0]
    if axes != None:
        sns.scatterplot(x='cnt'+str(test_no), y='intel'+str(test_no), data=intellig, marker=markers[test_no],
                        hue=intellig.index, ax = axes, legend=show_legend, alpha=0.7)
        sns.lineplot([0, 1], [mean, mean], alpha = 0.5, linewidth = 1, ax = axes)
        axes.lines[-1].set_linestyle("--")
    return mean


def showIntel(sameness, user_data, qu_data, text_frame, axes, fig):
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    intellig = pd.DataFrame()
    
#    axes.clear()

    mean2 = drawAnIntel(user_data, all_resn, intellig, sameness, 2, 40, axes, 'brief')
    mean1 = drawAnIntel(user_data, all_resn, intellig, sameness, 1, 43, axes)
    mean3 = drawAnIntel(user_data, all_resn, intellig, sameness, 3, 35, axes)
    mean4 = drawAnIntel(user_data, all_resn, intellig, sameness, 4, 44, axes)
    mean5 = drawAnIntel(user_data, all_resn, intellig, sameness, 5, 37, axes)
    mean6 = drawAnIntel(user_data, all_resn, intellig, sameness, 6, 37, axes)
    intellig.loc["No Parallel Text"] = (None, mean1, None, mean2, None, mean3, None, mean4, None, mean5, None, mean6)
    intellig.columns = ['%Sim.Words, Test1', 'Avg.Results, Test1', 
                        '%Sim.Words, Test2', 'Avg.Results, Test2',
                        '%Sim.Words, Test3', 'Avg.Results, Test3',
                        '%Sim.Words, Test4', 'Avg.Results, Test4',
                        '%Sim.Words, Test5', 'Avg.Results, Test5',
                        '%Sim.Words, Test6', 'Avg.Results, Test6']
    sns.lineplot([0, 1], [0, 1], color = 'r', ax = axes)
    axes.set_ylabel('Понятность теста')

    langs2 = ['Язык пар. текста', 'Белорусский', 'Болгарский', 'Чешский', 
              'Польский', 'Сербский', 'Словацкий', 'Словенский', 'Украинский']
    axes.legend(loc="lower right")
    for i, l in enumerate(langs2):
        axes.get_legend().get_texts()[i].set_text(l)
    same2title = {"Same": "Однокоренные", "Similar":"Сходные", "No analogues": "Без аналогов", "False Friends":"Ложные друзья"}
    axes.set_title(same2title.get(sameness, "XXX"))
    axes.set_xlabel("Доля слов данного типа")
    display(intellig)
    
    # same2fig = {"Same": "_1", "Similar":"_2", "No analogues": "_3", "False Friends":"_4"}
    # fig.savefig('img_res/Fig_5'+same2fig.get(sameness, "_X")+'_correlation_on_type.png', dpi = 600)
    
    
    print("Correlation by tests")
    t1, t2 = [], []
    for i in range(6):
        t = intellig.iloc[0:-1, i*2:i*2+2].dropna()
        t1.extend(t.iloc[:, 0])
        t2.extend(t.iloc[:, 1])
        print(f"Test {i+1}", pearsonr(t.iloc[:,0], t.iloc[:,1])[0])
    print("All Tests", pearsonr(t1, t2)[0])
    print("\nCorrelation by language")
    for i in range(intellig.shape[0]-1):
        t = pd.DataFrame([[intellig.iloc[i, 2*j], intellig.iloc[i, 2*j+1]] 
                          for j in range(int(intellig.shape[1]/2))])
        t = t.dropna()
        print(intellig.index[i], pearsonr(t.iloc[:, 0].dropna(),
                                          t.iloc[:, 1].dropna())[0])

def show_intel_all(user_data, qu_data, text_frame):
    sameness = ['Same', 'Similar', 'No analogues', 'False Friends']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), num=4)
    for i, same in enumerate(sameness):
        print("\n----------\n", same, "\n----------")
        showIntel(same, user_data, qu_data, text_frame, axes[int(i/2)][i%2], fig)
    fig.savefig('img_res/Fig_5_all_correlation_on_type.png', dpi = 600)

def showWords2(test_n, coef, word_type, thr, qu_data, text_frame):
    all_resn, all_tests = processIntelligibility(qu_data, text_frame)
    test_n = int(test_n) - 1
    coef = {'<': 1, '>': -1}[coef]
    if word_type != 'All':
        word_type = {'Same':'1', 'Similar': '2', 'No analagues': '0', 'False friend': '!'}[word_type]
        dat = all_resn[test_n][(all_resn[test_n]["type"]==word_type) & (all_resn[test_n]["w_par"]*coef<thr*coef)].copy()
    else:
        dat = all_resn[test_n][(all_resn[test_n]["w_par"]*coef<thr*coef)].copy()
    dat["type"] = dat["type"].replace({'1': 'same', '2': 'similar', '0': 'no analogues', '!': 'false friend'})
    dat.columns = ["Russian Word", "Language", "Translation", "Word Class", "Control Group", "Parallel text", "Relation"]
    display(dat)
