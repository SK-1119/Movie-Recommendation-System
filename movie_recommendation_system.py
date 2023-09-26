#=====================================================================================
#  Author: Kunal SK Sukhija
#=====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import missingno as msno
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,minmax_scale
from sklearn.model_selection import train_test_split

#%%
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
#%%
movies=movies.merge(credits,on='title')
#%%
mdata=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
#%%
r=mdata.head()['genres'][0]
#%%
import ast
#%%
# mdata['genres']=ast.literal_eval(mdata['genres'])
#%%
mdata.head()['genres'][0]
#%%
def findgenres(row):
    l=[]
    row=ast.literal_eval(row)
    for i in row:
        l.append(i['name'])
    return l
#%%
mdata['genres']=mdata['genres'].apply(findgenres)
#%%
def findkey(row):
    l=[]
    c=1
    row=ast.literal_eval(row)
    for i in row:
        if c<=4:
            l.append(i['name'])
        else:
            break
    return l
#%%
mdata['keywords']=mdata['keywords'].apply(findkey)
#%%
def findcast(row):
    l=[]
    c=1
    row=ast.literal_eval(row)
    for i in row:
        if c<=3:
            print(i['name'])
            l.append(i['name'].replace(' ',''))
            c+=1
        else:
            break
    return l
#%%
mdata['cast']=mdata['cast'].apply(findcast)
#%%
mdata['crew'][0]
#%%
def findcrew(row):
    row=ast.literal_eval(row)
    for i in row:
        if 'Director' in i.values():
            return i['name']
#%%
mdata['crew']=mdata['crew'].apply(findcrew)
#%%
mdata.dropna(inplace=True)
mdata.reset_index(drop=True)
#%%
mdata['overview']=mdata['overview'].apply(lambda x:x.split())
#%%
mdata.info()
#%%
# mdata['cast'].apply(lambda x:x.replace(" ",""))
mdata['crew']=mdata['crew'].apply(lambda x:x.replace(' ',''))
#%%
mdata['crew']=mdata['crew'].apply(lambda x:x.split())
#%%
mdata['tags']=mdata['overview']+mdata['genres']+mdata['cast']+mdata['crew']
#%%
main_movies=mdata.drop(['overview','genres','cast','crew','keywords'],axis=1)
#%%
main_movies['tags'][0]
#%%
from nltk.stem import PorterStemmer
#%%
word_stem=PorterStemmer()
#%%
word_stem.stem('action in')
#%%
def word_stemming(row):
    nl=[]
    for word in row:
        nl.append(word_stem.stem(word))
    return list(set(nl))
#%%
main_movies['tags']=main_movies['tags'].apply(word_stemming)
#%%
main_movies['tags']=main_movies['tags'].apply(lambda x:' '.join(x))
#%%
from sklearn.feature_extraction.text import CountVectorizer
#%%
cv=CountVectorizer(max_features=5500,stop_words='english')
#%%
vectors=cv.fit_transform(main_movies['tags']).toarray()
#%%
similarity=metrics.pairwise.cosine_similarity(vectors)
#%%
def recommend(movie):
    index=main_movies[main_movies['title']==movie].index[0]
    distances=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x:x[1])
    for i in distances[1:6]:
        print(main_movies.iloc[i[0]].title)
#%%
recommend('Teenage Mutant Ninja Turtles')