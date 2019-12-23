#Sentiments Analysis
#Using Random Forests
#1

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scikitplot as skplt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Dresses'])]

sentiments = []

for i in range(len(df)):
    if df.iloc[i][1] == 5 or df.iloc[i][1] == 4:
        sentiments.append('Positive')
    elif df.iloc[i][1] == 3:
        sentiments.append('Neutral')
    else:
        sentiments.append('Negative')

target = pd.DataFrame(sentiments)

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")

processed_reviews = preprocessingReviews(df)

vectorizer = TfidfVectorizer (max_features=5000, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
processed_reviews = pd.DataFrame(vectorizer.fit_transform(processed_reviews).toarray())

kf = KFold(n_splits=10)

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
confusionMatrix=[]
i=0
for train_index, test_index in kf.split(df):
    i=i+1
    print(train_index, " ", test_index)
    X_train, X_test = processed_reviews.iloc[train_index], processed_reviews.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = RandomForestClassifier(n_estimators=20, criterion= 'entropy', random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test, y_pred,average='micro')
    f1 = metrics.f1_score(y_test,y_pred,average='micro') 
    confMat =metrics.confusion_matrix(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    confusionMatrix.append(confMat)
    if i==10:
      print(metrics.classification_report(y_test,y_pred))

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]

print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print ("Time to Build Model: ",datetime.now() - startTime)

skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test),title='ROC Curves - Random Forest')

#2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scikitplot as skplt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Dresses'])]

sentiments = []

for i in range(len(df)):
    if df.iloc[i][1] == 5 or df.iloc[i][1] == 4:
        sentiments.append('Positive')
    elif df.iloc[i][1] == 3:
        sentiments.append('Neutral')
    else:
        sentiments.append('Negative')

target = pd.DataFrame(sentiments)

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")

processed_reviews = preprocessingReviews(df)

vectorizer = TfidfVectorizer (max_features=5000, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
processed_reviews = pd.DataFrame(vectorizer.fit_transform(processed_reviews).toarray())

kf = KFold(n_splits=10)

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
confusionMatrix=[]
i=0
for train_index, test_index in kf.split(df):
    i=i+1
    print(train_index, " ", test_index)
    X_train, X_test = processed_reviews.iloc[train_index], processed_reviews.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = RandomForestClassifier(n_estimators=40, criterion= 'gini', random_state=0,min_samples_leaf=10)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test, y_pred,average='micro')
    f1 = metrics.f1_score(y_test,y_pred,average='micro') 
    confMat =metrics.confusion_matrix(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    confusionMatrix.append(confMat)
    if i==10:
      print(metrics.classification_report(y_test,y_pred))

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]

print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print ("Time to Build Model: ",datetime.now() - startTime)

skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test),title='ROC Curves - Random Forest') 

#Using Neural Networks:
#1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import scikitplot as skplt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Dresses'])]

sentiments = []

for i in range(len(df)):
    if df.iloc[i][1] == 5 or df.iloc[i][1] == 4:
        sentiments.append('Positive')
    elif df.iloc[i][1] == 3:
        sentiments.append('Neutral')
    else:
        sentiments.append('Negative')

target = pd.DataFrame(sentiments)

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")

processed_reviews = preprocessingReviews(df)

vectorizer = TfidfVectorizer (max_features=5000, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
processed_reviews = pd.DataFrame(vectorizer.fit_transform(processed_reviews).toarray())

kf = KFold(n_splits=10)

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
confusionMatrix=[]
i=0
for train_index, test_index in kf.split(df):
    i=i+1
    print(train_index, " ", test_index)
    X_train, X_test = processed_reviews.iloc[train_index], processed_reviews.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', random_state=1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test, y_pred,average='micro')
    f1 = metrics.f1_score(y_test,y_pred,average='micro') 
    confMat =metrics.confusion_matrix(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    confusionMatrix.append(confMat)
    if i==10:
      print(metrics.classification_report(y_test,y_pred))

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]

print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print ("Time to Build Model: ",datetime.now() - startTime)

skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test),title='ROC Curves - Neural Network') 

#2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import scikitplot as skplt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Dresses'])]

sentiments = []

for i in range(len(df)):
    if df.iloc[i][1] == 5 or df.iloc[i][1] == 4:
        sentiments.append('Positive')
    elif df.iloc[i][1] == 3:
        sentiments.append('Neutral')
    else:
        sentiments.append('Negative')

target = pd.DataFrame(sentiments)

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")

processed_reviews = preprocessingReviews(df)

vectorizer = TfidfVectorizer (max_features=5000, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
processed_reviews = pd.DataFrame(vectorizer.fit_transform(processed_reviews).toarray())

kf = KFold(n_splits=10)

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
confusionMatrix=[]
i=0
for train_index, test_index in kf.split(df):
    i=i+1
    print(train_index, " ", test_index)
    X_train, X_test = processed_reviews.iloc[train_index], processed_reviews.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = MLPClassifier(hidden_layer_sizes=(10,5,2), activation='tanh', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', random_state=1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test, y_pred,average='micro')
    f1 = metrics.f1_score(y_test,y_pred,average='micro') 
    confMat =metrics.confusion_matrix(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    confusionMatrix.append(confMat)
    if i==10:
      print(metrics.classification_report(y_test,y_pred))

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]

print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print ("Time to Build Model: ",datetime.now() - startTime)

skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test),title='ROC Curves - Neural Network')

#Clustering
#Using KMeans with TFIDF
#1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Class Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Class Name"].isin(['Blouses','Skirts','Casual bottoms','Sweaters','Jackets'])]
df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,norm='l1',use_idf=True)
matrix = vec.fit_transform(processed_reviews)
df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

k=3

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
kmeans.fit(matrix)

y_method1 = kmeans.predict(matrix)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

s_score = silhouette_score(matrix, labels)
print(s_score)


order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print ("{}".format(terms[ind]))

print ("Time to Build Model: ",datetime.now() - startTime)

pca = TruncatedSVD(n_components = 2, random_state=1)
X_pca = pca.fit_transform(matrix)

center = pca.fit_transform(center)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)

#2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Class Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Class Name"].isin(['Blouses','Skirts','Casual bottoms','Sweaters','Jackets'])]
df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,norm='l1',use_idf=True)
matrix = vec.fit_transform(processed_reviews)
df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

k=3

kmeans = KMeans(n_clusters=k, init='random', max_iter=100, n_init=1,algorithm='elkan')
kmeans.fit(matrix)

y_method1 = kmeans.predict(matrix)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

s_score = silhouette_score(matrix, labels)
print(s_score)


order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print ("{}".format(terms[ind]))

print ("Time to Build Model: ",datetime.now() - startTime)

pca = TruncatedSVD(n_components = 2, random_state=1)
X_pca = pca.fit_transform(matrix)

center = pca.fit_transform(center)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)

#Using K-Means with GloVE
from gensim.models import Word2Vec
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
def textblob_tokenizer(str_input):
    tokenized_words=[]
    for i in range(len(str_input)):
      text = str_input.iloc[i]
      blob = TextBlob(text.lower())
      tokens = blob.words
      words = [token.stem() for token in tokens]
      tokenized_words.append(words)
    return tokenized_words
    
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass     
    return np.asarray(sent_vec) / numw

startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Class Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Class Name"].isin(['Blouses','Skirts','Casual bottoms','Sweaters','Jackets'])]

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

tokenized_words = textblob_tokenizer(processed_reviews)
glove_model = Word2Vec(tokenized_words,min_count=1)

X=[]
for sentence in tokenized_words:
    X.append(sent_vectorizer(sentence, glove_model)) 
X = pd.DataFrame(X)
X = X.fillna(0)

k=3

kmeans = KMeans(n_clusters=k, init='random', max_iter=100, n_init=1, algorithm='elkan')
kmeans.fit(X)

y_method1 = kmeans.predict(X)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

s_score = silhouette_score(X, labels)
print(s_score)

print ("Time to Build Model: ",datetime.now() - startTime)

pca = TruncatedSVD(n_components = 3)
X_pca = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:,2], c=y_method1)

#Anomaly Detection
#Using Gaussian Distribution
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from sklearn import metrics
from sklearn.metrics import silhouette_score
from textblob import TextBlob
import seaborn as sns
from datetime import datetime

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Bottoms'])]#

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,norm='l1',use_idf=True)
matrix = vec.fit_transform(processed_reviews)
df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

z = np.abs(stats.zscore(df))
print(z)

threshold = 3
print(np.where(z > 3))

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

sns.boxplot(x=IQR)

#Using Local Outlier Factor
#1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.metrics import silhouette_score
from textblob import TextBlob
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime   

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Bottoms'])]#

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,norm='l1',use_idf=True)
matrix = vec.fit_transform(processed_reviews)
df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

df = df.head(5000)
lof = LocalOutlierFactor(n_neighbors=10, algorithm='kd_tree',contamination=0.05)
y_pred = lof.fit_predict(df)
LOF_Scores = lof.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=df[LOF_pred==1]
print(LOF_anomalies)
print ("Time to Build Model: ",datetime.now() - startTime)

pca = TruncatedSVD(n_components = 3)
X_pca = pca.fit_transform(df)
lof_2d = pca.fit_transform(LOF_anomalies)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.array(['#377eb8', '#ff7f00'])

ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:,2], color=colors[(y_pred + 1) // 2],s=50)
ax.scatter(lof_2d[:,0],lof_2d[:,1],lof_2d[:,2],c='red',s=50)

#2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.metrics import silhouette_score
from textblob import TextBlob
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime   

def preprocessingReviews(df):
    text = df["Review Text"]
    print(text)
    stop=stopwords.words('english')
    stop.append("i'm")
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.astype(str)        
    return text
    
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words
    
startTime = datetime.now()
path = '/content/drive/My Drive/MS CS/KDDM/Project 5 - Text Mining/Womens_Clothing_E-Commerce_Reviews.csv'
fields = ["Review Text","Department Name","Rating"]
data=pd.read_csv(path,skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=data.loc[data["Department Name"].isin(['Tops','Bottoms'])]#

df = pd.DataFrame(df["Review Text"])
df = df.fillna("")
processed_reviews = preprocessingReviews(df)

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,norm='l1',use_idf=True)
matrix = vec.fit_transform(processed_reviews)
df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

df = df.head(5000)
lof = LocalOutlierFactor(n_neighbors=30, algorithm='auto',contamination=0.1)
y_pred = lof.fit_predict(df)
LOF_Scores = lof.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=df[LOF_pred==1]
print(LOF_anomalies)
print ("Time to Build Model: ",datetime.now() - startTime)

pca = TruncatedSVD(n_components = 3)
X_pca = pca.fit_transform(df)
lof_2d = pca.fit_transform(LOF_anomalies)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.array(['#377eb8', '#ff7f00'])

ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:,2], color=colors[(y_pred + 1) // 2],s=50)
ax.scatter(lof_2d[:,0],lof_2d[:,1],lof_2d[:,2],c='red',s=50)
