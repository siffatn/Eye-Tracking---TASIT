import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from pyts.classification import  TimeSeriesForest
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, make_scorer,accuracy_score, \
    precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, KernelPCA
import scipy.stats as stats
from sklearn.utils import resample
from tslearn.svm import TimeSeriesSVC
# from tslearn import clustering
#creating customized scoring for cross validation


def prec(y_true, y_pred): return precision_score(y_true, y_pred)
def recl(y_true, y_pred): return recall_score(y_true, y_pred)
def f1(y_true, y_pred): return f1_score(y_true, y_pred)
def acc(y_true, y_pred): return accuracy_score(y_true, y_pred)
scoring = {'precision': make_scorer(prec),
           'recall': make_scorer(recl),
           'f1': make_scorer(f1),
           'acc': make_scorer(acc)}

#
# scoring = {'precision': make_scorer(precision_score),
#            'recall': make_scorer(recall_score),
#            'f1': make_scorer(f1_score),
#            'acc': make_scorer(accuracy_score)}

#importing data

df = pd.read_csv('Part3 Video10 Vectorized R02.csv',header=None)

#replacing the nan values using mean value

# imp = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)
# df = df.reset_index(drop=True)
# df = df.fillna(df.mean())
# df = df.fillna(0)

#extracting participant id number

participant = df.pop(14014).values

#creating another list for tbi and hc marker

participant1 = list(participant)
participant1 = np.array(participant1)


for index,ii in enumerate(participant1):
    if ii//1000==11:
        participant1[index] = 1
    else:
        participant1[index] = 2

#class labels

label = df.pop(14013).values

#taking only the columns related to features excluding facial landmarks
# df1 = df.iloc[:,3000:4000] #3000:4000, consistent with different random seed
# df2 = df.iloc[:,5000:6000]
# df3 = df.iloc[:,9000:10000]
df1 = df.iloc[:,0:9000]
df = pd.concat([df1], axis=1)

# np.savetxt('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\part3video10_firstpaper_participant.csv',participant,delimiter=',')
# np.savetxt('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\part3video10_firstpaper_label.csv',label,delimiter=',')

#12,11, 10, 9, 8
#star 5, 3, 2
#function for outlier detection

def outliers(df):
    z_scores = stats.zscore(df)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    return df

#dividing test and train data
df2 = df
X_train2, X_test2, label_train2, label_test, participant_train, participant_test =train_test_split(df2, label, participant,test_size=0.20, random_state=500, shuffle=True)
# X_train, X_val, label_train, label_val = train_test_split(X_train2, label_train2, test_size=0.20, random_state=99,
#                                                           stratify=label_train2)
# np.savetxt('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\part3video10_firstpaper_participant_test.csv',participant_test,delimiter=',')
# np.savetxt('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\part3video10_firstpaper_label_test.csv',label_test,delimiter=',')
NUM_TRIALS = 5
scores = np.zeros(NUM_TRIALS)
test_scores = np.zeros(NUM_TRIALS)
scores_precision = np.zeros(NUM_TRIALS)
test_precision = np.zeros(NUM_TRIALS)
scores_recall = np.zeros(NUM_TRIALS)
test_recall = np.zeros(NUM_TRIALS)
scores_f1 = np.zeros(NUM_TRIALS)
test_f1 = np.zeros(NUM_TRIALS)
test_spec = np.zeros(NUM_TRIALS)
random_state = [9, 99, 1099, 11900, 12099, 20099, 10499, 134099, 190304, 140569]


for i in range(NUM_TRIALS):
    X_train, X_val, label_train, label_val = train_test_split(X_train2, label_train2, test_size=0.20,
                                    random_state=99, stratify=label_train2)
    #preprocessing using robust scaler function

    # scaler = preprocessing.RobustScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # df = scaler.transform(df)
    # X_train["label"] = label_train
    # X_train_min = resample(X_train[label_train == 0], replace=True,
    #                                n_samples=len(X_train[label_train == 0])-len(X_train[label_train == 1]),
    #                                random_state=134)
    #
    # X_train = pd.concat([X_train, X_train_min])
    # label_train = X_train.pop("label").values

    #doing principal component analysis on the train data
    # pca = KernelPCA(n_components=round(len(X_train)/1.5), kernel='poly', degree=1, coef0=0, alpha=1,
    #                         fit_inverse_transform=True,random_state=200,
    #                         eigen_solver='dense', remove_zero_eig=True)
    pca = PCA()
    print(pca.get_params())
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test2)
    df = pca.transform(df2)
    # np.savetxt('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\part3video10_firstpaper_test.csv', X_test, delimiter=',')

    # outlier (based on quantile values)

    X_train = pd.DataFrame(data=X_train)
    X_val = pd.DataFrame(data=X_val)
    X_test = pd.DataFrame(data=X_test)

    # ul = 0.99
    # ll = 0.01
    # lb = X_train[label_train==0].quantile(ll)
    # ub = X_train[label_train==0].quantile(ul)
    # X_train[label_train==0] = X_train[label_train == 0][(X_train[label_train==0] < ub) & (X_train[label_train==0] > lb)]
    # X_train[label_train==0] = X_train[label_train==0].fillna(X_train[label_train==0].mean())
    # lb = X_train[label_train == 1].quantile(ll)
    # ub = X_train[label_train == 1].quantile(ul)
    # X_train[label_train == 1] = X_train[label_train == 1][(X_train[label_train == 1] < ub) & (X_train[label_train==1] > lb)]
    # X_train[label_train == 1] = X_train[label_train == 1].fillna(X_train[label_train == 1].mean())

    #isolation forest

    from sklearn.ensemble import IsolationForest

    X_train["scores"] = np.zeros(shape=len(X_train))
    X_train["anomaly"] = np.zeros(shape=len(X_train))
    isf = IsolationForest(n_estimators=200, max_features=20, random_state=10)
    isf.fit(X_train[label_train == 0])
    X_train['scores'][label_train == 0] = isf.decision_function(X_train[label_train == 0])
    X_train['anomaly'][label_train == 0] = isf.predict(X_train[label_train == 0])

    isf.fit(X_train[label_train == 1])
    X_train['scores'][label_train == 1] = isf.decision_function(X_train[label_train == 1])
    X_train['anomaly'][label_train == 1] = isf.predict(X_train[label_train == 1])

    X_train, label_train = X_train[X_train['anomaly'] == 1], label_train[X_train['anomaly'] == 1]
    X_train.drop(["scores"], axis=1, inplace=True)
    X_train.drop(["anomaly"], axis=1, inplace=True)

    #parameter selection

    random_search = {'n_estimators': [300,400,500]}
    svm_search = {'degree': [1,2,3]}

    #setting cross validation parameters

    cv = 3
    count_correct = len(X_train[label_train == 1])
    count_incorrect = len(X_train[label_train == 0])
    class_weight = {0:count_correct/count_incorrect, 1: 1.5}
    class_weight1 = {0:count_correct/count_incorrect, 1: 3.5}
    random_state = [1,100,1000,1100,12000,20000,10400,134000,190304,140560]

    # instaniate individual models

    lr = LogisticRegression(solver = 'liblinear', penalty= 'l2',class_weight=class_weight)
    # rf = RandomForestClassifier(n_estimators=200, max_features=5, min_samples_leaf=10, max_depth=7,
    #                             criterion='entropy', class_weight=class_weight)
    rf = RandomForestClassifier(n_estimators=200, max_features=4, max_depth=5,
                                min_samples_split = 2, criterion='gini', class_weight=class_weight,
                                random_state=random_state[i])
    #     , criterion='entropy',random_state=random_state[i],
    #                       min_samples_split = 3, min_impurity_decrease=0.3, max_depth=5,
    # rf = TimeSeriesForest(min_window_size = 1, min_impurity_decrease=0.1, class_weight=class_weight)
    xgb = LinearDiscriminantAnalysis()
    mnb = KNeighborsClassifier(p=3, n_neighbors = 10, weights='uniform')
    #mnb = clustering.KShape(n_clusters=2)
    # sv = TimeSeriesSVC(kernel='poly', degree=2 ,gamma='auto', probability= True,
    #             class_weight=class_weight1)
    sv = svm.SVC(kernel='poly', degree=1, gamma='auto', coef0=1.0, C= 0.01, probability= True,
                 class_weight=class_weight1)

    #------------------------------------------------------------
    # create an ensemble for improved accuracy

    vc = VotingClassifier([('clf1', lr), ('clf2', rf), ("clf3", mnb), ("clf4", sv),("clf5", xgb)],
                          weights = [0,1,0,.3,0], voting = 'soft')
    # vc = sv
    #cross validating

    cv_results = cross_validate(vc.fit(X_train,label_train), X_val, label_val, scoring=scoring)
    print('manual scoring',cv_results)

    #prediction on train data
    prediction_train = vc.predict(X_train)
    print(classification_report(label_train, prediction_train), 'train data')
    print(f1(label_train, prediction_train), 'f1_train data', accuracy_score(label_train, prediction_train),'acc_train')


    #prediction on val data
    prediction_val = vc.predict(X_val)
    print(classification_report(label_val, prediction_val), 'val data')
    print(f1(label_val, prediction_val), 'f1_val data', accuracy_score(label_val, prediction_val),'acc_val')


    #prediction on test data
    prediction_test = vc.predict(X_test)
    print(classification_report(label_test, prediction_test), 'test data')
    print(f1(label_test, prediction_test), 'f1_test data', accuracy_score(label_test, prediction_test),'acc_test')

    #prediction on whole data

    prediction_whole = vc.predict(df)
    print(classification_report(label, prediction_whole), 'whole data')

    scores[i] = roc_auc_score(label_val, prediction_val, average='macro')
    scores_recall[i] = recall_score(label_val, prediction_val, average='macro')
    scores_precision[i] = precision_score(label_val, prediction_val, average='macro')
    scores_f1[i] = f1_score(label_val, prediction_val, average='macro')
    test_scores[i] = roc_auc_score(label_test, prediction_test, average='macro')
    test_recall[i] = recall_score(label_test, prediction_test, average='macro')
    test_precision[i] = precision_score(label_test, prediction_test, average='macro')
    test_f1[i] = f1_score(label_test, prediction_test, average='macro')
    tn, fp, fn, tp = confusion_matrix(label_test, prediction_test).ravel()
    specificity = tn / (tn + fp)
    test_spec[i] = specificity


print("Validation scores")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Recall : %0.2f (+/- %0.2f)" % (scores_recall.mean(), scores_recall.std()))
print("Precision : %0.2f (+/- %0.2f)" % (scores_precision.mean(), scores_precision.std()))
print("F1 : %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std()))
print("Test scores")
print("Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std()))
print("Recall : %0.2f (+/- %0.2f)" % (test_recall.mean(), test_recall.std()))
print("Precision : %0.2f (+/- %0.2f)" % (test_precision.mean(), test_precision.std()))
print("F1 : %0.2f (+/- %0.2f)" % (test_f1.mean(), test_f1.std()))
print("Specificity : %0.2f (+/- %0.2f)" % (test_spec.mean(), test_spec.std()))
    # #function for intersection of lists
    #
    # def Intersection(lst1, lst2):
    #     # Use of hybrid method
    #     temp = set(lst2)
    #     lst3 = [value for value in lst1 if value in temp]
    #     return lst3
    #
    # #finding the participant id no with impaired and not impaired
    #
    # HC_notimp = Intersection(participant[label==1],participant[participant1==2])
    # TBI_notimp = Intersection(participant[label==1],participant[participant1==1])
    # predicted_notimp_tbi = Intersection(participant[prediction_whole==1],participant[participant1==1])
    # predicted_notimp_hc = Intersection(participant[prediction_whole==1],participant[participant1==2])
    #
    # HC_imp = Intersection(participant[label==0],participant[participant1==2])
    # TBI_imp = Intersection(participant[label==0],participant[participant1==1])
    # predicted_imp_tbi = Intersection(participant[prediction_whole==0],participant[participant1==1])
    # predicted_imp_hc = Intersection(participant[prediction_whole==0],participant[participant1==2])
    #
    # print(HC_imp, TBI_imp, predicted_imp_hc, predicted_imp_tbi)

# save the model to disk

# pickle.dump(vc, open('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\model_part3_video10_salient_without', 'wb'))

    #for loading the model from disk

    #model = pickle.load(open('C:/Users\siffa\.PyCharmCE2018.2\config\scratches\model_part3_video13_salient', 'rb'))




