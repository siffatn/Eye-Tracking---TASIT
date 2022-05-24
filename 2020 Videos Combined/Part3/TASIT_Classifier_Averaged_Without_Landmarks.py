from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report
import statistics as stats
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,r2_score, \
    recall_score, precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from sklearn.decomposition import PCA, KernelPCA

res = []
label=[]
flag=1

tasit = pd.read_csv('tasit_eye_part1.csv')
tasit = tasit.loc[0:130]

labels=tasit['subid']
#tasit = tasit.set_index("subid")
#tasit = tasit[['educyr', 'age','TraitAnxietyScaleTotal','Total PHQ9']]

df = pd.read_csv('Feature Vector Initial Exp.csv')
# df = df[(df["Part No."]==1)]
df.drop(["Part No."], axis=1, inplace=True)
df.drop(["Video No."], axis=1, inplace=True)
df.drop(["Total Length"], axis=1, inplace=True)
df.drop(["percentage of Fixation in Face"], axis=1, inplace=True)
df.drop(["percentage of Fixation in Eye"], axis=1, inplace=True)
df.drop(["percentage of Fixation in Mouth"], axis=1, inplace=True)
df.drop(["percentage of Fixation in Forehead"], axis=1, inplace=True)
df.drop(["Avg. Distance from Intersection of Eye and Nose"], axis=1, inplace=True)

df = df[df["Label As per Tasit Score"].notna()]
df = df.fillna(df.mean())
df = df.set_index("Participant No.")
df = df.drop(11012)
df = df.drop(12074)

label = df.pop("Label As per Tasit Score")
label[label<4] = 0
label[label==4] = 1
label = label.values

df.drop(["Label"], axis=1, inplace=True)

# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# label = le.fit_transform(label)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')


def outliers(df):
    z_scores = stats.zscore(df)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    return df
#print(X_test_imp.shape[1])

score = []

NUM_TRIALS = 5
scores  = np.zeros(NUM_TRIALS)
scores_recall = np.zeros(NUM_TRIALS)
scores_precision = np.zeros(NUM_TRIALS)
scores_f1 = np.zeros(NUM_TRIALS)

test_scores = np.zeros(NUM_TRIALS)
test_recall = np.zeros(NUM_TRIALS)
test_precision = np.zeros(NUM_TRIALS)
test_f1 = np.zeros(NUM_TRIALS)

random_state = [9, 99, 1099, 11900, 12099, 20099, 10499, 134099, 190304, 140569]

#dividing test and train data

X_train, X_test, label_train, label_test=train_test_split(df, label,stratify=label,
                                        test_size=0.20, random_state=500, shuffle=True)


for i in range(NUM_TRIALS):
    scores_in = []
    scores_in_recall = []
    scores_in_precision = []
    scores_in_f1 = []
    #df=quantile_transformer.fit_transform(df)
    skf = StratifiedKFold(n_splits=3, random_state=1023, shuffle=True)
    skf.get_n_splits(X_train, label_train)
    #print(skf)

    for train_index, test_index in skf.split(X_train, label_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        label_train2, label_val = label_train[train_index], label_train[test_index]

        #resampling minority (impaired class)

        # X_train2["label"] = label_train2
        # X_train_min = resample(X_train2[label_train2 == 0], replace=True,
        #                        n_samples=len(X_train2[label_train2 == 1])-len(X_train2[label_train2 == 0]),
        #                        random_state=134)
        #
        # X_train2 = pd.concat([X_train2, X_train_min])
        # label_train2 = X_train2.pop("label").values

        # outlier
        X_train2 = pd.DataFrame(data=X_train2)
        X_val = pd.DataFrame(data=X_val)
        ul = 0.99
        ll = 0.01
        lb = X_train2[label_train2==0].quantile(ll)
        ub = X_train2[label_train2==0].quantile(ul)
        X_train2[label_train2==0] = X_train2[label_train2 == 0][(X_train2[label_train2==0] < ub) & (X_train2[label_train2==0] > lb)]
        X_train2[label_train2==0] = X_train2[label_train2==0].fillna(X_train2[label_train2==0].mean())
        lb = X_train2[label_train2 == 1].quantile(ll)
        ub = X_train2[label_train2 == 1].quantile(ul)
        X_train2[label_train2 == 1] = X_train2[label_train2 == 1][(X_train2[label_train2 == 1] < ub) & (X_train2[label_train2==1] > lb)]
        X_train2[label_train2 == 1] = X_train2[label_train2 == 1].fillna(X_train2[label_train2 == 1].mean())

        quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
        X_train2 = quantile_transformer.fit_transform(X_train2)
        X_val = quantile_transformer.transform(X_val)


        ##selecting k best features
        bestfeatures = SelectKBest(score_func=f_classif, k=10)
        fit = bestfeatures.fit(X_train2, label_train2)
        # dfscores = pd.DataFrame(fit.scores_)
        # dfcolumns = pd.DataFrame(X_train2.columns)
        # bestfeatures = SelectKBest(score_func=f_classif, k=14)
        X_train2 = bestfeatures.fit_transform(X_train2,label_train2)
        X_val = bestfeatures.transform(X_val)
        # X_test2 = bestfeatures.transform(X_test2)

        old_stdout = sys.stdout
        log_file = open("message.log", "w")
        sys.stdout = log_file

        #parameter selection
        random_search = {'criterion': ['gini'],
                         'max_depth': [4],
                         'min_samples_split' : [3],
                         'min_samples_leaf': [2],
                         'max_features' : [4]}

        # random_search = {'criterion': ['gini', 'entropy'],
        #                  'max_features' : [ 'log2','sqrt'],
        #                  'min_impurity_decrease' : [0.1,0.2,0.3],
        #                  'min_samples_leaf' : [2,3,5,8,10],
        #                  'min_samples_split' : [2,4,6,8,10],
        #                  'max_depth' : [3,5,7,9,11,13]}
        #                  #'max_depth': list(np.linspace(10, 100, 10, dtype = int)) + [None],
        #                 # 'max_features': ['auto', 'sqrt','log2', None]}


        #cross validation
        cv = 3
        class_weight = {0: 1, 1: 1.5}

        # instanciate individual models
        lr = LogisticRegression(solver = 'liblinear', penalty= 'l2',class_weight=class_weight)
        # rf = RandomForestClassifier(n_estimators = 200, max_features="sqrt",
        #                             max_depth= 5, min_samples_leaf=3,
        #                             min_impurity_decrease=0.5,criterion='gini',
        #                             class_weight=class_weight, min_samples_split=7)
        rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                    random_state=random_state[i],class_weight=class_weight)
        #rf = GridSearchCV(rf, param_grid=random_search, cv = cv, verbose= 5)
        sv = svm.SVC(kernel='poly', probability=True, C=1, degree=1, class_weight='balanced')
        # sv = rf
        sys.stdout = old_stdout
        log_file.close()
        # vc = VotingClassifier([('clf2', rf), ("clf4", sv)],
        #                       weights=[.3, 1], voting='soft')
        # sv = vc
        prediction = cross_val_predict(sv.fit(X_train2,label_train2), X_val, label_val, cv = cv)
        ...
        # summarize result
        # print('Best Score: %s' % sv.best_score_)
        # print('Best Hyperparameters: %s' % sv.best_params_)
        print(classification_report(label_val, prediction))
        conf = confusion_matrix(label_val, prediction)
        print(confusion_matrix(label_val, prediction))
        print("CV Score of a VotingClassifier with GridSearchCV on cross validation data is {}".format(
            roc_auc_score(label_val, prediction)))

        print("CV Score of a VotingClassifier with GridSearchCV on train data is {}".format(roc_auc_score(label_val, prediction)))

        scores_in.append(roc_auc_score(label_val, prediction))
        scores_in_recall.append(recall_score(label_val, prediction, average='macro'))
        scores_in_precision.append(precision_score(label_val, prediction, average='macro'))
        scores_in_f1.append(f1_score(label_val, prediction, average='macro'))

    mean_in_scores = stats.mean(scores_in)
    scores[i] = mean_in_scores
    mean_in_scores = stats.mean(scores_in_recall)
    scores_recall[i] = mean_in_scores
    mean_in_scores = stats.mean(scores_in_precision)
    scores_precision[i] = mean_in_scores
    mean_in_scores = stats.mean(scores_in_f1)
    scores_f1[i] = mean_in_scores

    #scores = cross_val_score(grid, X_train2, label_train2, cv=cv, scoring='roc_auc')
    #print("CV Score of a VotingClassifier with GridSearchCV on train data is: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    prediction = sv.predict(X_val)
    print(classification_report(label_val, prediction))
    print(confusion_matrix(label_val,prediction))
    print(r2_score(label_val,prediction))
    print("Score of a VotingClassifier with GridSearchCV on val data is {}".format(roc_auc_score(label_val, prediction)))
    X_test2 = bestfeatures.transform(X_test)
    prediction = sv.predict(X_test2)
    print(classification_report(label_test, prediction))
    test_scores[i] = roc_auc_score(label_test, prediction)
    test_recall[i] = recall_score(label_test, prediction, average='macro')
    test_precision[i] = precision_score(label_test, prediction, average='macro')
    test_f1[i] = f1_score(label_test, prediction, average='macro')
    print("CV Score of a VotingClassifier with GridSearchCV on test data is {}".format(
        roc_auc_score(label_test, prediction)))

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