from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch

def lossFunc(args, DATA, gt_LABEL):

    D_train, D_fit, L_train, L_fit = train_test_split(DATA, gt_LABEL, test_size=0.8)
    scaler = StandardScaler()
    scaler.fit(D_train)

    D_train = scaler.transform(D_train)
    D_fit = scaler.transform(D_fit)
    clf = KNeighborsClassifier(n_neighbors=8)
    clf.fit(D_train, L_train)

    L_pred = clf.predict(D_fit)
    L_fit = L_fit.astype(float)
    L_pred = L_pred.astype(float)
    L_fit = torch.from_numpy(L_fit)
    L_pred = torch.tensor(L_pred, requires_grad=True)
    if gt_LABEL.max() == 1:
        return L_fit, L_pred
    else: 
        return L_fit, L_pred