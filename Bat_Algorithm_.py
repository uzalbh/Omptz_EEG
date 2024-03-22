#------------BAT Algorithm--------
from FS.ba import jfs
fold={'xt':X_train,'yt':y_train,'xv':X_test,'yv':y_test}
#Parameter
N=10 #number of solutions
T=100 #maximum number of itereation
fmax   = 2      # maximum frequency
fmin   = 0      # minimum frequency
alpha  = 0.9    # constant
gamma  = 0.9    # constant
A      = 2      # maximum loudness
r      = 1      # maximum pulse rate
opts = {'fold':fold, 'N':N, 'T':T, 'fmax':fmax, 'fmin':fmin, 'alpha':alpha, 'gamma':gamma, 'A':A, 'r':r}

#perform model selection
fmdl = jfs(feature_matrix_DWT, feature_matrix_label, opts)
sf   = fmdl['sf']

# Classification model with selected features by BAT
num_train = np.size(X_train, 0)
num_valid = np.size(X_test, 0)
x_train   = X_train[:, sf]
Y_train   = y_train.reshape(num_train)  # Solve bug
x_valid   = X_test[:, sf]
y_valid   = y_test.reshape(num_valid)  # Solve bug

mdl= GradientBoostingClassifier(random_state=0)
# acc_score(x_train, Y_train)
mdl.fit(x_train, Y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy DWT+EMD with BAT algorithm & Gradient Boosting:", 100 * Acc)


# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)





