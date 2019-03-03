from eszsl import *
if __name__=='__main__':
    class dataGenerator:
        def __init__(self,a=100,d=2):
            self.a = a
            self.d = d
            self.V = np.random.randn(self.a,self.d)
        def getAttributes(self,z = 3):
            return np.random.binomial(1,0.5,(z,self.a))        
        def getData(self,S, n = 50, useOldNormalization = False):
            z,a = S.shape
            assert a==self.a
            d = self.d
            C = np.dot(S,self.V)
            X = []
            Y = -1*np.ones((n*z,z))
            for i in range(z):
                X.append(np.random.randn(n,d)+C[i,:])
                Y[i*n:(i+1)*n,i]=1.0
            X = np.vstack(X)
            if not useOldNormalization:
                self.mean = np.mean(X,axis=0)
                self.std = np.std(X,axis=0)
            X = (X-self.mean)/self.std
            return X,Y
            
    def MCAccuracy(Y,Z):
        return np.mean(np.argmax(Z,axis=1)==np.argmax(Y,axis=1))
    
    a = 100
    d = 10
    dgen = dataGenerator(a,d)
    S = dgen.getAttributes(z=100)
    X,Y = dgen.getData(S, n = 50)
    clf = ESZSL(sigmap = 1e-1, lambdap = 1e4, kernel = poly, degree = 1)
    clf.fit(X,Y,S)
    Z = clf.decision_function(X,S)
    print("Train Accuracy",MCAccuracy(Y,Z))
    S2 = dgen.getAttributes(z=100)
    X2,Y2 = dgen.getData(S2, n = 50, useOldNormalization = True)
    Z2 = clf.decision_function(X2,S2)
    print("Test Accuracy",MCAccuracy(Y2,Z2))
