import numpy as np

class getQuadran:
    def __init__(self, coorData) :
        self.dataA = coorData[:, 4]
        self.dataB = coorData[:, 5]
    
    def setQuadran(self):
        Quadran = np.empty((len(self.dataA), 6), dtype=object)

        for i in range(len(self.dataA)):
            X = np.int_(self.dataA[i])
            Y = np.int_(self.dataB[i])

            tetha = np.degrees(np.arctan2(Y,X)) + 360*(Y<0)
            magnitude = np.sqrt(np.power(X,2) + np.power(Y,2))
            quadranLabel = ''  

            if (X == 0) and (Y == 0):
                quadranLabel = 'No Quadran X Y = 0'
            else :
                if tetha >= 0 and tetha < 90:
                    quadranLabel = 'Q1'
                elif tetha >= 90 and tetha < 180:
                    quadranLabel = 'Q2'
                elif tetha >= 180 and tetha < 270 : 
                    quadranLabel = 'Q3'
                elif tetha >= 270 and tetha < 360:
                    quadranLabel = 'Q4'
                else :
                    quadranLabel = 'No Quadran'
            Quadran[i, :] = [ np.str_(i) ,X, Y, tetha, magnitude, quadranLabel]
        return Quadran