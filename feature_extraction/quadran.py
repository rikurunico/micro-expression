import numpy as np
from helper.helper import format_number_and_round_numpy


class Quadran:
    def __init__(self, coorData):
        self.dataA = coorData[:, 4]
        self.dataB = coorData[:, 5]

    def getQuadran(self):
        quadranData = np.empty((len(self.dataA), 6), dtype=object)

        for i in range(len(self.dataA)):
            X = np.int_(self.dataA[i])
            Y = np.int_(self.dataB[i])

            # # Cek apakah nilai X berada di dalam rentang yang diinginkan
            # if X < minimum_value or X > maximum_value:
            #     X = np.clip(self.dataA[i], minimum_value, maximum_value).astype(int)

            # # Cek apakah nilai Y berada di dalam rentang yang diinginkan
            # if Y < minimum_value or Y > maximum_value:
            #     Y = np.clip(self.dataB[i], minimum_value, maximum_value).astype(int)

            # print('Data getQuadran ' , i, ' : ', self.dataA[i])
            # print('Data x ' , i, ' : ', X, ' Tipe', type(X))
            # # Y = np.int_(self.dataB[i])
            # print('Data getQuadran ' , i, ' : ', self.dataB[i])
            # print('Data y ' , i, ' : ', Y, ' Tipe', type(Y))

            tetha = np.degrees(np.arctan2(Y, X)) + 360 * (Y < 0)
            magnitude = np.sqrt(np.power(X, 2) + np.power(Y, 2))
            quadranLabel = ""

            if (X == 0) and (Y == 0):
                quadranLabel = "No Quadran X Y = 0"
            else:
                if tetha >= 0 and tetha < 90:
                    quadranLabel = "Q1"
                elif tetha >= 90 and tetha < 180:
                    quadranLabel = "Q2"
                elif tetha >= 180 and tetha < 270:
                    quadranLabel = "Q3"
                elif tetha >= 270 and tetha < 360:
                    quadranLabel = "Q4"
                else:
                    quadranLabel = "No Quadran"
            quadranData[i, :] = [
                np.str_(i),
                X, 
                Y, 
                format_number_and_round_numpy(tetha), 
                format_number_and_round_numpy(magnitude), 
                quadranLabel
            ]
        return quadranData
