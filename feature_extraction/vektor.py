import numpy as np

class Vektor:
    def __init__(self, pocOutput, blockSize):
        self.poc = pocOutput[0]
        self.coorAwal = pocOutput[1] 
        self.blockSize = blockSize

    def getVektor(self):
        mb_x = self.blockSize  # panjang macroblock
        mb_y = self.blockSize  # lebar macroblock

        minimum_value = -self.blockSize/2
        maximum_value = self.blockSize/2

        cur_x = np.arange(0, mb_x)
        cur_y = np.arange(0, mb_y)

        nilTeng = np.int16(np.median(cur_x))
        medX = nilTeng + 1
        medY = nilTeng + 1

        rep_x = np.arange(-(nilTeng), medX)   
        rep_y = np.arange(nilTeng, -(medX), -1) 

        # output = np.empty((len(self.coorAwal), 6))
        output = np.zeros((len(self.coorAwal), 6))
        
        # print('len(self.coorAwal) : ', len(self.coorAwal))
        # print('Data Output Pertama (0) : ', output[0])

        valPOC = self.poc

        for i in range(valPOC.shape[2]):
            r = valPOC[:, :, i]
            
            val_max = np.max(np.max(r))
            temp_y, temp_x = np.where(r == np.max(r))
            
            if (len(temp_y) > 1 or len(temp_y) > 1) :
                temp_x = nilTeng
                temp_y = nilTeng
            else :
                temp_x = temp_x[0]
                temp_y = temp_y[0]
                
                if temp_x != nilTeng or temp_y !=nilTeng: 
                    corX = self.coorAwal[i][0]  # koordinat X mulai
                    corY = self.coorAwal[i][1]  # koordinat Y mulai
                
                    tX = corX-medX
                    tY = corY-medY

                    oX = rep_x[cur_x[temp_x]]
                    oY = rep_y[cur_y[temp_y]]

                    mX = (corX - (mb_x - temp_x))
                    mY = (corY - (mb_y - temp_y))
                    
                    p1 = [tX, tY]
                    p2 = [mX, mY]
                    V = np.array(p2) - np.array(p1)

                    # # Cek apakah nilai X berada di dalam rentang yang diinginkan
                    # if oX < minimum_value or oX > maximum_value:
                    #     oX = np.clip(oX, minimum_value, maximum_value).astype(int)
                    
                    # # Cek apakah nilai Y berada di dalam rentang yang diinginkan
                    # if oY < minimum_value or oY > maximum_value:
                    #     oY = np.clip(oY, minimum_value, maximum_value).astype(int)
                    
                    output[i, 0] = p1[0]
                    output[i, 1] = p1[1]
                    output[i, 2] = V[0]
                    output[i, 3] = V[1]
                    output[i, 4] = oX
                    output[i, 5] = oY
        return output
