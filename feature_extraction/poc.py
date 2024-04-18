import numpy as np 
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt 

class POC:

    def __init__(self, imgBlockCur, imgBlockRef, blockSize):
        self.imgBlockCur = imgBlockCur
        self.imgBlockRef = imgBlockRef
        self.blockSize = blockSize

    def hannCalc(self):
        window = np.hanning(self.blockSize)
        window = np.dot(window.T, window)

        return window

    def calcPOC(self, block_ref,block_curr ,window, mb_x, mb_y):
        # Menghitung transformasi Fourier dari blok citra saat ini dengan jendela Hanning
        fft_ref  = fft2(np.dot(block_ref, window), (mb_x, mb_y))
        fft_curr = fft2(np.dot(block_curr, window), (mb_x, mb_y))
        # Menghitung korelasi fase antara dua blok citra
        R1  = fft_ref  * np.conj(fft_curr) 
        # Menghitung magnitudo dari hasil korelasi fase
        R2  = abs(R1) 
        # Mengganti nilai-nol dalam R2 dengan nilai yang sangat kecil untuk menghindari pembagian oleh nol
        R2[R2 == 0] = 1e-31 
        # Menghitung korelasi fase normalisasi
        R   = R1/R2 
        # Menghitung invers transformasi Fourier dari hasil korelasi fase normalisasi
        r   = ifft2(R)
        # Menghitung magnitudo dari hasil invers transformasi Fourier
        r   = abs(r) 
        # Menggeser hasil invers transformasi Fourier agar titik nol berada di tengah
        r   = fftshift(r) 
        return r


    def getPOC(self):
        mb_x = self.blockSize  # panjang macroblock
        mb_y = self.blockSize  # lebar macroblock

        # Perhitunggan Hanning Window
        window = self.hannCalc()

        img0 = self.imgBlockCur
        img1 = self.imgBlockRef

        # konversi image float ke int
        cols, rows = img0.shape
        img0 = img0.astype(int)
        img1 = img1.astype(int)

        # menghitung berapa blok yang dihasilkan
        # dengan pembagian width atau height dan dibagi dengan blocksize
        colsY = np.int16(np.floor(cols / mb_y))
        rowsX = np.int16(np.floor(rows / mb_x))

        # inisiasi untuk menyimpan matrik image yang dipecah dalam blok
        BlocksCurr = np.empty((colsY, rowsX), dtype=object)
        BlocksRef = np.empty((colsY, rowsX), dtype=object)

        # untuk mengetahui sisa pixel
        modY = cols % mb_y
        modX = rows % mb_x

        # inisiasi untuk menyimpan nilai poc
        poc = np.zeros((mb_y, mb_x, colsY * rowsX))
        coorAwal = np.zeros((colsY * rowsX,2))
        rect = np.zeros((colsY * rowsX,4))

        nm = 0
        nY = 0
        nYY =1

        # perulangan y dan x dimulai dari 1
        # perulangan ini akan loncat sesuai dengan blocksize yang ditentukan
        # fungsi cols-modY atau rows-modX untuk pembatas, supaya area perulangan tidak melampaui ukuran gambar
        for y in range(0, cols - modY, mb_y):
            nX = 0
            nXX = 1
            for x in range(0, rows - modX, mb_x):

                # untuk menyimpan block array yang di crop sesuai ukuran
                BlocksCurr[nY, nX] = img0[y:y+mb_y, x: x+mb_x]
                BlocksRef[nY, nX]  = img1[y:y+mb_y, x: x+mb_x]

                rect[nm, :] = [x, y, mb_x, mb_y] #untuk membentuk kotak setiap blok

                block_ref = BlocksRef[nY, nX]
                block_curr = BlocksCurr[nY, nX]

                # Perhitungan POC 
                r = self.calcPOC(block_ref, block_curr, window, mb_x, mb_y)
                # menyimpan nilai poc sesuai nomor blok
                poc[:, :, nm] = r

                coorAwal[nm, 0] = nXX * mb_x  # koordinat X mulai
                coorAwal[nm, 1] = nYY * mb_y  # koordinat Y mulai
                nX  += 1
                nXX += 1
                nm  += 1
            nY  += 1
            nYY += 1
        # kembalian nilai
        # poc : untuk penyimpanan nilai poc disetiap blok
        # coorAwal : sebagai koordinat awal penanda batas blok
        # rect : untuk menyimpak penanda kotak x y width height
        return [poc, coorAwal, rect]
