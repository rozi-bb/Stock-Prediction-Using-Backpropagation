import sys
import time
import numpy as np
import pandas as pd 
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QTableWidgetItem
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi
from JST import *
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from os import path
import matplotlib.image as mpimg
from datetime import datetime
import math



class FormUtama(QMainWindow):


    #menciptakan objek dari kelas JST
    jst=JST();
        
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi('form_utama.ui',self)
        self.setWindowTitle('IMPLEMENTASI JST BACKPROPAGATION UNTUK PREDIKSI HARGA SAHAM PT SMARTFREN TELECOM TBK')
        img=mpimg.imread('logo.png')
        self.widgetLogo.canvas.axis1.imshow(img)
        self.widgetLogo.canvas.axis1.axis('off')
        self.btnBacaDataLatih.clicked.connect(self.BacaDataLatih)
        self.btnInisialisasiBobot.clicked.connect(self.InisialisasiBobot)
        self.btnProsesPelatihan.clicked.connect(self.ProsesPelatihan)
        self.btnBacaDataUji.clicked.connect(self.BacaDataUji)
        self.btnProsesPengujian.clicked.connect(self.ProsesPengujian)
        self.btnGrafikHasilPengujian.clicked.connect(self.GrafikHasilPengujian)
        self.btnBacaDataPrediksi.clicked.connect(self.BacaDataPrediksi)
        self.btnProsesPrediksi.clicked.connect(self.ProsesPrediksi)

    #mendefinisikan fungsi untuk membaca file data latih
    def BacaDataLatih(self):
        try:
            path=QFileDialog.getOpenFileName(self, 'Pilih file data latih', '',"Excel files (*.xls *.xlsx)")
            filename=path[0]
            data=pd.read_excel(filename,sheet_name="DATA_LATIH")
            data=data.to_numpy()

            #mengakeses kolom data saham pt smartfren telecom tbk
            tanggal=data[:,0]
            Open=data[:,1]
            High=data[:,2]
            Low=data[:,3]
            Close=data[:,4]
            Volume=data[:,5]
            Inflasi=data[:,6]
            Suku_Bunga=data[:,7]
            Kurs_Rupiah=data[:,8]
            
            

            #normalisasi data harga saham
            Inflasi_norm=self.jst.Normalisasi(Inflasi)
            Suku_Bunga_norm=self.jst.Normalisasi(Suku_Bunga)
            Kurs_Rupiah_norm=self.jst.Normalisasi(Kurs_Rupiah)
            Open_norm=self.jst.Normalisasi(Open)
            High_norm=self.jst.Normalisasi(High)
            Low_norm=self.jst.Normalisasi(Low)
            Close_norm=self.jst.Normalisasi(Close)
            Volume_norm=self.jst.Normalisasi(Volume)

                      
            #menentukan data latih
            n_data=len(Inflasi)
            data_latih=np.concatenate((Inflasi[:,None], Suku_Bunga[:,None], Kurs_Rupiah[:,None], Open[:,None], High[:,None], Low[:,None], Volume[:,None]), axis=1)
            data_latih_norm=np.concatenate((Inflasi_norm, Suku_Bunga_norm, Kurs_Rupiah_norm, Open_norm, High_norm, Low_norm, Volume_norm), axis=1)
            target_output=Close
            target_output_norm=Close_norm

            #menampilkan data pada tabel
            n_datalatih=len(tanggal)
            self.tblDataLatih.setRowCount(n_datalatih)
            for i in range(n_datalatih):
                tgl=datetime.strptime(str(tanggal[i]), '%Y-%m-%d %H:%M:%S')
                self.tblDataLatih.setItem(i,0,QTableWidgetItem(tgl.strftime("%d/%m/%Y")))
                self.tblDataLatih.setItem(i,1,QTableWidgetItem(str(Inflasi[i])))
                self.tblDataLatih.setItem(i,2,QTableWidgetItem(str(Suku_Bunga[i])))
                self.tblDataLatih.setItem(i,3,QTableWidgetItem(str(Kurs_Rupiah[i])))
                self.tblDataLatih.setItem(i,4,QTableWidgetItem(str(Open[i])))
                self.tblDataLatih.setItem(i,5,QTableWidgetItem(str(High[i])))
                self.tblDataLatih.setItem(i,6,QTableWidgetItem(str(Low[i])))
                self.tblDataLatih.setItem(i,7,QTableWidgetItem(str(Volume[i])))
                self.tblDataLatih.setItem(i,8,QTableWidgetItem(str(Close[i])))
                self.tblDataLatih.setItem(i,9,QTableWidgetItem(str(Inflasi_norm[i,0])))
                self.tblDataLatih.setItem(i,10,QTableWidgetItem(str(Suku_Bunga_norm[i,0])))
                self.tblDataLatih.setItem(i,11,QTableWidgetItem(str(Kurs_Rupiah_norm[i,0])))
                self.tblDataLatih.setItem(i,12,QTableWidgetItem(str(Open_norm[i,0])))
                self.tblDataLatih.setItem(i,13,QTableWidgetItem(str(High_norm[i,0])))
                self.tblDataLatih.setItem(i,14,QTableWidgetItem(str(Low_norm[i,0])))
                self.tblDataLatih.setItem(i,15,QTableWidgetItem(str(Volume_norm[i,0])))
                self.tblDataLatih.setItem(i,16,QTableWidgetItem(str(Close_norm[i,0])))
                

            #menampilkan jumlah data latih
            self.editJumlahDataLatih.setText(str(n_datalatih))
           
            #menyimpan data latih pada variabel global
            self.data_latih=data_latih
            self.data_latih_norm=data_latih_norm
            self.target_output=target_output
            self.target_output_norm=target_output_norm


            
        except:
            print('Terjadi kesalahan pada pembacaan data latih',sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(sys.exc_info()[2])


    #mendefinisikan fungsi untuk inisialisasi bobot
    def InisialisasiBobot(self):
        try:
            #mendapatkan inputan pada kotak tekx
            n_input=int(self.editNeuronInput.displayText())
            n_hidden=int(self.editNeuronHidden.displayText())
            n_output=int(self.editNeuronOutput.displayText())

            V=0
            W=0
            flag=False

            #mengecek apakah file bobot awal CSV sudah ada atau belum, apabila sudah ada maka bobot V dan W tidak akan dibangkitkan scr acak lagi, melainkan menggunakan bobot yang awal yang sudah ada
            if(path.exists('bobotawal_V.csv')==True and path.exists('bobotawal_W.csv')==True):
                #membaca bobot awal yang tersimpan pada file csv
                V=np.genfromtxt('bobotawal_V.csv',delimiter=',')  
                W=np.genfromtxt('bobotawal_W.csv',delimiter=',')
                baris,kolom=V.shape

                if(n_input==(baris-1) and n_hidden==kolom):
                    W_tmp=np.zeros((n_hidden+1,n_output))
                    W_tmp[:,0]=W
                    W=W_tmp
                    flag=True

            #Membangkitkan bobot V dan bobot W jika ukuran bobot awal yang terdapat dalam file CSV tidak sesuai dengan ukuran neuron yang dimasukkan
            if flag==False:
                [V,W]=self.jst.InisialisasiBobot(n_input,n_hidden,n_output)


            #menampilkan bobot V pada tabel
            row,col=V.shape
            self.tblBobotV.setColumnCount(col)
            self.tblBobotV.setRowCount(row)
            for i in range(row):
                for j in range(col):
                    self.tblBobotV.setItem(i,j,QTableWidgetItem(str(round(V[i,j],3))))

            #menampilkan bobot W pada Tabel
            row,col=W.shape
            self.tblBobotW.setColumnCount(col)
            self.tblBobotW.setRowCount(row)
            for i in range(row):
                for j in range(col):
                    self.tblBobotW.setItem(i,j,QTableWidgetItem(str(round(W[i,j],3))))

            #menyimpan bobot V dan bobot W pada variabel global
            self.V=V
            self.W=W

            #menyimpan bobot V dan B awal pada file csv
            #print('BOBOT V AWAL :')
            np.savetxt("bobotawal_V.csv", V, delimiter=",")
            #print('BOBOT W Awal :')
            np.savetxt("bobotawal_W.csv", W, delimiter=",")
            
        except:
            print('Terjadi kesalahan pada proses pembangkitan bobot awal',sys.exc_info()[0])


    #mendefinisikan fungsi untuk melakukan proses pelatihan
    def ProsesPelatihan(self):
        try:
            time_start=time.perf_counter()
            
            n_input=int(self.editNeuronInput.displayText())
            n_hidden=int(self.editNeuronHidden.displayText())
            n_output=int(self.editNeuronOutput.displayText())
            alpha=float(self.editAlpha.displayText())
            toleransi_eror=float(self.editToleransiEror.displayText())
            iterasi=int(self.editIterasi.displayText())

            data_latih=self.data_latih_norm
            target_output=self.target_output_norm

            V=self.V
            W=self.W

            print('Ukuran V :',V.shape)
            print('Ukuran W : ',W.shape)

            n_datalatih=len(data_latih)
            error=np.zeros((n_datalatih,1))
            errorsquare=np.zeros((n_datalatih,1))
            rmse=np.zeros((iterasi,1))
            jumlah_iterasi=0

            target_jst=np.zeros((n_datalatih,1))
            keluaran_jst=np.zeros((n_datalatih,1))
                        
            for i in range(iterasi):
                print('Iterasi ke-',(i+1))
                for j in range(n_datalatih):

                    [Z,Y]=self.jst.PerambatanMaju(data_latih[j,:],V,W,n_hidden,n_output)
                    [W,V]=self.jst.PerambatanMundur(target_output[j,:],Y,data_latih[j,:],alpha,Z,W,V)

                    error[j,0]=round(abs(target_output[j,0]-Y[0,0]),6)
                    errorsquare[j,0]=round(error[j,0]**2,6)
                    keluaran_jst[j,0]=Y[0,0]
        
                rmse[i,0]=round(math.sqrt(sum(errorsquare[:,0])/(n_datalatih-n_input)),6)
                print('RMSE : ',rmse[i,0])
    
                if rmse[i,0] <= toleransi_eror:
                    jumlah_iterasi=i+1
                    break
    
                jumlah_iterasi=i+1
            
            #menampilkan bobot V terbaru pada tabel
            row,col=V.shape
            self.tblBobotV.setColumnCount(col)
            self.tblBobotV.setRowCount(row)
            for i in range(row):
                for j in range(col):
                    self.tblBobotV.setItem(i,j,QTableWidgetItem(str(round(V[i,j],3))))

            #menampilkan bobot W terbaru pada Tabel
            row,col=W.shape
            self.tblBobotW.setColumnCount(col)
            self.tblBobotW.setRowCount(row)
            for i in range(row):
                for j in range(col):
                    self.tblBobotW.setItem(i,j,QTableWidgetItem(str(round(W[i,j],3))))

            #menyimpan bobot V dan W pada variabel global
            self.V=V
            self.W=W

            #menyimpan bobot terlatih pada file csv
            #print('BOBOT V TERLATIH :')
            np.savetxt("bobotterlatih_V.csv", V, delimiter=",")
            #print('BOBOT W TERLATIH :')
            np.savetxt("bobotterlatih_W.csv", W, delimiter=",")

            #menampilkan grafik konvergensi pada form
            self.widgetGrafikKonvergensi.canvas.axis1.clear()
            self.widgetGrafikKonvergensi.canvas.axis1.plot(rmse[0:jumlah_iterasi,0])
            self.widgetGrafikKonvergensi.canvas.axis1.set_title('Grafik Konvergensi Proses Pelatihan')
            self.widgetGrafikKonvergensi.canvas.axis1.set_ylabel('RMSE')
            self.widgetGrafikKonvergensi.canvas.axis1.set_xlabel('Iterasi')
            self.widgetGrafikKonvergensi.canvas.draw()

            #Menampilkan waktu komputasi dan RMSE pelatihan
            time_stop = (time.perf_counter() - time_start)
            self.editWaktuPelatihan.setText(str(round(time_stop,3)))
            self.editRMSEPelatihan.setText(str(rmse[jumlah_iterasi-1,0]))

            #menampilkan grafik konvergensi dalam figure secara terspisah
            plt.Figure()
            plt.subplot(111)
            plt.plot(rmse[0:jumlah_iterasi,0])
            plt.title('Grafik Konvergensi Proses Pelatihan')
            plt.ylabel('MSE')
            plt.xlabel('Iterasi')
            plt.grid()
            plt.show()

        except:
            print('Terjadi kesalahan pada proses pelatihan',sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(sys.exc_info()[2])


    #mendefinisikan fungsi untuk membaca file data uji
    def BacaDataUji(self):
        try:
            path=QFileDialog.getOpenFileName(self, 'Pilih file data uji', '',"Excel files (*.xls *.xlsx)")
            filename=path[0]
            data=pd.read_excel(filename,sheet_name="DATA_UJI")
            data=data.to_numpy()

            #mengakeses kolom data saham pt smartfren telecom tbk
            tanggal=data[:,0]
            Open=data[:,1]
            High=data[:,2]
            Low=data[:,3]
            Close=data[:,4]
            Volume=data[:,5]
            Inflasi=data[:,6]
            Suku_Bunga=data[:,7]
            Kurs_Rupiah=data[:,8]
            
        
            #normalisasi data harga saham
            Inflasi_norm=self.jst.Normalisasi(Inflasi)
            Suku_Bunga_norm=self.jst.Normalisasi(Suku_Bunga)
            Kurs_Rupiah_norm=self.jst.Normalisasi(Kurs_Rupiah)
            Open_norm=self.jst.Normalisasi(Open)
            High_norm=self.jst.Normalisasi(High)
            Low_norm=self.jst.Normalisasi(Low)
            Close_norm=self.jst.Normalisasi(Close)
            Volume_norm=self.jst.Normalisasi(Volume)

                      
            #menentukan data uji
            n_data=len(Inflasi)
            data_uji=np.concatenate((Inflasi[:,None], Suku_Bunga[:,None], Kurs_Rupiah[:,None], Open[:,None], High[:,None], Low[:,None], Volume[:,None]), axis=1)
            data_uji_norm=np.concatenate((Inflasi_norm, Suku_Bunga_norm, Kurs_Rupiah_norm, Open_norm, High_norm, Low_norm, Volume_norm), axis=1)
            output_sebenarnya=Close
            output_sebenarnya_norm=Close_norm

            #menampilkan data pada tabel
            n_datauji=len(data_uji)
            self.tblDataUji.setRowCount(n_datauji)
            for i in range(n_datauji):
                tgl=datetime.strptime(str(tanggal[i]), '%Y-%m-%d %H:%M:%S')
                self.tblDataUji.setItem(i,0,QTableWidgetItem(tgl.strftime("%d/%m/%Y")))
                self.tblDataUji.setItem(i,1,QTableWidgetItem(str(Inflasi[i])))
                self.tblDataUji.setItem(i,2,QTableWidgetItem(str(Suku_Bunga[i])))
                self.tblDataUji.setItem(i,3,QTableWidgetItem(str(Kurs_Rupiah[i])))
                self.tblDataUji.setItem(i,4,QTableWidgetItem(str(Open[i])))
                self.tblDataUji.setItem(i,5,QTableWidgetItem(str(High[i])))
                self.tblDataUji.setItem(i,6,QTableWidgetItem(str(Low[i])))
                self.tblDataUji.setItem(i,7,QTableWidgetItem(str(Volume[i])))
                self.tblDataUji.setItem(i,8,QTableWidgetItem(str(Close[i])))
                self.tblDataUji.setItem(i,9,QTableWidgetItem(str(Inflasi_norm[i,0])))
                self.tblDataUji.setItem(i,10,QTableWidgetItem(str(Suku_Bunga_norm[i,0])))
                self.tblDataUji.setItem(i,11,QTableWidgetItem(str(Kurs_Rupiah_norm[i,0])))
                self.tblDataUji.setItem(i,12,QTableWidgetItem(str(Open_norm[i,0])))
                self.tblDataUji.setItem(i,13,QTableWidgetItem(str(High_norm[i,0])))
                self.tblDataUji.setItem(i,14,QTableWidgetItem(str(Low_norm[i,0])))
                self.tblDataUji.setItem(i,15,QTableWidgetItem(str(Volume_norm[i,0])))
                self.tblDataUji.setItem(i,16,QTableWidgetItem(str(Close_norm[i,0])))
                

            #menampilkan jumlah data uji
            self.editJumlahDataUji.setText(str(n_datauji))
           
            #menyimpan data uji pada variabel global
            self.data_uji=data_uji
            self.data_uji_norm=data_uji_norm
            self.output_sebenarnya=output_sebenarnya
            self.output_sebenarnya_norm=output_sebenarnya_norm

        except:
            print('Terjadi kesalahan pada pembacaan data Uji',sys.exc_info()[0])


    #mendefinisikan fungsi untuk proses pengujian
    def ProsesPengujian(self):
        try:
            time_start=time.perf_counter()
            
            n_input=int(self.editNeuronInput.displayText())
            n_hidden=int(self.editNeuronHidden.displayText())
            n_output=int(self.editNeuronOutput.displayText())
            
            V=self.V
            W=self.W

           
            data_uji=self.data_uji_norm
            n_datauji=len(data_uji)
            hasil_prediksi=np.zeros((n_datauji,1))

            for j in range(n_datauji):
                [Z,Y]=self.jst.PerambatanMaju(data_uji[j,:],V,W,n_hidden,n_output)
                hasil_prediksi[j,0]=Y[0,0]

            #menghitung RMSE
            output_sebenarnya_norm=self.output_sebenarnya_norm
            total_eror=sum(abs(output_sebenarnya_norm-hasil_prediksi))
            MSE=total_eror[0]/len(output_sebenarnya_norm)
            RMSE=round(math.sqrt(MSE),5)
           
            
            #melakukan denormalisasi hasil prediksi dan keluaran sebenarnya
            output_sebenarnya=self.output_sebenarnya
            close_min=min(output_sebenarnya)
            close_max=max(output_sebenarnya)

            hasilprediksi_denorm=np.zeros((n_datauji,1))

            for i in range(n_datauji):
                hasilprediksi_denorm[i,0]=self.jst.Denormalisasi(hasil_prediksi[i,0],close_min,close_max)
           
            #menampilkan hasil prediksi pada tabel 
            rata2akurasi=0
            rata2eror=0
            self.tblHasilPengujian.setRowCount(n_datauji)
            for i in range(n_datauji):
                eror_=round(abs(hasilprediksi_denorm[i,0]-output_sebenarnya[i]),3)
                a=0

                if(hasilprediksi_denorm[i,0] > output_sebenarnya[i]):
                    a=hasilprediksi_denorm[i,0]
                else:
                    a=output_sebenarnya[i]


                if(a==0):
                    a=1

                rata2eror+=round((eror_/a*100),2)    
                akurasi=round(100-(eror_/a*100),2)
                rata2akurasi+=akurasi
                self.tblHasilPengujian.setItem(i,0,QTableWidgetItem(str(hasilprediksi_denorm[i,0])))
                self.tblHasilPengujian.setItem(i,1,QTableWidgetItem(str(output_sebenarnya[i])))
                self.tblHasilPengujian.setItem(i,2,QTableWidgetItem(str(eror_)))
                self.tblHasilPengujian.setItem(i,3,QTableWidgetItem(str(akurasi)))

            rata2eror=round(rata2eror/n_datauji,3)
            rata2akurasi=round(rata2akurasi/n_datauji,3)
            
            #menampilkan waktu komputasi, rata2 eror, dan rata2 akurasi
            time_stop = (time.perf_counter() - time_start)
            self.editWaktuPengujian.setText(str(round(time_stop,3)))
            self.editEror.setText(str(rata2eror))
            self.editAkurasi.setText(str(rata2akurasi))
            self.editRMSEPengujian.setText(str(RMSE))

            

            #menyimpan data pada variabel global
            self.hasilprediksi_denorm=hasilprediksi_denorm
            
            
        except:
            print('Terjadi kesalahan pada proses pengujian ',sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(sys.exc_info()[2])


    #mendefinisikan fungsi untuk menampilkan grafik hasil pengujian
    def GrafikHasilPengujian(self):
        try:
            hasilprediksi_denorm=self.hasilprediksi_denorm
            outputsebenarnya=self.output_sebenarnya
            
            y1=hasilprediksi_denorm
            y2=outputsebenarnya
            n_datauji=len(y1)
            x_tmp=list(range(1,n_datauji+1))
            x=np.array([x_tmp]).transpose()

            self.widgetGrafikHasilPengujian.canvas.axis1.clear()
            self.widgetGrafikHasilPengujian.canvas.axis1.plot(x,y1,color='blue')
            self.widgetGrafikHasilPengujian.canvas.axis1.plot(x,y2,color='red')
            self.widgetGrafikHasilPengujian.canvas.axis1.set_title('Grafik Perbandingan Antara Hasil Prediksi dan Data Sebenarnya ')
            self.widgetGrafikHasilPengujian.canvas.axis1.set_ylabel('Harga Saham')
            self.widgetGrafikHasilPengujian.canvas.axis1.set_xlabel('Data Uji')
            self.widgetGrafikHasilPengujian.canvas.axis1.legend(('Hasil Prediksi', 'Data Sebenarnya'),loc='upper right')
            self.widgetGrafikHasilPengujian.canvas.draw()

            #menampilkan grafik konvergensi dalam figure secara terspisah
            plt.figure()
            plt.subplot(111)
            plt.plot(x,y1,color='blue')
            plt.plot(x,y2,color='red')
            plt.title('Grafik Perbandingan Antara Hasil Prediksi dan Data Sebenarnya ')
            plt.ylabel('Harga Saham')
            plt.xlabel('Data Uji')
            plt.legend(('Hasil Prediksi', 'Data Sebenarnya'),loc='upper right')
            plt.grid()
            plt.show()
                        
        except:
            print('Terjadi Kesalahan pada saat menampilkan grafik hasil pengujian',sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(sys.exc_info()[2])


    #pendefinisian fungsi untuk melakukan pembacaan data yang akan diprediksi
    def BacaDataPrediksi(self):
        try:
            path=QFileDialog.getOpenFileName(self, 'Pilih file data uji', '',"Excel files (*.xls *.xlsx)")
            filename=path[0]
            data=pd.read_excel(filename,sheet_name="DATA_PREDIKSI")
            data=data.to_numpy()

            #mengakeses kolom data saham pt smartfren telecom tbk
            tanggal=data[:,0]
            Open=data[:,1]
            High=data[:,2]
            Low=data[:,3]
            Volume=data[:,4]
            Inflasi=data[:,5]
            Suku_Bunga=data[:,6]
            Kurs_Rupiah=data[:,7]
            
            #menentukan data prediksi
            n_data=len(Inflasi)
            data_prediksi=np.concatenate((Inflasi[:,None], Suku_Bunga[:,None], Kurs_Rupiah[:,None], Open[:,None], High[:,None], Low[:,None], Volume[:,None]), axis=1)             

         
            #menampilkan data pada tabel
            n_dataprediksi=len(Inflasi)
            self.tblDataPrediksi.setRowCount(n_dataprediksi)
            for i in range(n_dataprediksi):
                tgl=datetime.strptime(str(tanggal[i]), '%Y-%m-%d %H:%M:%S')
                self.tblDataPrediksi.setItem(i,0,QTableWidgetItem(tgl.strftime("%d/%m/%Y")))
                self.tblDataPrediksi.setItem(i,1,QTableWidgetItem(str(Inflasi[i])))
                self.tblDataPrediksi.setItem(i,2,QTableWidgetItem(str(Suku_Bunga[i])))
                self.tblDataPrediksi.setItem(i,3,QTableWidgetItem(str(Kurs_Rupiah[i])))
                self.tblDataPrediksi.setItem(i,4,QTableWidgetItem(str(Open[i])))
                self.tblDataPrediksi.setItem(i,5,QTableWidgetItem(str(High[i])))
                self.tblDataPrediksi.setItem(i,6,QTableWidgetItem(str(Low[i])))
                self.tblDataPrediksi.setItem(i,7,QTableWidgetItem(str(Volume[i])))
                

            #menampilkan jumlah data yang akan diprediksi
            self.editJumlahDataPrediksi.setText(str(n_dataprediksi))

            #menyimpan data pada variabel global
            self.data_prediksi=data_prediksi
        except:
            print('Terjadi Kesalahan pada saat menampilkan data sebelumnya',sys.exc_info()[0])
            

    #mendefinisikan fungsi untuk proses prediksi
    def ProsesPrediksi(self):
        try:
            n_input=int(self.editNeuronInput.displayText())
            n_hidden=int(self.editNeuronHidden.displayText())
            n_output=int(self.editNeuronOutput.displayText())
            n_dataprediksi=int(self.editJumlahDataPrediksi.displayText())
            
            V=self.V
            W=self.W

            #menggabungkan data uji dan data yang akan diprediksi untuk keperluan normalisasi karena data prediksi tidak bisa di normalisasi scr langsung, ia harus digabungkan dngn data uji terlebih dahulu
            data_uji=self.data_uji
            data_prediksi=self.data_prediksi
            data_full=np.concatenate((data_uji, data_prediksi), axis=0)

            #mengetahui ukuran data prediksi dan data uji
            m, n=data_prediksi.shape
            m1, n1=data_uji.shape

            #normalisasi data harga saham
            Inflasi_norm=self.jst.Normalisasi(data_full[:,0])
            Suku_Bunga_norm=self.jst.Normalisasi(data_full[:,1])
            Kurs_Rupiah_norm=self.jst.Normalisasi(data_full[:,2])
            Open_norm=self.jst.Normalisasi(data_full[:,3])
            High_norm=self.jst.Normalisasi(data_full[:,4])
            Low_norm=self.jst.Normalisasi(data_full[:,5])
            Volume_norm=self.jst.Normalisasi(data_full[:,6])


            data_prediksi_norm=np.concatenate((Inflasi_norm[m1:m1+m+1,:], Suku_Bunga_norm[m1:m1+m+1,:], Kurs_Rupiah_norm[m1:m1+m+1,:], Open_norm[m1:m1+m+1,:], High_norm[m1:m1+m+1,:], Low_norm[m1:m1+m+1,:], Volume_norm[m1:m1+m+1,:]), axis=1)
            hasil_prediksi=np.zeros((m,1))

            #melakukan proses prediksi sebanyak data yang diinginkan
            for j in range(m):                      
                [Z,Y]=self.jst.PerambatanMaju(data_prediksi_norm[j,:],V,W,n_hidden,n_output)
                hasil_prediksi[j,0]=Y[0,0]

            #melakukan denormalisasi hasil prediksi dan menampilkannya pada tabel
            close_min=min(self.output_sebenarnya)
            close_max=max(self.output_sebenarnya)

            hasilprediksi_denorm=np.zeros((m,1))
            self.tblHasilPrediksi.setRowCount(m)
            for i in range(m):
                hasilprediksi_denorm[i,0]=self.jst.Denormalisasi(hasil_prediksi[i,0],close_min,close_max)
                self.tblHasilPrediksi.setItem(i,0,QTableWidgetItem(str(hasil_prediksi[i,0])))
                self.tblHasilPrediksi.setItem(i,1,QTableWidgetItem(str(hasilprediksi_denorm[i,0])))



            #menampilkan grafik perbandingan hasil prediksi dan data sebenarnya pada form
            x=list(range(1,m+1))
            y=hasilprediksi_denorm[:,0]
            self.widgetGrafikHasilPrediksi.canvas.axis1.clear()
            self.widgetGrafikHasilPrediksi.canvas.axis1.plot(x,y,color='red',label='Data Hasil Prediksi')
            self.widgetGrafikHasilPrediksi.canvas.axis1.set_title('Grafik Hasil Prediksi')
            self.widgetGrafikHasilPrediksi.canvas.axis1.set_ylabel('Harga Saham')
            self.widgetGrafikHasilPrediksi.canvas.axis1.set_xlabel('Data ke-')
            self.widgetGrafikHasilPrediksi.canvas.axis1.legend(loc='upper right')
            self.widgetGrafikHasilPrediksi.canvas.draw()
            


            #menampilkan grafik pada figure
            plt.figure()
            plt.plot(x,y,'b')
            plt.xlabel('Data Ke')
            plt.ylabel('Harga Saham')
            plt.title('Grafik Hasil Prediksi')
            plt.show()

            
        except:
            print('Terjadi kesalahan pada proses prediksi ',sys.exc_info()[0])
            print(sys.exc_info()[1])
            print(sys.exc_info()[2])


        


if __name__=="__main__":
    app=QApplication(sys.argv)
    form=FormUtama()
    form.show()
    sys.exit(app.exec_())
