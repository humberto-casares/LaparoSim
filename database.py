import mysql.connector
from datetime import datetime
import random, json, string, os, requests
import numpy as np
from scipy import signal

class Database:
    def __init__(self):
        self.host="143.110.148.122"
        self.user="endotrainer"
        self.passwd="!2345678"
        self.database="sistemaweb"
        
    def run_query(self, query, flag):
        connection = None  # Initialize the connection variable
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                database=self.database
            )
            mycursor = connection.cursor()
            mycursor.execute(query)
            if flag == 0:
                res = mycursor.fetchall()
            elif flag == 1:
                res = 1
            return res
        except Exception as e:
            print("Insertion Error: ", e)
            return e
        finally:
            if connection is not None:
                connection.commit()
                connection.close()
            
    def getData(self, idUser):
        qr = "SELECT * FROM data WHERE userKey ='"+idUser+"';"
        print("Query: ", qr)
        res=self.run_query(qr, 0)
        
        return res  

    def addMaps(self, idUser, maps):
        # Constructing the query 
        qr = "UPDATE data SET `maps` = '" + maps + "' WHERE userKey = " + idUser + ";"
        res=self.run_query(qr, 1)
        return res
    
    def insertData(self, userKey, name, exercise, archive, img3d, maps, inference):
        # Constructing the query using f-string
        qr = f"INSERT INTO dataa (`userKey`, `name`, `create_date`, `exercise`, `file`, `img3d`, `maps`, `inference`) VALUES ('{userKey}', '{name}', '{self.now_datetime()}', '{exercise}', '{archive}', '{img3d}', '{maps}', '{inference}');"
        res=self.run_query(qr, 1)
        return res
    
    def addDataBase(self, fileName, userKey, name, exercise):
        maps=self.maps(fileName)
        # Convert the maps dictionary to a JSON string
        maps_json = json.dumps(maps)
        
        # Extract the name and extension using os.path.splitext()
        baseName, _ = os.path.splitext(os.path.basename(fileName))
        archive=baseName+".csv"
        img3d=baseName+".png"
        inference="Resultado de API Backend"

        try: 
            _=self.insertData(userKey, name, exercise, archive, img3d, maps_json, inference)
            return 1
        except Exception as e:
            return 0

    def now_datetime(self):
        now_datetime = datetime.now()
        return now_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # get random string of letters and digits
        src = string.ascii_letters + string.digits
        token = ''.join((random.choice(src) for i in range(14)))
        return token

    def maps(self, ruta):
        Data = np.genfromtxt(ruta, delimiter=',')

        # Assign extracted column values to variables
        xa, ya, za, xb, yb, zb = [Data[:-1, i] for i in [0,1,2,3,4,5]]    
        
        # Extracting last value of last column, which is Time
        tiempo_escalar = Data[:, -1][-1] if Data.size > 0 else None
        tiempo = Data[:, -1]

        # Converting cm to m
        xa, ya, za, xb, yb, zb = xa/100, ya/100, za/100, xb/100, yb/100, zb/100
            
        #EndoViS Path Length - Derecha, Izquierda
        PLD = np.sum(np.sqrt(np.diff(xa,1)**2 + np.diff(ya,1)**2 + np.diff(za,1)**2))
        PLI = np.sum(np.sqrt(np.diff(xb,1)**2 + np.diff(yb,1)**2 + np.diff(zb,1)**2))
        
        #EndoViS Depth Perception - Derecha, Izquierda
        DPD = np.sum(np.sqrt(np.diff(ya,1)**2 + np.diff(za,1)**2))
        DPI = np.sum(np.sqrt(np.diff(yb,1)**2 + np.diff(zb,1)**2))

        cte = (tiempo_escalar ** 5) / (2 * PLD ** 2)
        MS_prevd = np.sum((np.diff(xa, 3) ** 2) + (np.diff(ya, 3) ** 2) + (np.diff(za, 3) ** 2))
        MSD = np.sqrt(cte * MS_prevd)
        # Izquierda
        Xv,Yv,Zv=xb,yb,zb
        
        cte = (tiempo_escalar ** 5) / (2 * PLI ** 2)
        MS_previ = np.sum(np.diff(Xv, 3) ** 2 + np.diff(Yv, 3) ** 2 + np.diff(Zv, 3) ** 2)
        MSI = np.sqrt(cte * MS_previ)

        # Resampleo de la señal a cada segundo
        num = round(len(xa)/30)
        f = round(len(xa)/num)
        variables = [xa, ya, za, xb, yb, zb]
        windows = [3.2, 2.6, 0.5, 1.5, 0.2, 0.0]
        resampled = [signal.resample_poly(var, 1, f, window=('kaiser', w)) for var, w in zip(variables, windows)]
        xxa, yya, zza, xxb, yyb, zzb = resampled
        #Se convierten los datos en centimetros 
        xxa, yya, zza = xxa*1000, yya*1000, zza*1000
        xxb, yyb, zzb = xxb*1000, yyb*1000, zzb*1000

        #EndoViS Average Speed (mm/s) - Derecha, Izquierda
        SpeedD = np.sqrt(np.diff(xxa,1)**2 + np.diff(yya,1)**2 + np.diff(zza,1)**2)
        Mean_SpeedD = np.mean(SpeedD)
        SpeedI = np.sqrt(np.diff(xxb,1)**2 + np.diff(yyb,1)**2 + np.diff(zzb,1)**2)
        Mean_SpeedI = np.mean(SpeedI)

        #EndoViS Average Acceleration (mm/s^2) - Derecha, Izquierda
        Accd = np.sqrt(np.diff(xxa,2)**2 + np.diff(yya,2)**2 + np.diff(zza,2)**2)
        Mean_AccD = np.mean(Accd)
        Acci = np.sqrt(np.diff(xxb,2)**2 + np.diff(yyb,2)**2 + np.diff(zzb,2)**2)
        Mean_AccI = np.mean(Acci)

        #EndoViS Idle Time (%) - Derecha, Izquierda
        idle1D = np.argwhere(SpeedD<=5)
        idleD =(len(idle1D)/len(SpeedD))*100
        idle1I = np.argwhere(SpeedI<=5)
        idleI =(len(idle1I)/len(SpeedI))*100

        #EndoViS Max. Area (m^2) - Derecha, Izquierda
        max_horD = max(xa)-min(xa)
        max_vertD = max(ya)-min(ya)
        MaxAreaD = max_vertD*max_horD
        max_horI = max(xb)-min(xb)
        max_vertI = max(yb)-min(yb)
        MaxAreaI = max_vertI*max_horI

        #EndoViS Max. Volume (m^3) - Derecha, Izquierda
        max_altD = max(za)-min(za)
        MaxVolD = MaxAreaD*max_altD
        max_altI = max(zb)-min(zb)
        MaxVolI = MaxAreaI*max_altI

        #EndoViS Area/PL : EOA - Derecha, Izquierda
        A_PLD = np.sqrt(MaxAreaD)/PLD
        A_PLI = np.sqrt(MaxAreaI)/PLI

        #EndoViS Volume/PL: EOV - Derecha, Izquierda
        A_VD =  MaxVolD**(1/3)/PLD
        A_VI =  MaxVolI**(1/3)/PLI
        
        #EndoViS Bimanual Dexterity
        b= np.sum((SpeedI - Mean_SpeedI)*(SpeedD - Mean_SpeedD))
        d= np.sum(np.sqrt(((SpeedI - Mean_SpeedI)**2)*((SpeedD - Mean_SpeedD)**2)));   
        BD = b/d

        #EndoViS Energia - Derecha, Izquierda
        EXa = np.sum(xa**2)
        EYa = np.sum(ya**2)
        EZa = np.sum(za**2)
        EndoEAD = (EXa+EYa)/(MaxAreaD*100) #J/cm^2
        EndoEVD = (EXa+EYa+EZa)/(MaxVolD*100) #J/cm^3

        EXb = np.sum(xb**2)
        EYb = np.sum(yb**2)
        EZb = np.sum(zb**2)
        EndoEAI = (EXb+EYb)/(MaxAreaI*100) #J/cm^2
        EndoEVI = (EXb+EYb+EZb)/(MaxVolI*100) #J/cm^3

        parameters = {
        "Time (sec.)": tiempo_escalar,
        "Path Length (m.)": (PLD, PLI),
        "Depth Perception (m.)": (DPD, DPI),
        "Motion Smoothness (in m/s^3)": (MSD, MSI),
        "Average Speed (mm/s)": (Mean_SpeedD, Mean_SpeedI),
        "Average Acceleration (mm/s^2)": (Mean_AccD, Mean_AccI),
        "Idle Time (%)": (idleD, idleI),
        "Economy of Area (au.)": (A_PLD, A_PLI),
        "Economy of Volume (au.)": (A_VD, A_VI),
        "Bimanual Dexterity": BD,
        "Energy of Area (J/cm^2.)": (EndoEAD, EndoEAI),
        "Energy of Volume (J/cm^3.)": (EndoEVD, EndoEVI)
        }

        return parameters

    def upload_files_to_endpoint(self, server_url, csv_file_path, png_file_path):
        try:
            url = f"{server_url}/uploadFiles"
            files = {
                'csv_file': open(csv_file_path, 'rb'),
                'png_file': open(png_file_path, 'rb')
            }
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                print("Files uploaded successfully.")
            else:
                print("Failed to upload files. Status code:", response.status_code)
            
            for file in files.values():
                file.close()
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            
'''
obj = Database()

argument="Humberto Casares"
userKey="7"
file_name="Datos_Transferencia/Humberto Casares_2023-06-02_12-45-04.csv"
insertion = obj.addDataBase(file_name, userKey, argument, "1")

print(insertion)
'''
