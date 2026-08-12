#python3 -m pip install rioxarray
#conda activate pyNOAAenv

import rioxarray as rxr
import pandas as pd
import requests
import tarfile
import os
import datetime
import shutil

#Remove tiff git folder to update its files
try:
  shutil.rmtree("./prediccion-aemet-harmonie-arome/tiff")
  os.mkdir("./prediccion-aemet-harmonie-arome/tiff")
except:
  print("/tiff exists")
#Download forecast file
response=requests.get('https://www.aemet.es/es/api-eltiempo/modelos/download/harmonie/PB')
response.raise_for_status() # ensure we notice bad responses
with open(os.getcwd()+"/harmonie.tar.gz", "wb") as file:
  file.write(response.content)

#Untar file
if("harmonie.tar.gz" in os.listdir()):
  tar = tarfile.open(os.getcwd()+'/harmonie.tar.gz') 
  tar.extractall(os.getcwd()+'/harmonie_data') 
  tar.close()

#Get file list
os.chdir(os.getcwd()+"/harmonie_data")
files=sorted(os.listdir())

#Get date from first file name
idate=files[0][5:24]
idate=datetime.datetime.strptime(idate, "%Y-%m-%dT%H:%M:%S")
#Convert to timestamp
its=idate.timestamp()

#Define Aemet Harmonie-Arome encoding
#[temperatura, viento, lluvia1h, lluvia3h, lluvia6h, nubosidad, rayos,rayos3h, rachas, rachas3h]
#d_code=["_11.tif","_32.tif","_61_1HH.tif","_61_3HH.tif","_61_6HH.tif","_71.tif","_207.tif","_207_3HH.tif","_228.tif","_228_3HH.tif"]
d_code=["_11.tif","_61_1HH.tif"]

for i,d in enumerate(d_code):
    #i is the index of the data, 0 for temperature 1 for rainfall
    #d is the coding defined in d_code
    #Now define the scale according to data type
    if(d=="_11.tif"):
        hescala={
        "R":[122,150,181,209,138,166,196,224,255,26,26,28,28,31,0,10,20,33,102,204,255,255,255,255,255,255,255,232,209,178,161,138],
        "G":[56,48,41,33,43,97,148,201,255,26,54,84,112,143,237,217,196,178,255,255,255,222,191,158,128,0,51,51,51,51,54,54],
        "B":[140,140,140,143,227,232,240,247,255,112,148,184,219,255,237,214,191,171,102,0,0,0,0,0,0,0,178,145,112,79,46,15],
        "v":[-27.5,-22.5,-17.5,-12.5,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,44]
        }
        hescala_df=pd.DataFrame(hescala)
    elif(d=="_32.tif"):
        hescala={
        "R":[216,175,102,80,168,217,255,255,255,255,255,153,153],
        "G":[250,249,246,249,244,242,201,138,105,89,120,102,51],
        "B":[248,246,236,183,1,0,11,23,31,89,212,255,204],
        "v":[5,15,25,35,45,55,65,75,85,95,105,115,120]
        }
        hescala_df=pd.DataFrame(hescala)
    elif(d=="_61_1HH.tif" or d=="_61_3HH.tif" or d=="_61_6HH.tif"):
        hescala={
        "R":[19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G":[49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B":[52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v":[0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275]
        }
        hescala_df=pd.DataFrame(hescala)
    elif(d=="_71.tif"):
        hescala={
        "R":[255,217,229,196,160,114,76,33,12],
        "G":[255,217,235,214,196,169,141,115,41],
        "B":[255,219,250,229,211,195,178,184,74],
        "v":[15,25,35,45,55,65,75,85,90]
        }
        hescala_df=pd.DataFrame(hescala)
    elif(d=="_207.tif" or d=="_207_3HH.tif"):
        hescala={
        "R":[53,47,41,35,63,147,233,232,229,224,191],
        "G":[151,231,247,244,241,238,235,144,48,0,0],
        "B":[253,250,180,90,30,24,19,14,9,147,61],
        "v":[0.0105,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.2]
        }
        hescala_df=pd.DataFrame(hescala)
    elif(d=="_228.tif" or d=="_228_3HH.tif"):
        hescala={
        "R":[217,176,102,79,168,217,255,255,255,255,255,153,153,171,190],
        "G":[248,250,247,250,245,242,201,138,105,89,120,102,51,26,0],
        "B":[248,247,237,184,0,0,10,23,31,89,212,255,204,133,61],
        "v":[5,15,25,35,45,55,65,75,85,95,105,115,125,135,140]
        }
        hescala_df=pd.DataFrame(hescala)
    for H in range(48):
      #Get data time
      d_time=idate+datetime.timedelta(hours=H)
      #Get data file
      d_file="down_"+d_time.strftime("%Y-%m-%dT%H:%M:%S")+"+00:00"+d
      #print("file "+d_file)
      
      if(d_file in files):
        print(d_file+" OK")
      else:
        print(d_file+" ERR")
        continue
      
      #Open tiff file
      hxr = rxr.open_rasterio(d_file)

      #Set max latlon (trim dataset)
      #SPAIN
      lat_min, lat_max = 34.5, 44.3
      lon_min, lon_max = -9.65,4.5

      #Trim data to coordinates
      hxr = hxr.sel(y=slice(lat_max, lat_min), x=slice(lon_min, lon_max))

      #Extract RGB bands = indices 012
      #datatype Int16 can contain negative values. Default is int8 from 0 to 255
      Rband=hxr[0].to_dataframe(name='R').astype('Int16')
      Gband=hxr[1].to_dataframe(name='G').astype('Int16')
      Bband=hxr[2].to_dataframe(name='B').astype('Int16')

      #Auxiliary columns for calculations (Total and New value)
      Tband=hxr[3].to_dataframe(name='T').astype('Int16')
      Nband=hxr[3].to_dataframe(name='N').astype('Int16')+999
      Vband=hxr[3].to_dataframe(name='v').astype('Float32')

      #As dataframe
      hxr_data={
        "Rband":Rband['R'],
        "Gband":Gband['G'],
        "Bband":Bband['B'],
        "Tband":Tband['T'],
        "Nband":Nband['N'],
        "Vband":Vband['v']
        }
      hxr_df=pd.DataFrame(hxr_data)

    #Loop to compare with each scale step (relative to scale length)
      for n in range(len(hescala['R'])):
        #Calculate total difference with scale steps
        hxr_df['Tband']=abs(hxr_df['Rband']-hescala['R'][n])+abs(hxr_df['Gband']-hescala['G'][n])+abs(hxr_df['Bband']-hescala['B'][n])
        #The minimum difference will be the most similar value
        for ind,val in enumerate(hxr_df['Nband']):
          #If the new value is lower than the current total, update the new value
          if(hxr_df['Tband'].values[ind] < hxr_df['Nband'].values[ind]):
            hxr_df['Nband'].values[ind]=hxr_df['Tband'].values[ind]
            #Use the value index to get the real value from the scale
            hxr_df['Vband'].values[ind]=hescala_df['v'][n]

      #Export to raster
      band = hxr.isel(band=3)
      for y in range(band.shape[0]-1):
          for x in range(band.shape[1]-1):
            band.values[y,x]=hxr_df['Vband'].values[x+(y*(band.shape[1]))]
      band.rio.to_raster("../prediccion-aemet-harmonie-arome/tiff/"+d_file+"f")

else:
    print("--¡Aemet HARMONIE-AROME Forecast succesfully processed!")
