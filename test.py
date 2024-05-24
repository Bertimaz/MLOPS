import backend as back
import json
import requests
import config
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if not config.dev:
    # URL de produção
    url = "http://18.226.169.27:5000/identify" 
    url_test = "http://18.226.169.27:5000/test" 
    file_path=os.path.join(current_dir, f"uploads/temp_file")
   
else:
    # Url de desenvolvimento
    url='http://127.0.0.1:5000/identify'
    url_test='http://127.0.0.1:5000/test'
    file_path=os.path.join(current_dir, f"uploads/temp_file")

# Testando Status do Servidor
print('Testando Status do Servidor')
response=requests.get(url_test)
json_data=response.json()
json_string=json.dumps(json_data)
print(json_data)
if json_data['status']==50:
    print('Servidor está online')
else:
    quit()

#Testando funcao de identificacao
print('Testando identificacao')
video_bytes=open(file_path,'rb')
response = requests.post(url, data=video_bytes, verify=False)
json_data=response.json()
if json_data['code']==50:
    print('Identificao ok')
print(json_data)