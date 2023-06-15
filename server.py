from fastapi import FastAPI, HTTPException
import subprocess
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS settings
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://endotrainer.duckdns.org",
    "http://143.110.148.122",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"Endo Trainer API": "Welcome to Endo Trainer API"}

@app.get("/run_transferencia")
def run_transferencia(username: str, userKey : str):
    try:
        command = ["python", "Transferencia.py", username, userKey]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Transferencia.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Transferencia.py script: " + error.decode("utf-8"))

    return f"Transferencia.py script has been run. Output: {output.decode('utf-8')}"

@app.get("/run_sutura")
def run_sutura(username: str, userKey : str):
    try:
        command = ["python", "Sutura.py", username, userKey]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Sutura.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Sutura.py script: " + error.decode("utf-8"))

    return f"Sutura.py script has been run. Output: {output.decode('utf-8')}"

@app.get("/run_corte")
def run_corte(username: str, userKey : str):
    try:
        command = ["python", "Corte.py", username, userKey]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Corte.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Corte.py script: " + error.decode("utf-8"))

    return f"Corte.py script has been run. Output: {output.decode('utf-8')}"

@app.get("/run")
def run(username: str, userKey : str):
    time.sleep(3)
    return {"status_code":200, "username": username, "userKey": userKey}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8900)
     