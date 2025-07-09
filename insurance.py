'''
import os
import requests
import random
import time
import cv2
import easyocr
from fastapi import FastAPI, Form
from fastapi.responses import Response
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from pymongo import MongoClient
from fpdf import FPDF
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure Groq-compatible OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Setup FastAPI
app = FastAPI()
sessions = {}

# Setup MongoDB
client_mongo = MongoClient("mongodb://localhost:27017/")
db = client_mongo["insurance"]
claims = db["car_claims"]

# Ensure image and PDF directories exist
os.makedirs("images", exist_ok=True)
os.makedirs("claim_forms", exist_ok=True)

reader = easyocr.Reader(['en'])

def groq_response(prompt, session_id):
    messages = sessions.get(session_id, [])
    messages.append({"role": "user", "content": prompt})

    res = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )

    ai_reply = res.choices[0].message
    messages.append({"role": ai_reply.role, "content": ai_reply.content})
    sessions[session_id] = messages
    return ai_reply.content

def generate_pdf(session_id):
    chat = sessions.get(session_id, [])
    filename = f"claim_forms/{session_id.replace(':', '')}_claim.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Car Accident Insurance Claim", ln=True, align='C')
    for msg in chat:
        prefix = "User:" if msg["role"] == "user" else "AI:"
        pdf.multi_cell(0, 10, f"{prefix} {msg['content']}")
    pdf.output(filename)
    return filename

def detect_license_plate(image_path):
    result = reader.readtext(image_path)
    for bbox, text, conf in result:
        if 6 <= len(text) <= 12 and any(char.isdigit() for char in text):
            return text.upper()
    return "Not Detected"

def submit_to_demo_provider(claim_id, license_plate, amount=5000):
    time.sleep(2)  # simulate delay
    return {
        "claim_id": claim_id,
        "status": "Approved",
        "payout_amount": amount,
        "transaction_id": f"TID{random.randint(100000,999999)}",
        "message": f"✅ Claim for vehicle {license_plate} approved. ₹{amount} will be credited."
    }

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None)):

    session_id = From
    response = MessagingResponse()

    if Body.lower() in ["status", "check claim", "claim status"]:
        doc = claims.find_one({"session_id": session_id})
        if doc:
            payout = doc.get("payout", {})
            status = doc["status"]
            license_plate = doc.get("license_plate", "Unknown")
            msg = f"📋 Status: {status}\n🚗 License Plate: {license_plate}"
            if payout:
                msg += f"\n💰 Payout: ₹{payout['payout_amount']} (Txn ID: {payout['transaction_id']})"
            response.message(msg)
        else:
            response.message("❌ No claim found for your number.")
        return Response(content=str(response), media_type="application/xml")

    reply = ""
    images = []
    license_plate = None

    if int(NumMedia) > 0:
        img_data = requests.get(MediaUrl0).content
        img_filename = f"images/{session_id.replace(':', '')}.jpg"
        with open(img_filename, "wb") as f:
            f.write(img_data)
        images.append(img_filename)
        license_plate = detect_license_plate(img_filename)
        reply += f"📸 Image received. Detected License Plate: {license_plate}\n"

    ai_reply = groq_response(Body, session_id)
    reply += ai_reply
    response.message(reply)

    if any(x in Body.lower() for x in ["submit", "done", "complete"]):
        pdf_path = generate_pdf(session_id)
        claim_id = session_id.replace(":", "")

        claim_doc = {
            "session_id": session_id,
            "phone": From,
            "chat_history": sessions[session_id],
            "images": images,
            "license_plate": license_plate or "Not Detected",
            "pdf": pdf_path,
            "status": "Submitted"
        }

        result = submit_to_demo_provider(claim_id, license_plate or "UNKNOWN")
        claim_doc["payout"] = result
        claim_doc["status"] = result["status"]

        claims.insert_one(claim_doc)
        response.message(result["message"])

    return Response(content=str(response), media_type="application/xml")
'''













'''

from fastapi import FastAPI, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4
import os, shutil, datetime, httpx
import uvicorn

# ----- CONFIG -----
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"
MONGO_URI = "mongodb://localhost:27017/"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- SETUP -----
app = FastAPI()
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["insurance_db"]
claims = db["claims"]
sessions = {}

# ----- HTML UI -----
def html_page(messages, session_id):
    chat_html = "".join([
        f"<p><b>{msg['role'].capitalize()}:</b> {msg['content']}</p>"
        for msg in messages
    ])
    return f"""
    <html><body>
        <h2>Insurance Claim AI Assistant</h2>
        <form action="/chat" method="post" enctype="multipart/form-data">
            <input type="hidden" name="session_id" value="{session_id}" />
            <textarea name="message" rows="2" cols="60" placeholder="Type here..."></textarea><br>
            <input type="file" name="file" /><br>
            <button type="submit">Send</button>
        </form>
        <div>{chat_html}</div>
    </body></html>
    """

# ----- GROQ CHAT -----
async def ask_groq(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# ----- ROUTES -----
@app.post("/start")
async def start():
    return HTMLResponse(html_page([], str(uuid4())))

@app.post("/chat")
async def chat(message: str = Form(...), file: UploadFile = None, session_id: str = Form(...)):
    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "data": {}}

    s = sessions[session_id]
    s["messages"].append({"role": "user", "content": message})

    if file:
        path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}_{file.filename}")
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        s["data"]["image"] = path
        s["data"]["license_plate"] = "MP04AB1234"  # fake detection

    messages = [{"role": "system", "content": "You are an AI helping users file accident claims."}] + s["messages"]
    reply = await ask_groq(messages)
    s["messages"].append({"role": "assistant", "content": reply})

    if "submit" in message.lower():
        status = "approved" if "license_plate" in s["data"] else "manual_review"
        s["data"].update({
            "status": status,
            "submitted_at": datetime.datetime.now(),
            "chat": s["messages"]
        })
        claims.insert_one(s["data"])
        return HTMLResponse(f"<html><body><h2>Claim {status.upper()}</h2><a href='/start'>Start New</a></body></html>")

    return HTMLResponse(html_page(s["messages"], session_id))

# ----- MAIN -----
if __name__ == "__main__":
    uvicorn.run("insurance:app", host="0.0.0.0", port=8000, reload=True)

'''















'''

import os
import uuid
import shutil
import pymongo
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# MongoDB setup (local)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["insurance"]
claims_collection = db["claims"]

# Groq setup
groq_api_key = os.getenv("GROQ_API_KEY")  # You must set this in your .env or replace directly
client = Groq(api_key=groq_api_key)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dir (optional)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

@app.get("/", response_class=HTMLResponse)
def home():
    session_id = str(uuid.uuid4())
    return f"""
    <html>
    <body>
        <h2>Insurance Claim AI Assistant</h2>
        <form action='/chat' method='post' enctype='multipart/form-data'>
            <input type='hidden' name='session_id' value='{session_id}' />
            <textarea name='message' rows='2' cols='60' placeholder='Type here...'></textarea><br>
            <input type='file' name='file' /><br>
            <button type='submit'>Send</button>
        </form>
        <div></div>
    </body>
    </html>
    """

@app.get("/start")
def start():
    return RedirectResponse("/")

@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(...), file: UploadFile = File(None)):
    file_info = ""

    # Save file if uploaded
    file_path = None
    if file and file.filename:
        extension = file.filename.split(".")[-1].lower()
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}.{extension}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_info = f"\n\nUser uploaded a file named: {file.filename}. Please include that in response if needed."

    # Ask Groq
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI insurance claim assistant. Help the user file their claim by asking for documents like police reports, medical bills, photos, etc. Then explain whether claim will proceed or go to manual review."},
                {"role": "user", "content": message + file_info},
            ],
            model="mixtral-8x7b-32768",
        )
        reply = completion.choices[0].message.content

        # Save to DB
        claims_collection.insert_one({
            "session_id": session_id,
            "message": message,
            "file": file.filename if file else None,
            "reply": reply
        })

        return f"""
        <html><body>
        <h2>Insurance Claim AI Assistant</h2>
        <form action='/chat' method='post' enctype='multipart/form-data'>
            <input type='hidden' name='session_id' value='{session_id}' />
            <textarea name='message' rows='2' cols='60' placeholder='Type here...'></textarea><br>
            <input type='file' name='file' /><br>
            <button type='submit'>Send</button>
        </form>
        <hr>
        <strong>You said:</strong> {message}<br>
        <strong>Assistant:</strong><br><pre>{reply}</pre>
        </body></html>
        """
    except Exception as e:
        return f"<html><body><h2>Error:</h2><pre>{str(e)}</pre></body></html>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("insurance:app", host="0.0.0.0", port=8000, reload=True)

'''















#it works 
'''

import uuid
import base64
import shutil
import os
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from groq import Groq
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

# Allow CORS for Postman access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client.insurance
claims = db.claims

# Mount static directory
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Groq client setup (replace YOUR_GROQ_API_KEY)
groq_client = Groq(api_key="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")

# Store user session messages
sessions = {}

@app.post("/start")
async def start():
    session_id = str(uuid.uuid4())
    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)

@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_page(session_id: str):
    return f"""
    <html>
    <head><title>Insurance Claim AI Assistant</title></head>
    <body>
        <h2>Insurance Claim AI Assistant</h2>
        <form action="/chat" method="post" enctype="multipart/form-data">
            <input type="hidden" name="session_id" value="{session_id}" />
            <textarea name="message" rows="2" cols="60" placeholder="Type here..."></textarea><br>
            <input type="file" name="file" /><br>
            <button type="submit">Send</button>
        </form>
        <div id="chatbox">
            {"".join(f"<p><b>You:</b> {m['user']}<br><b>AI:</b> {m['ai']}</p>" for m in sessions.get(session_id, []))}
        </div>
    </body>
    </html>
    """

@app.post("/chat", response_class=HTMLResponse)
async def chat_submit(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(""),
    file: UploadFile = File(None)
):
    if session_id not in sessions:
        sessions[session_id] = []

    file_info = ""
    if file:
        contents = await file.read()
        image_path = f"static/{uuid.uuid4()}.jpg"
        with open(image_path, "wb") as f:
            f.write(contents)
        file_info = f"[File received: {file.filename}]"

        # License plate detection
        try:
            img_array = np.asarray(bytearray(contents), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
            found = plates.detectMultiScale(gray, 1.1, 4)
            if len(found) > 0:
                file_info += " — License plate detected ✅"
            else:
                file_info += " — No license plate detected 🚫"
        except Exception as e:
            file_info += f" — [Error in image processing: {e}]"

    # AI response from Groq
    try:
        history = sessions[session_id]
        prompt = f"You are an insurance claims assistant. {message} {file_info}"
        chat_response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You help users file insurance claims after accidents."},
                *[{"role": "user", "content": h["user"]} for h in history],
                {"role": "user", "content": prompt},
            ]
        )
        reply = chat_response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"[Error getting AI response: {str(e)}]"

    # Save to MongoDB
    claims.insert_one({
        "session_id": session_id,
        "user_message": message,
        "ai_reply": reply,
        "file_uploaded": file.filename if file else None
    })

    sessions[session_id].append({
        "user": message,
        "ai": reply
    })

    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("insurance:app", host="0.0.0.0", port=8000)

'''















#IT WORKS WELL
'''

import uuid
import pytesseract
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import os
import shutil
from pyngrok import ngrok
import requests
import uvicorn

# === CONFIG ===
groq_api_key = "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"
groq_model = "llama3-8b-8192"
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["insurance"]
claims = db["claims"]
users = db["users"]
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === FASTAPI APP ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# === Start ngrok ===
port = 8000
public_url = ngrok.connect(port).public_url
print(f"* Ngrok tunnel running at: {public_url}")

# === Session storage ===
sessions = {}

# === HTML Template ===
html_template = """
<html>
<head><title>Insurance Claim AI Assistant</title></head>
<body>
    <h2>Insurance Claim AI Assistant</h2>
    <form action="/chat" method="post" enctype="multipart/form-data">
        <input type="hidden" name="session_id" value="{session_id}" />
        <textarea name="message" rows="2" cols="60" placeholder="Type here..."></textarea><br>
        <input type="file" name="file" /><br>
        <button type="submit">Send</button>
    </form>
    <div id="chatbox">{chat}</div>
</body>
</html>
"""

# === AI CHAT (GROQ) ===
def chat_with_groq(message, history):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": groq_model,
                "messages": history + [{"role": "user", "content": message}]
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error getting AI response: {str(e)}]"

# === License Plate Detection ===
def detect_license_plate(file_path):
    try:
        image = Image.open(file_path).convert("L")  # convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # upscale image for better OCR
        text = pytesseract.image_to_string(image, config='--psm 6')  # use Page Segmentation Mode 6
        print("OCR Result:\n", text)  # for debugging
        lines = text.splitlines()
        plates = [line.strip().replace(" ", "").replace("-", "") for line in lines if any(char.isdigit() for char in line)]
        return plates[0] if plates else None
    except Exception as e:
        print("OCR Error:", str(e))
        return None

# === START SESSION ===
@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chat": [], "plate": None}
    return {"url": f"{public_url}/claim/{session_id}"}

# === LOAD CHAT UI ===
@app.get("/claim/{session_id}", response_class=HTMLResponse)
def chat_page(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h2>Invalid session</h2>", status_code=404)
    chat_html = "<br>".join(sessions[session_id]["chat"])
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

# === CHAT POST ===
@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(""), file: UploadFile = File(None)):
    if session_id not in sessions:
        return HTMLResponse("<h2>Invalid session</h2>", status_code=404)

    chat_log = sessions[session_id]["chat"]
    history = []

    # Load previous conversation into history
    for line in chat_log:
        if line.startswith("You:"):
            history.append({"role": "user", "content": line[4:].strip()})
        elif line.startswith("AI:"):
            history.append({"role": "assistant", "content": line[3:].strip()})

    # Add user message
    if message:
        chat_log.append(f"You: {message}")

    # Handle file upload
    if file:
        file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chat_log.append(f"You uploaded: {file.filename}")

        # Try detecting license plate
        plate = detect_license_plate(file_path)
        if plate:
            sessions[session_id]["plate"] = plate
            user = users.find_one({"plate": plate})
            if user:
                user_info = f"License Plate: {plate}\nName: {user['name']}\nPolicy #: {user['policy_number']}\nCompany: {user['insurance_company']}"
                chat_log.append(f"AI: Found user data:\n{user_info}")
                history.append({"role": "assistant", "content": f"User data:\n{user_info}"})
            else:
                chat_log.append("AI: License plate detected, but no user data found.")
                history.append({"role": "assistant", "content": "License plate detected, but no user data found."})
        else:
            chat_log.append("AI: No license plate detected. Please provide more accident details.")
            history.append({"role": "assistant", "content": "No license plate detected. Please provide more accident details."})

    # Get AI response
    if message:
        ai_response = chat_with_groq(message, history)
        chat_log.append(f"AI: {ai_response}")

    return HTMLResponse(content=html_template.format(session_id=session_id, chat="<br>".join(chat_log)))

# === START SERVER ===
if __name__ == "__main__":
    uvicorn.run("insurance:app", host="0.0.0.0", port=8000)

'''









#work well and good
'''


import uuid, shutil, os
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from pymongo import MongoClient
from PIL import Image
import pytesseract
import requests

# ==== CONFIG ====
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # UPDATE this path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==== MONGODB ====
mongo = MongoClient("mongodb://localhost:27017")
db = mongo["insurance_db"]
users = db["users"]


# ==== FASTAPI ====
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# ==== SESSION STORE ====
sessions = {}

# ==== NGROK ====
public_url = ngrok.connect(8000).public_url
print(f"🚀 Ngrok tunnel running at: {public_url}")

# ==== CHAT UI HTML ====
html_template = """
<html><head>
<title>Insurance AI Assistant</title>
<style>
body {{ font-family: Arial, sans-serif; background: #f9f9f9; padding: 20px; }}
h2 {{ color: #2c3e50; }}
form {{ margin-top: 20px; }}
#chatbox {{ margin-top: 20px; background: #fff; padding: 15px; border-radius: 8px; max-width: 600px; }}
.msg {{ margin-bottom: 10px; }}
.you {{ color: #2980b9; }}
.ai {{ color: #27ae60; }}
</style>
</head>
<body>
    <h2>Insurance Claim AI Assistant</h2>
    <form action="/chat" method="post" enctype="multipart/form-data">
        <input type="hidden" name="session_id" value="{session_id}" />
        <textarea name="message" rows="3" cols="70" placeholder="Describe your accident or upload image..."></textarea><br><br>
        <input type="file" name="file" /><br><br>
        <button type="submit">Send</button>
    </form>
    <div id="chatbox">
        {chat}
    </div>
</body>
</html>
"""

# ==== LICENSE PLATE DETECTION ====
def detect_license_plate(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config='--psm 6')
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= 6 and any(c.isdigit() for c in line):
                return line.upper().replace(" ", "")
    except:
        pass
    return None

# ==== GROQ CHAT ====
def chat_with_groq(message, history):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": history + [{"role": "user", "content": message}]}
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"

# ==== ROUTES ====

@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chat": [], "plate": None}
    return {"url": f"{public_url}/claim/{session_id}"}

@app.get("/claim/{session_id}", response_class=HTMLResponse)
def claim_page(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]
    chat_html = "".join([f'<div class="msg {"you" if m.startswith("You:") else "ai"}">{m}</div>' for m in chat_log])
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(""), file: UploadFile = File(None)):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)

    chat_log = sessions[session_id]["chat"]
    history = [{"role": "user" if m.startswith("You:") else "assistant", "content": m.split(":", 1)[1].strip()} for m in chat_log if ":" in m]

    if message:
        chat_log.append(f"You: {message}")

        # Check if message is a license plate
        if len(message) >= 6 and any(c.isdigit() for c in message):
            plate = message.upper().replace(" ", "")
            sessions[session_id]["plate"] = plate
            user = users.find_one({"plate": plate})
            if user:
                summary = f"Name: {user['name']}, Policy: {user['policy_number']}, Insurer: {user['insurance_company']}, Vehicle: {user['vehicle']['make']} {user['vehicle']['model']}, Color: {user['vehicle']['color']}"
                reply = f"Found user record for plate {plate}: {summary}"
                chat_log.append(f"AI: {reply}")
                history.append({"role": "assistant", "content": reply})

    # Handle image
    if file:
        path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chat_log.append(f"You uploaded image: {file.filename}")

        plate = detect_license_plate(path)
        if plate:
            sessions[session_id]["plate"] = plate
            user = users.find_one({"plate": plate})
            if user:
                info = f"Detected plate: {plate}\nName: {user['name']}, Policy: {user['policy_number']}, Insurer: {user['insurance_company']}"
                chat_log.append(f"AI: {info}")
                history.append({"role": "assistant", "content": info})
            else:
                chat_log.append("AI: Plate detected but no matching user found.")
        else:
            chat_log.append("AI: No plate detected in the image. Please provide more details.")

    if message:
        reply = chat_with_groq(message, history)
        chat_log.append(f"AI: {reply}")

    return HTMLResponse(content=html_template.format(session_id=session_id, chat="".join([f'<div class="msg {"you" if m.startswith("You:") else "ai"}">{m}</div>' for m in chat_log])))

# ==== MAIN ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''

















#this is good except image everything work
'''
import uuid, shutil, os
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from pymongo import MongoClient
import pytesseract
import requests
import cv2
import numpy as np

# ==== CONFIG ====
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==== MONGODB ====
mongo = MongoClient("mongodb://localhost:27017")
db = mongo["insurance_db"]
users = db["users"]

# ==== FASTAPI ====
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# ==== SESSION STORE ====
sessions = {}

# ==== NGROK ====
public_url = ngrok.connect(8000).public_url
print(f"🚀 Ngrok tunnel running at: {public_url}")

# ==== HTML UI ====
html_template = """
<html><head>
<title>Insurance AI Assistant</title>
<style>
body {{ font-family: Arial; background: #f9f9f9; padding: 20px; }}
h2 {{ color: #2c3e50; }}
form {{ margin-top: 20px; }}
#chatbox {{ margin-top: 20px; background: #fff; padding: 15px; border-radius: 8px; max-width: 600px; }}
.msg {{ margin-bottom: 10px; }}
.you {{ color: #2980b9; }}
.ai {{ color: #27ae60; }}
</style>
</head><body>
<h2>Insurance Claim AI Assistant</h2>
<form action="/chat" method="post" enctype="multipart/form-data">
<input type="hidden" name="session_id" value="{session_id}" />
<textarea name="message" rows="3" cols="70" placeholder="Describe your accident or upload image..."></textarea><br><br>
<input type="file" name="file" /><br><br>
<button type="submit">Send</button>
</form>
<div id="chatbox">{chat}</div>
</body></html>
"""

# ==== OCR with Preprocessing ====
def detect_license_plate(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        print("🔠 OCR Raw Output:", text)
        for line in text.splitlines():
            line = line.strip().replace(" ", "")
            if len(line) >= 6 and any(c.isdigit() for c in line):
                return line.upper()
    except Exception as e:
        print("❌ OCR Error:", str(e))
    return None

# ==== Groq AI ====
def chat_with_groq(message, history):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": history + [{"role": "user", "content": message}]}
        )
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"

# ==== Routes ====
@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chat": [], "plate": None}
    return {"url": f"{public_url}/claim/{session_id}"}

@app.get("/claim/{session_id}", response_class=HTMLResponse)
def claim_page(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]
    chat_html = "".join([f'<div class="msg {"you" if m.startswith("You:") else "ai"}">{m}</div>' for m in chat_log])
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(""), file: UploadFile = File(None)):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)

    chat_log = sessions[session_id]["chat"]
    history = [{"role": "user" if m.startswith("You:") else "assistant", "content": m.split(":", 1)[1].strip()} for m in chat_log if ":" in m]
    ai_already_replied = False

    # Text message (plate)
    if message:
        chat_log.append(f"You: {message}")
        plate_candidates = [word.strip().upper().replace(" ", "") for word in message.split() if len(word) >= 6 and any(c.isdigit() for c in word)]
        for plate in plate_candidates:
            print("📥 Manual Plate Input:", plate)
            sessions[session_id]["plate"] = plate
            user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
            print("🔍 DB Lookup Result:", user)
            if user:
                summary = f"Name: {user['name']}, Policy: {user['policy_number']}, Insurer: {user['insurance_company']}, Vehicle: {user['vehicle']['make']} {user['vehicle']['model']}, Color: {user['vehicle']['color']}"
                chat_log.append(f"AI: Detected license plate is {plate}")
                chat_log.append(f"AI: {summary}")
                history.append({"role": "assistant", "content": summary})
                ai_already_replied = True
                break

    # Image file upload
    if file and file.filename:
        path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chat_log.append(f"You uploaded image: {file.filename}")
        plate = detect_license_plate(path)
        print("✅ Extracted Plate:", plate)
        if plate:
            sessions[session_id]["plate"] = plate
            user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
            chat_log.append(f"AI: Detected license plate is {plate}")
            if user:
                info = f"Detected plate: {plate}\nName: {user['name']}, Policy: {user['policy_number']}, Insurer: {user['insurance_company']}"
                chat_log.append(f"AI: {info}")
                history.append({"role": "assistant", "content": info})
                ai_already_replied = True
            else:
                chat_log.append("AI: Plate detected but no matching user found.")
                ai_already_replied = True
        else:
            chat_log.append("AI: No plate detected in the image. Please provide more details.")
            ai_already_replied = True

    # Default AI response
    if message and not ai_already_replied:
        reply = chat_with_groq(message, history)
        chat_log.append(f"AI: {reply}")

    return HTMLResponse(content=html_template.format(session_id=session_id, chat="".join([f'<div class="msg {"you" if m.startswith("You:") else "ai"}">{m}</div>' for m in chat_log])))

# ==== MAIN ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


'''













#this is same as previous but with extra function


'''


import os, uuid, shutil, datetime
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from pymongo import MongoClient
import pytesseract, requests
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
import re

# CONFIG
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
UPLOAD_DIR = "uploads"
PDF_DIR = "claims"
DOC_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

# MONGODB
mongo = MongoClient("mongodb://localhost:27017")
db = mongo["insurance_db"]
users = db["users"]
claims = db["claims"]

# FASTAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# SESSION STORE
sessions = {}

# NGROK
public_url = ngrok.connect(8000).public_url
print(f"\U0001F680 Ngrok tunnel running at: {public_url}")

# OCR

def detect_license_plate(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        for line in text.splitlines():
            line = line.strip().replace(" ", "")
            if len(line) >= 6 and any(c.isdigit() for c in line):
                return line.upper()
    except:
        pass
    return None

# Groq

def chat_with_groq(message, history):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": history + [{"role": "user", "content": message}]}
        )
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"

# PDF Generator

def generate_pdf(session_id, plate, user):
    filepath = f"{PDF_DIR}/{session_id}_{plate}.pdf"
    c = canvas.Canvas(filepath)
    c.drawString(100, 800, f"Insurance Claim Summary - {datetime.date.today()}")
    c.drawString(100, 770, f"Plate: {plate}")
    c.drawString(100, 750, f"Name: {user['name']}")
    c.drawString(100, 730, f"Policy No: {user['policy_number']}")
    c.drawString(100, 710, f"Insurer: {user['insurance_company']}")
    c.drawString(100, 690, f"Vehicle: {user['vehicle']['make']} {user['vehicle']['model']} ({user['vehicle']['color']})")
    c.drawString(100, 670, f"Contact: {user['phone']} / {user['email']}")
    c.drawString(100, 650, f"Address: {user['address']}")
    c.drawString(100, 620, "Status: Submitted")
    c.save()
    return filepath
#this is for simple webpage layout
'''''' # detele the extra
# HTML FORM (simplified)
html_template = """
<html><body>
<h2>Insurance Claim AI</h2>
<form action="/chat" method="post" enctype="multipart/form-data">
<input type="hidden" name="session_id" value="{session_id}" />
<textarea name="message" rows="4" cols="60"></textarea><br><br>
Upload Accident Image: <input type="file" name="file" /><br><br>
Upload RC/FIR: <input type="file" name="doc" /><br><br>
<button type="submit">Send</button>
</form>
<div>{chat}</div>
</body></html>
"""'''
'''
#this is for better layout

html_template = """
<html>
<head>
    <title>Insurance Claim Assistant</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 30px; display: flex; flex-direction: column; align-items: center; }}
        h2 {{ color: #2c3e50; margin-bottom: 10px; }}
        form {{ margin-bottom: 20px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%; max-width: 600px; }}
        textarea {{ width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #ccc; resize: vertical; }}
        input[type="file"] {{ margin-top: 10px; }}
        button {{ margin-top: 15px; background: #2c7be5; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; }}
        #chatbox {{ width: 100%; max-width: 600px; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .msg {{ margin-bottom: 15px; padding: 12px; border-radius: 10px; line-height: 1.4em; max-width: 80%; }}
        .you {{ background: #e8f0fe; align-self: flex-end; text-align: right; margin-left: auto; }}
        .ai {{ background: #d4edda; align-self: flex-start; text-align: left; margin-right: auto; }}
    </style>
</head>
<body>
    <h2>🚗 Insurance Claim Assistant</h2>
    <form action="/chat" method="post" enctype="multipart/form-data">
        <input type="hidden" name="session_id" value="{session_id}" />
        <textarea name="message" rows="3" placeholder="Describe the accident or enter your plate number..."></textarea><br>
        <label>Upload Accident Image:</label><br>
        <input type="file" name="file" /><br><br>
        <label>Upload RC / FIR / Other Document:</label><br>
        <input type="file" name="doc" /><br><br>
        <button type="submit">Send</button>
    </form>
    <div id="chatbox">
        {chat}
    </div>
</body>
</html>
"""




@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chat": [], "plate": None, "status": "draft"}
    return {"url": f"{public_url}/claim/{session_id}"}

@app.get("/claim/{session_id}", response_class=HTMLResponse)
def claim_page(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]
    chat_html = "<br>".join(chat_log)
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(""), file: UploadFile = File(None), doc: UploadFile = File(None)):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]

    # Handle Text Input
    chat_log.append(f"You: {message}")
    plate_candidates = re.findall(r"[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}", message.upper())
    user = None
    for plate in plate_candidates:
        user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
        if user:
            sessions[session_id]["plate"] = plate
            chat_log.append(f"AI: Found record for {plate}")
            filepath = generate_pdf(session_id, plate, user)
            claims.insert_one({"session_id": session_id, "plate": plate, "status": "submitted", "pdf": filepath})
            chat_log.append(f"AI: PDF generated and claim submitted ✅")
            break

    # Handle Image
    if file and file.filename:
        path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        plate = detect_license_plate(path)
        chat_log.append(f"AI: Detected plate: {plate}")
        user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
        if user:
            sessions[session_id]["plate"] = plate
            filepath = generate_pdf(session_id, plate, user)
            claims.insert_one({"session_id": session_id, "plate": plate, "status": "submitted", "pdf": filepath})
            chat_log.append(f"AI: PDF generated and claim submitted ✅")

    # Upload RC/FIR etc.
    if doc and doc.filename:
        doc_dir = f"{DOC_DIR}/{session_id}/"
        os.makedirs(doc_dir, exist_ok=True)
        doc_path = f"{doc_dir}{uuid.uuid4()}_{doc.filename}"
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(doc.file, buffer)
        chat_log.append(f"AI: Supporting document received: {doc.filename}")

    # AI Fallback
    reply = chat_with_groq(message, [])
    chat_log.append(f"AI: {reply}")
    chat_html = "<br>".join(chat_log)
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.get("/status/{plate}")
def check_status(plate: str):
    claim = claims.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}}, sort=[("_id", -1)])
    if claim:
        return {"plate": plate, "status": claim["status"], "pdf": claim["pdf"]}
    return {"error": "No claim found for this plate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''














#this works and same as above but with good ai




import os, uuid, shutil, datetime
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from pymongo import MongoClient
import pytesseract, requests
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
import re

# CONFIG
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
UPLOAD_DIR = "uploads"
PDF_DIR = "claims"
DOC_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

# MONGODB
mongo = MongoClient("mongodb://localhost:27017")
db = mongo["insurance_db"]
users = db["users"]
claims = db["claims"]

# FASTAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# SESSION STORE
sessions = {}

# NGROK
public_url = ngrok.connect(8000).public_url
print(f"\U0001F680 Ngrok tunnel running at: {public_url}")

# OCR
def detect_license_plate(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        for line in text.splitlines():
            line = line.strip().replace(" ", "")
            if len(line) >= 6 and any(c.isdigit() for c in line):
                return line.upper()
    except:
        pass
    return None

# Groq Chat
def chat_with_groq(message, history):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": history + [{"role": "user", "content": message}]}
        )
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"

# PDF Generator
def generate_pdf(session_id, plate, user):
    filepath = f"{PDF_DIR}/{session_id}_{plate}.pdf"
    c = canvas.Canvas(filepath)
    c.drawString(100, 800, f"Insurance Claim Summary - {datetime.date.today()}")
    c.drawString(100, 770, f"Plate: {plate}")
    c.drawString(100, 750, f"Name: {user['name']}")
    c.drawString(100, 730, f"Policy No: {user['policy_number']}")
    c.drawString(100, 710, f"Insurer: {user['insurance_company']}")
    c.drawString(100, 690, f"Vehicle: {user['vehicle']['make']} {user['vehicle']['model']} ({user['vehicle']['color']})")
    c.drawString(100, 670, f"Contact: {user['phone']} / {user['email']}")
    c.drawString(100, 650, f"Address: {user['address']}")
    c.drawString(100, 620, "Status: Submitted")
    c.save()
    return filepath

# Better Layout
html_template = """
<html>
<head>
    <title>Insurance Claim Assistant</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 30px; display: flex; flex-direction: column; align-items: center; }}
        h2 {{ color: #2c3e50; margin-bottom: 10px; }}
        form {{ margin-bottom: 20px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%; max-width: 600px; }}
        textarea {{ width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #ccc; resize: vertical; }}
        input[type="file"] {{ margin-top: 10px; }}
        button {{ margin-top: 15px; background: #2c7be5; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; }}
        #chatbox {{ width: 100%; max-width: 600px; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .msg {{ margin-bottom: 15px; padding: 12px; border-radius: 10px; line-height: 1.4em; max-width: 80%; }}
        .you {{ background: #e8f0fe; align-self: flex-end; text-align: right; margin-left: auto; }}
        .ai {{ background: #d4edda; align-self: flex-start; text-align: left; margin-right: auto; }}
    </style>
</head>
<body>
    <h2>🚗 Insurance Claim Assistant</h2>
    <form action="/chat" method="post" enctype="multipart/form-data">
        <input type="hidden" name="session_id" value="{session_id}" />
        <textarea name="message" rows="3" placeholder="Describe the accident or enter your plate number..."></textarea><br>
        <label>Upload Accident Image:</label><br>
        <input type="file" name="file" /><br><br>
        <label>Upload RC / FIR / Other Document:</label><br>
        <input type="file" name="doc" /><br><br>
        <button type="submit">Send</button>
    </form>
    <div id="chatbox">
        {chat}
    </div>
</body>
</html>
"""

@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chat": [], "plate": None, "status": "draft"}
    return {"url": f"{public_url}/claim/{session_id}"}

@app.get("/claim/{session_id}", response_class=HTMLResponse)
def claim_page(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]
    chat_html = "<br>".join(chat_log)
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.post("/chat", response_class=HTMLResponse)
async def chat(session_id: str = Form(...), message: str = Form(""), file: UploadFile = File(None), doc: UploadFile = File(None)):
    if session_id not in sessions:
        return HTMLResponse("<h3>Invalid session ID</h3>", status_code=404)
    chat_log = sessions[session_id]["chat"]

    chat_log.append(f"You: {message}")
    history = []

    # Text plate detection
    plate_candidates = re.findall(r"[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}", message.upper())
    for plate in plate_candidates:
        user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
        if user:
            sessions[session_id]["plate"] = plate
            user_context = f"User with plate {plate} identified as {user['name']}, policy number {user['policy_number']}, insured with {user['insurance_company']}. Vehicle: {user['vehicle']['make']} {user['vehicle']['model']} ({user['vehicle']['color']})."
            chat_log.append(f"AI: Found record for {plate}")
            chat_log.append(f"AI: {user_context}")
            filepath = generate_pdf(session_id, plate, user)
            claims.insert_one({"session_id": session_id, "plate": plate, "status": "submitted", "pdf": filepath})
            chat_log.append(f"AI: PDF generated and claim submitted ✅")
            history.append({"role": "assistant", "content": user_context})
            break

    # Image plate detection
    if file and file.filename:
        path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        plate = detect_license_plate(path)
        chat_log.append(f"AI: Detected plate: {plate}")
        user = users.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}})
        if user:
            sessions[session_id]["plate"] = plate
            user_context = f"User with plate {plate} identified as {user['name']}, policy number {user['policy_number']}, insured with {user['insurance_company']}. Vehicle: {user['vehicle']['make']} {user['vehicle']['model']} ({user['vehicle']['color']})."
            chat_log.append(f"AI: {user_context}")
            filepath = generate_pdf(session_id, plate, user)
            claims.insert_one({"session_id": session_id, "plate": plate, "status": "submitted", "pdf": filepath})
            chat_log.append(f"AI: PDF generated and claim submitted ✅")
            history.append({"role": "assistant", "content": user_context})

    # Supporting docs
    if doc and doc.filename:
        doc_dir = f"{DOC_DIR}/{session_id}/"
        os.makedirs(doc_dir, exist_ok=True)
        doc_path = f"{doc_dir}{uuid.uuid4()}_{doc.filename}"
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(doc.file, buffer)
        chat_log.append(f"AI: Supporting document received: {doc.filename}")

    # Chat with Groq
    reply = chat_with_groq(message, history)
    chat_log.append(f"AI: {reply}")
    chat_html = "<br>".join(chat_log)
    return HTMLResponse(content=html_template.format(session_id=session_id, chat=chat_html))

@app.get("/status/{plate}")
def check_status(plate: str):
    claim = claims.find_one({"plate": {"$regex": f"^{plate}$", "$options": "i"}}, sort=[("_id", -1)])
    if claim:
        return {"plate": plate, "status": claim["status"], "pdf": claim["pdf"]}
    return {"error": "No claim found for this plate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









