#for mongo upload
'''
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client.insurance_db
users_collection = db.users

# Sample user data
users_collection.insert_many([
    {
        "plate": "TN07CY3098",
        "name": "Aditya Dubey",
        "policy_number": "POL123456789",
        "insurance_company": "SafeDrive Insurance Co.",
        "vehicle": {
            "make": "Honda",
            "model": "Activa",
            "color": "Red"
        },
        "phone": "+91-9876543210",
        "email": "aditya@example.com",
        "address": "Nai Basti, Indore, MP"
    },
    {
        "plate": "MH12XY7890",
        "name": "Nitesh Tripathi",
        "policy_number": "POL987654321",
        "insurance_company": "SecureCover Ltd.",
        "vehicle": {
            "make": "TVS",
            "model": "Jupiter",
            "color": "Blue"
        },
        "phone": "+91-9123456780",
        "email": "nitesh@example.com",
        "address": "Shivaji Nagar, Pune, MH"
    }
])
print("Sample users inserted.")
'''













from PIL import Image
import pytesseract

img = Image.open("IMG_6440 1.JPG").convert("L")
img = img.resize((img.width * 2, img.height * 2))
text = pytesseract.image_to_string(img, config='--psm 6')
print(text)






