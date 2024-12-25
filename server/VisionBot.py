import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class Camera:
    def __init__(self, resolution=(2560, 1440), devicenumber=0, focus=50): 
        self.cap = cv2.VideoCapture(devicenumber)
        if not self.cap.isOpened():
            raise Exception("Fehler: Kamera konnte nicht geöffnet werden.")

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    def take_photo(self, filename="photo.png"):
        ret, frame = self.cap.read()
        if not ret:
            print("Fehler: Kein Bild erhalten.")
            return None
        cv2.imwrite(filename, frame)
        print(f"Foto erfolgreich aufgenommen: {filename}")
        return filename

    #def __del__(self):
        #if self.cap.isOpened():
            #self.cap.release()
            #cv2.destroyAllWindows()

# Kamera initialisieren und Foto aufnehmen
cam = Camera(devicenumber=0, focus=50)
photo_path = cam.take_photo()

# CUDA prüfen
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Modell laden
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large", torch_dtype=dtype
    ).to(device)
    print("Modell erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit()



if photo_path:
    # Bild laden und sicherstellen, dass es RGB ist
    raw_image = Image.open(photo_path)

    # Bedingungslose Bildbeschreibung
    inputs_unconditional = processor(raw_image, return_tensors="pt").to(device, dtype)
    out_unconditional = model.generate(**inputs_unconditional)
    print("Beschreibung (unkonditional):", processor.decode(out_unconditional[0], skip_special_tokens=True))
