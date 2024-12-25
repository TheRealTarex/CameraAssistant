import cv2

# Kamera öffnen (Kameraindex 0 verwenden)
cap = cv2.VideoCapture(0)

# Überprüfen, ob die Kamera erfolgreich geöffnet wurde
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

# Maximale Auflösung einstellen
# Beispiel: 1920x1080 (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

# Überprüfen, ob die eingestellten Werte übernommen wurden
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Eingestellte Auflösung: {width}x{height}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Kein Bild erhalten.")
        break

    cv2.imshow('Kamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
