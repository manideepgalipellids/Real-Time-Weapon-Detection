import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("Press Q or ESC to close")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera")
        break

    cv2.imshow("My Camera - Weapon Detection Project", frame)

    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q") or key == 27:
        print("Closing camera...")
        break

camera.release()
cv2.destroyAllWindows()