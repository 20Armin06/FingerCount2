import cv2
import mediapipe as mp
import math

dot_style  = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=5)
line_style = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3)
mp_hands   = mp.solutions.hands
model      = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap        = cv2.VideoCapture(0)
cv2.namedWindow("window", cv2.WINDOW_NORMAL)


def distance(p1, p2):

    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def count_fingers_smart(landmarks, hand_label):

    fingers = []

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    thumb_distance = distance(thumb_tip, landmarks[0])
    thumb_ip_distance = distance(thumb_ip, landmarks[0])

    distance_check = thumb_distance > thumb_ip_distance


    if hand_label == "Right":
        direction_check = thumb_tip.x < thumb_ip.x
    else:
        direction_check = thumb_tip.x > thumb_ip.x


    if distance_check and direction_check:
        fingers.append(1)
    else:
        fingers.append(0)


    finger_tips = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]

    for tip_id, base_id in zip(finger_tips, finger_bases):
        tip_distance = distance(landmarks[tip_id], landmarks[0])
        base_distance = distance(landmarks[base_id], landmarks[0])


        if tip_distance > base_distance:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    s, f = cap.read()
    if not s:
        break

    f = cv2.flip(f, 1)
    f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    r = model.process(f_rgb)

    if r.multi_hand_landmarks:
        all_fingers = 0

        for d_hand_landmark, d_info in zip(r.multi_hand_landmarks, r.multi_handedness):
            mp_drawing.draw_landmarks(f, d_hand_landmark, mp_hands.HAND_CONNECTIONS, dot_style, line_style)

            jahat = d_info.classification[0].label
            landmarks = d_hand_landmark.landmark

            # شمارش انگشتان
            fingers = count_fingers_smart(landmarks, jahat)
            hand_fingers = sum(fingers)
            all_fingers += hand_fingers


        cv2.putText(f, f"Total Fingers: {all_fingers}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("window", f)
    if cv2.waitKey(1) in [27, ord('q'), ord('e')] or cv2.getWindowProperty("window", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()