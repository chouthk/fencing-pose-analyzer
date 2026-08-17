import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# 1. 角度計算輔助函式
def calculate_angle(a, b, c):
    """計算三點形成的夾角 (例如: 肩-肘-腕 或 髖-膝-踝)"""
    a = np.array(a) # 第一點
    b = np.array(b) # 頂點 (關節點)
    c = np.array(c) # 第三點

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# 2. UI 介面配置
st.set_page_config(page_title="Fencing AI Pose Analyzer", layout="wide")
st.title("🤺 劍擊動作 AI 骨骼分析系統")
st.write("上載弓步（Lunge）或刺擊影片，系統將自動分析持劍手伸展度與下肢角度。")

# 側邊欄參數設定
st.sidebar.header("分析設定")
hand_side = st.sidebar.selectbox("持劍手", ["右手 (Right)", "左手 (Left)"])
min_detection_confidence = st.sidebar.slider("偵測置信度", 0.1, 1.0, 0.5)

uploaded_file = st.file_uploader("選擇影片檔案 (MP4 / MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # 輸出暫存檔案
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    st.info("AI 正在逐影格分析動作中，請稍候...")
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    elbow_angles = []
    knee_angles = []
    frame_idx = 0

    # 3. 骨骼分析管線
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                      min_tracking_confidence=0.5,
                      model_complexity=2) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # 依持劍手選擇關節節點
                if "右手" in hand_side:
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                else:
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # 計算夾角
                arm_angle = calculate_angle(shoulder, elbow, wrist)
                leg_angle = calculate_angle(hip, knee, ankle)

                elbow_angles.append(arm_angle)
                knee_angles.append(leg_angle)

                # 畫面標註數值
                cv2.putText(image, f"Arm Ext: {int(arm_angle)} deg",
                            (int(elbow[0] * width), int(elbow[1] * height) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, f"Knee: {int(leg_angle)} deg",
                            (int(knee[0] * width), int(knee[1] * height) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2, cv2.LINE_AA)

            except Exception:
                pass

            # 繪製骨架連接線
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            out.write(image)
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()
        st.success("分析完成！")

        # 4. 分析報告與視覺化
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 動作數據指標")
            max_arm = max(elbow_angles) if elbow_angles else 0
            min_knee = min(knee_angles) if knee_angles else 0

            st.metric("最大手臂伸展角", f"{int(max_arm)}°", help="理想刺擊伸展應接近 165° - 180°")
            st.metric("弓步前膝最低角", f"{int(min_knee)}°", help="標準弓步大腿與小腿夾角約在 90° - 100°")

            # 診斷建議
            st.write("**AI 動作診斷：**")
            if max_arm < 160:
                st.warning("⚠️ 出劍手臂伸展不足，可能存在「推劍」或未完全伸臂的問題。")
            else:
                st.success("✅ 手臂伸展充分。")

            if min_knee < 80:
                st.error("⚠️ 前膝下沉過度（< 80°），容易造成膝關節過度負擔並影響回位速度。")
            elif min_knee > 110:
                st.info("ℹ️ 弓步下沉深度不足，可適度加深以延伸有效攻擊距離。")
            else:
                st.success("✅ 弓步前膝角度處於標準發力區間（90°-100°）。")

        with col2:
            st.subheader("📈 手臂伸展時序曲線")
            st.line_chart({"手臂開合角": elbow_angles})

            # 提供影片下載
            with open(out_file.name, "rb") as f:
                st.download_button(
                    label="📥 下載 AI 分析標註影片",
                    data=f,
                    file_name="analyzed_fencing.mp4",
                    mime="video/mp4"
                )
