import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from datetime import datetime

# --- 1. 모델 & 벡터라이저 불러오기 ---
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
xgb_model = joblib.load("xgbc_nlp_depression_level_model.pkl")

# --- 2. 라벨 맵 ---
label_map = {0: "정상", 1: "경미한 우울증", 2: "중등도 우울증"}

# --- 3. Streamlit UI ---
st.set_page_config(page_title="우울증 진단서", layout="centered")
st.title("🧠 인공지능 기반 우울증 중증도 예측 진단서")

st.markdown("본 인공지능 소프트웨어는 의료기관 정신건강의학과 전문의의 우울증 진단 경험을 인공지능으로 학습하여, 문장만으로 우울증 중증도를 진단 예측할 수 있습니다.")
st.markdown("아래 진단서는 인공지능 분석 결과이며, **정신건강의학과 전문의의 최종 진단을 대체하지 않습니다.**")

# 입력창
user_input = st.text_area("현재 느끼고 있는 감정을 문장으로 입력해보세요.", height=150)

if st.button("진단하기"):
    if not user_input.strip():
        st.warning("⚠️ 입력된 텍스트가 없습니다.")
    else:
        # 예측 확률
        vec = tfidf_vectorizer.transform([user_input])
        probs = xgb_model.predict_proba(vec)[0]

        # 확률을 퍼센트(%)로 변환 후 소수점 1자리까지 반올림
        probs_percent = np.round(probs * 100, 1)

        # 가장 높은 확률 클래스
        pred_idx = int(np.argmax(probs_percent))
        pred_label = label_map[pred_idx]
        pred_conf = probs_percent[pred_idx]

        # 발급일자
        today = datetime.today().strftime("%Y-%m-%d")

        # --- 진단서 카드 출력 ---
        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9fff9;
                font-family: Arial, sans-serif;
                ">
                <h2 style="color:#2E7D32; text-align:center;">🧾 인공지능 예측 진단서</h2>
                <p><b>발급일자</b>: {today}</p>
                <p><b>환자 진술</b>: {user_input}</p>
                <hr>
                <h3 style="color:#1565C0;">최종 예측 진단</h3>
                <p style="font-size:22px; font-weight:bold; color:#D32F2F;">
                    {pred_label} ({pred_conf:.1f}%)
                </p>
                <hr>
                <h3 style="color:#1565C0;">전체 예측 확률</h3>
                <ul>
                    <li>{label_map[0]}: <b>{probs_percent[0]:.1f}%</b></li>
                    <li>{label_map[1]}: <b>{probs_percent[1]:.1f}%</b></li>
                    <li>{label_map[2]}: <b>{probs_percent[2]:.1f}%</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("✅ 예측이 완료되었습니다. 결과는 참고용이며, 중등도 우울증일 경우 전문의 상담이 필요합니다.")

