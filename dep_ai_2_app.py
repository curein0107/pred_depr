import streamlit as st
import joblib
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
# --- MiniLM / lightweight LLM import and availability flag ---
# The following try/except block attempts to import the Hugging Face transformers
# library for use with a small language model (e.g. MiniLM or DistilGPT2).  If
# transformers is not available in the runtime environment, the app will
# gracefully fall back to a deterministic, template‑based explanation.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    _transformers_available = True
except ImportError:
    _transformers_available = False


# --- 1. 모델 & 벡터라이저 불러오기 ---
# load vocab
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# load idf
idf = np.load("idf.npy")

# 초기화
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)  # ✅ 변수명 통일
tfidf_vectorizer.idf_ = idf

xgb_model = joblib.load("xgbc_nlp_depression_level_model.pkl")


# --- 2. 라벨 맵 ---
label_map = {0: "정상", 1: "경미한 우울증", 2: "중등도 우울증"}


# --- LLM 기반 설명 생성 함수 ---
def generate_llm_explanation(user_text: str, pred_label: str, llm_model_name: str = "distilgpt2", device: str = "cpu") -> str:
    """
    사용자의 입력 문장과 예측 라벨을 기반으로 간단한 설명을 생성합니다.
    가능한 경우 HuggingFace 모델을 활용하고, 그렇지 않으면 기본 설명을 반환합니다.
    """
    prompt = (
        f"사용자의 진술: {user_text}\n"
        f"모델의 예측 우울증 중증도: {pred_label}\n"
        "위 두 정보를 바탕으로 진단 결과를 이해하기 쉽게 설명해 주세요. "
        "친절하고 공감가는 어조로 한국어로 3~5문장으로 작성해 주세요."
    )
    # 시도: transformers를 이용한 텍스트 생성
    if globals().get('_transformers_available'):
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(llm_model_name)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
            out = generator(prompt, max_length=len(prompt.split()) + 60, num_return_sequences=1)
            return out[0]["generated_text"]
        except Exception:
            pass
    # fallback 설명
    if pred_label == "정상":
        base = "모델 예측 결과 정상 범주로 판단됩니다. 현재 입력하신 내용으로 보아 큰 우울 증상은 나타나지 않습니다. 긍정적인 생활을 계속 유지하세요."
    elif pred_label == "경미한 우울증":
        base = "모델 예측 결과 경미한 우울 증상이 감지되었습니다. 최근의 감정 변화를 주의 깊게 관찰하고, 스트레스를 줄이는 활동을 시도해 보는 것이 좋습니다. 필요하면 주변 사람들과 대화를 나누거나 상담을 고려해 보세요."
    else:
        base = "모델 예측 결과 중등도 우울증으로 평가됩니다. 입력하신 내용을 고려할 때 전문적인 상담과 치료가 필요할 수 있습니다. 가족이나 친구에게 도움을 요청하고, 정신건강의학과 전문의와 상담을 권장합니다."
    return base

# --- 3. Streamlit UI ---
st.set_page_config(page_title="우울증 진단서", layout="centered")
st.title("🧠 인공지능 기반 우울증 중증도 예측 진단서")

st.markdown("본 소프트웨어는 분당차병원 정신건강의학과 전문의의 우울증 진단 경험을 인공지능으로 학습하여, 문장만으로 우울증 중증도를 진단 예측할 수 있습니다.")
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
        # LLM 기반 설명 생성 및 표시
        explanation_text = generate_llm_explanation(user_input, pred_label)
        st.markdown("### AI 설명")
        st.write(explanation_text)


st.markdown("#  Additional information")
st.markdown("* Patent title : APPARATUS AND METHOD FOR PREDICTING DEPRESSION LEVELS USING NATURAL LANGUAGE PROCESSING AND EXPLAINABLE ARTIFICIAL INTELLIGENCE")
st.markdown("* Patent number :10-2024-0119065")
st.markdown("* Developer: Myung-Gwan Kim")
st.markdown("* Applicant: CHA University Industry-Academic Cooperation Foundation")
st.markdown("* Inventors: Myung-Gwan Kim, Hyun Wook Han, DaWoon Wang, JoonHo Park")









