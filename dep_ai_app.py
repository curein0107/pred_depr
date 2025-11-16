import streamlit as st
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Try to import the Hugging Face transformers library.  If it's not
# available (for instance due to missing dependencies), we set a
# flag so that the app can fall back to deterministic responses.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    _transformers_available = True
except ImportError:
    _transformers_available = False


# -----------------------------------------------------------------------------
# Model and vectorizer loading
#
# The following section loads the vocabulary and inverse document frequency
# values used to initialize the TF‑IDF vectorizer.  It then loads a
# pre‑trained XGBoost model from disk.  These files must reside in the
# same directory as this script.  If they are missing, Streamlit will
# display an appropriate error when the app is executed.
# -----------------------------------------------------------------------------

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Load IDF values and attach them to a fresh vectorizer
idf = np.load("idf.npy")
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
tfidf_vectorizer.idf_ = idf

# Load the classification model
xgb_model = joblib.load("xgbc_nlp_depression_level_model.pkl")

label_map = {0: "정상", 1: "경미한 우울증", 2: "중등도 우울증"}


# -----------------------------------------------------------------------------
# Explanation generation
#
# When a prediction is made, this helper function attempts to use a
# lightweight language model to generate a helpful explanation.  If
# transformers are unavailable or an error occurs during generation,
# predefined fallback text is returned based on the predicted label.
# -----------------------------------------------------------------------------

def generate_llm_explanation(user_text: str, pred_label: str,
                             llm_model_name: str = "distilgpt2",
                             device: str = "cpu") -> str:
    """
    Generate a natural language explanation for the predicted label.

    Parameters
    ----------
    user_text : str
        The original user input text describing their feelings.
    pred_label : str
        The predicted category name (e.g. "정상", "경미한 우울증").
    llm_model_name : str, optional
        Name of the Hugging Face model to use for generation.  Default is
        "distilgpt2" because of its small size and permissive license.
    device : str, optional
        Device for model execution (e.g. "cpu" or "cuda").  Default is CPU.

    Returns
    -------
    str
        A Korean explanation describing the prediction in 3–5 sentences.
    """
    # Construct a prompt instructing the model to summarise the result in
    # Korean, emphasising empathy and avoiding diagnostic language.
    prompt = (
        f"사용자의 진술: {user_text}\n"
        f"모델의 예측 우울증 중증도: {pred_label}\n"
        "위 두 정보를 바탕으로 결과를 이해하기 쉽게 설명해 주세요. "
        "친절하고 공감가는 어조로 한국어로 3~5문장으로 작성해 주세요. "
        "의료적 진단은 하지 말고, 일반적인 감정 관리 및 스트레스 관리 팁을 포함하세요."
    )
    # Attempt to generate text using a transformers pipeline
    if _transformers_available:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(llm_model_name)
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
            )
            result = generator(
                prompt,
                max_length=len(prompt.split()) + 80,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
            )
            generated = result[0]["generated_text"]
            # Remove the prompt from the generated output
            return generated.split("\n")[-1].strip()
        except Exception:
            # If anything fails, fall back to deterministic responses
            pass
    # Fallback explanations for each label
    if pred_label == "정상":
        return (
            "모델 예측 결과 정상 범주로 판단됩니다. 현재 입력하신 내용으로 보아 큰 우울 증상은 나타나지 "
            "않습니다. 감정일지를 작성하거나 규칙적인 생활을 통해 긍정적인 상태를 유지해보세요."
        )
    if pred_label == "경미한 우울증":
        return (
            "모델 예측 결과 경미한 우울 증상이 감지되었습니다. 최근 감정 변화를 주의 깊게 관찰하고, "
            "스트레스를 줄이는 활동을 시도해보세요. 필요하면 친구나 가족과 대화를 나누거나 상담을 "
            "고려해보는 것도 도움이 됩니다."
        )
    # pred_label == "중등도 우울증" or unknown
    return (
        "모델 예측 결과 중등도 우울 증상이 감지되었습니다. 우울감을 감소시키기 위해 충분한 휴식과 규칙적인 "
        "생활습관을 유지해보세요. 전문적인 상담이나 정신건강의학과 전문의의 도움이 필요할 수 있으니 참고하시기 "
        "바랍니다."
    )


# -----------------------------------------------------------------------------
# Chatbot helper
#
# This function creates a reply to the user's follow‑up questions.  It uses
# a lightweight language model when available and otherwise returns
# pre‑defined advice.  The chatbot emphasises that it provides general
# guidance without medical diagnosis.  If a prediction was previously
# generated, the last predicted label can be supplied to contextualise the
# answer.
# -----------------------------------------------------------------------------

def chatbot_answer(user_msg: str, last_pred_label: str | None = None,
                   llm_model_name: str = "distilgpt2") -> str:
    """
    Generate a chatbot reply for a user's question.

    Parameters
    ----------
    user_msg : str
        The user's follow‑up question or comment.
    last_pred_label : str or None
        The most recent predicted label from the classification model (if any).
    llm_model_name : str, optional
        Name of the lightweight language model to use.  Default is "distilgpt2".

    Returns
    -------
    str
        A friendly, empathetic reply in Korean.
    """
    # Base prompt describing the assistant's persona and safety constraints
    base_prompt = (
        "너는 우울증 관련 상담 AI입니다. 진단을 내리지 않고, 사용자가 묻는 질문에 대한 일반적인 정보와 "
        "감정 조절 팁, 스트레스 관리 방법, 생활습관 개선 조언 등을 제공합니다. 또한 우울증 예측 결과의 "
        "의미를 이해할 수 있도록 돕지만 전문적인 의료 판단은 하지 않습니다.\n"
    )
    if last_pred_label:
        base_prompt += f"참고로 최근 모델 예측 결과는 '{last_pred_label}' 입니다.\n"
    base_prompt += (
        "사용자의 질문에 대해 공감가는 어조로 3~5문장으로 답변해 주세요."
    )
    prompt = base_prompt + f"\n사용자: {user_msg}\nAI:"  # Format conversation
    # Use transformers if available
    if _transformers_available:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(llm_model_name)
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )
            response = gen(
                prompt,
                max_length=len(prompt.split()) + 60,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
            )
            text = response[0]["generated_text"]
            # Extract the assistant's part after the last "AI:" marker
            return text.split("AI:")[-1].strip()
        except Exception:
            pass
    # Fallback generic responses based on keywords
    msg = user_msg.lower()
    if any(keyword in msg for keyword in ["스트레스", "stress"]):
        return (
            "스트레스를 관리하기 위해서는 규칙적인 운동, 충분한 수면, 깊은 호흡이나 명상과 같은 이완 기법을 시도해 보세요. "
            "가벼운 산책이나 취미 활동도 도움이 됩니다."
        )
    if any(keyword in msg for keyword in ["감정", "조절", "emotion"]):
        return (
            "감정을 조절하는 방법으로는 마음챙김이나 호흡 운동을 통해 현재 순간에 집중하는 것이 있습니다. "
            "또한 감정을 억누르기보다 일기 쓰기 등으로 표현해보는 것도 좋습니다."
        )
    if any(keyword in msg for keyword in ["생활", "습관", "lifestyle"]):
        return (
            "건강한 생활습관을 위해 균형 잡힌 식사와 규칙적인 운동을 유지하고, 충분한 수면을 취하세요. "
            "또한 지나친 카페인이나 알코올 섭취를 피하는 것이 좋습니다."
        )
    if any(keyword in msg for keyword in ["결과", "예측", "해석"]):
        return (
            "예측 결과는 참고용이며, 감정을 이해하는 데 도움을 주는 지표입니다. "
            "정확한 진단을 위해서는 전문의의 상담이 필요함을 기억하세요."
        )
    # Default fallback
    return (
        "질문을 해주셔서 감사합니다. 저는 일반적인 정보만 제공하며, 진단을 내리거나 치료를 대신하지 않습니다. "
        "감정 조절이나 스트레스 관리에 대해 궁금한 점이 있으면 편하게 물어봐 주세요."
    )


# -----------------------------------------------------------------------------
# Streamlit application layout
#
# The following section defines the user interface elements: text input for
# describing current feelings, a prediction trigger button, display of
# probabilities and interpretive information, and a chat interface for
# follow‑up questions.
# -----------------------------------------------------------------------------

st.set_page_config(page_title="우울증 예측 및 상담", layout="centered")
st.title("🧠 인공지능 기반 우울증 예측 및 상담")

st.markdown(
    "본 소프트웨어는 분당차병원 정신건강의학과 전문의의 우울증 진단 경험을 "
    "인공지능으로 학습하여, 문장만으로 우울증 중증도를 예측하고 관련 상담 정보를 제공합니다. "
    "**이 앱의 정보는 참고용이며 정신건강의학과 전문의의 진단을 대체하지 않습니다.**"
)

# User input area
user_input = st.text_area("현재 느끼고 있는 감정을 문장으로 입력해보세요.", height=150)

# Initialize session state for predictions and chat
if "last_pred_label" not in st.session_state:
    st.session_state["last_pred_label"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of (role, message)


if st.button("진단하기"):
    if not user_input.strip():
        st.warning("⚠️ 입력된 텍스트가 없습니다.")
    else:
        # Vectorize and predict
        vec = tfidf_vectorizer.transform([user_input])
        probs = xgb_model.predict_proba(vec)[0]
        probs_percent = np.round(probs * 100, 1)
        pred_idx = int(np.argmax(probs_percent))
        pred_label = label_map[pred_idx]
        pred_conf = probs_percent[pred_idx]
        st.session_state["last_pred_label"] = pred_label
        today = datetime.today().strftime("%Y-%m-%d")
        # Display results card
        st.markdown(
            f"""
            <div style="border:2px solid #4CAF50; border-radius:10px; padding:20px; background-color:#f9fff9;">
                <h2 style="color:#2E7D32; text-align:center;">🧾 인공지능 예측 결과</h2>
                <p><b>발급일자</b>: {today}</p>
                <p><b>환자 진술</b>: {user_input}</p>
                <hr>
                <h3 style="color:#1565C0;">최종 예측</h3>
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
            unsafe_allow_html=True,
        )
        st.success("✅ 예측이 완료되었습니다. 결과는 참고용이며, 심각한 증상이 지속될 경우 전문의 상담이 필요합니다.")
        # Display explanation
        explanation = generate_llm_explanation(user_input, pred_label)
        st.markdown("### AI 설명")
        st.write(explanation)

# Separator
st.markdown("---")
st.subheader("💬 AI 상담 챗봇")

# Display previous chat messages
for role, message in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(message)

# Chat input: if the user types a question, produce an answer
user_query = st.chat_input("우울증 관련 일반 상담, 감정 조절, 스트레스 관리 등에 대해 질문해보세요.")
if user_query:
    # Append user message
    st.session_state["chat_history"].append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)
    # Generate bot reply using last predicted label as context
    reply = chatbot_answer(user_query, st.session_state.get("last_pred_label"))
    st.session_state["chat_history"].append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)


# Footer with additional information
st.markdown("---")
st.markdown("### 참고 정보")
st.markdown("* Patent title: APPARATUS AND METHOD FOR PREDICTING DEPRESSION LEVELS USING NATURAL LANGUAGE PROCESSING AND EXPLAINABLE ARTIFICIAL INTELLIGENCE")
st.markdown("* Patent number: 10-2024-0119065")
st.markdown("* Developer: Myung-Gwan Kim")
st.markdown("* Applicant: CHA University Industry-Academic Cooperation Foundation")
st.markdown("* Inventors: Myung-Gwan Kim, Hyun Wook Han, DaWoon Wang, JoonHo Park")
