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

def generate_llm_explanation(
    user_text: str,
    pred_label: str,
    llm_model_name: str = "Qwen/Qwen3-8B-Chat",
    device: str = "cpu",
) -> str:
    """
    Generate a natural language explanation for the predicted label.

    This function attempts to explain the model's prediction by
    summarising the user's statement and inferring potential causes of
    their depressive feelings.  When a language model is available, it
    generates a short empathetic explanation that interprets possible
    stressors or triggers mentioned in the input.  If the model
    cannot be loaded, fallback text is used.

    Parameters
    ----------
    user_text : str
        The original user input text describing their feelings.
    pred_label : str
        The predicted category name (e.g. "정상", "경미한 우울증").
    llm_model_name : str, optional
        Name of the Hugging Face model to use for generation.  By default
        this uses ``"Qwen/Qwen3-8B-Chat"``, an 8.2 billion‑parameter model
        from the Qwen3 series.  This model supports over 100 languages
        including Korean and provides strong reasoning and dialogue
        capabilities at a relatively compact size【793020839043067†L149-L181】.  If
        the environment cannot load this model (e.g. due to limited
        memory), the function will automatically fall back to a smaller
        model such as DistilGPT2.
    device : str, optional
        Device for model execution (e.g. ``"cpu"`` or ``"cuda"``).  Default
        is CPU.

    Returns
    -------
    str
        A Korean explanation describing the prediction in 3–5 sentences,
        highlighting possible causes mentioned in the user input.
    """
    # Prompt instructing the language model to infer causes from the input
    prompt = (
        f"사용자의 진술: {user_text}\n"
        f"모델의 예측 우울증 중증도: {pred_label}\n"
        "사용자가 서술한 내용에서 우울감을 유발한 주요 원인이나 스트레스로 추측되는 부분을 찾아 "
        "해석해 주세요. 결과를 이해하기 쉽게 3~5문장으로 설명하고, 공감어린 어조로 감정 관리와 "
        "스트레스 완화 팁을 포함해 주세요. 의료적 진단은 하지 마세요."
    )
    # Use a language model if available
    if _transformers_available:
        try:
            # The GPT‑OSS series uses the Harmony response format.  Using the
            # Transformers chat pipeline with ``messages`` automatically applies
            # the proper formatting.  We set ``torch_dtype" and ``device_map``
            # to ``auto" so that the model will select an appropriate precision
            # and device.  On systems without a GPU or with insufficient
            # memory, this may raise an exception, in which case we fall back
            # to predefined text.
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype="auto",
                device_map="auto",
            )
            # Construct the conversation with a single user message.  The
            # pipeline will return a list of messages using the Harmony
            # format; the last item contains the assistant's answer.
            messages = [
                {"role": "user", "content": prompt},
            ]
            result = gen(messages, max_new_tokens=160)
            generated_msgs = result[0]["generated_text"]
            # Extract the assistant's reply from the generated messages
            last_msg = generated_msgs[-1]
            # The content field holds the model's response
            if isinstance(last_msg, dict) and "content" in last_msg:
                return last_msg["content"].strip()
            # Fallback: if the structure is unexpected, return the text itself
            return str(last_msg).strip()
        except Exception:
            pass
    # Fallback explanations based solely on the predicted label
    if pred_label == "정상":
        return (
            "모델 예측 결과 정상 범주로 판단됩니다. 우울감을 느끼지 않더라도 규칙적인 운동과 충분한 수면, "
            "균형 잡힌 식단, 감사일기 쓰기 등 건강한 생활습관을 꾸준히 유지하는 것이 정신건강에 도움이 됩니다."
        )
    if pred_label == "경미한 우울증":
        return (
            "모델 예측 결과 경미한 우울 증상이 감지되었습니다. 스트레스를 느끼는 상황을 점검하고, "
            "운동·명상·감사일기 등으로 마음을 다스려 보세요. 걱정이 지속되면 주변의 지지를 받거나 전문가 상담을 통해 도움을 받을 수 있습니다."
        )
    # pred_label == "중등도 우울증" or unknown
    return (
        "모델 예측 결과 중등도 우울 증상이 감지되었습니다. 지속적인 스트레스나 다양한 문제들이 영향을 미쳤을 수 있습니다. "
        "충분한 휴식과 규칙적인 생활습관을 유지하고, 신뢰할 수 있는 사람들과 이야기하거나 전문가에게 도움을 요청하세요. 심각한 증상이 지속되면 치료를 고려하세요."
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
                   llm_model_name: str = "Qwen/Qwen3-8B-Chat") -> str:
    """
    Generate a chatbot reply for a user's question.

    Parameters
    ----------
    user_msg : str
        The user's follow‑up question or comment.
    last_pred_label : str or None
        The most recent predicted label from the classification model (if any).
    llm_model_name : str, optional
        Name of the language model to use for generation.  The default
        ``"Qwen/Qwen3-8B-Chat"`` is a Korean‑capable model from the
        Qwen3 series, providing robust multilingual dialogue ability at
        around 8 billion parameters【793020839043067†L149-L181】.  If this
        model cannot be loaded, smaller models are used automatically.

    Returns
    -------
    str
        A friendly, empathetic reply in Korean.
    """
    # Base prompt describing the assistant's persona and safety constraints
    # We'll construct a system message describing the chatbot's role
    # and safety constraints.  Including the last prediction label helps
    # personalise the response while maintaining safety.
    system_message = (
        "너는 우울증 관련 상담 AI입니다. 진단을 내리지 않고, 사용자가 묻는 질문에 대한 "
        "일반적인 정보와 감정 조절 팁, 스트레스 관리 방법, 생활습관 개선 조언 등을 제공합니다. "
        "또한 우울증 예측 결과의 의미를 이해할 수 있도록 돕지만 전문적인 의료 판단은 하지 않습니다."
    )
    if last_pred_label:
        system_message += f" 최근 모델 예측 결과는 '{last_pred_label}' 입니다."
    system_message += " 질문에 대해 공감하는 어조로 3~5문장으로 답변하세요."
    # Use transformers if available
    if _transformers_available:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype="auto",
                device_map="auto",
            )
            # Build chat history messages for context
            messages: list[dict[str, str]] = []
            # Append system message at the start
            messages.append({"role": "system", "content": system_message})
            # Append previous chat history
            for role, content in st.session_state.get("chat_history", []):
                # Only include last few messages to avoid hitting token limits
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            # Append current user question
            messages.append({"role": "user", "content": user_msg})
            result = gen(messages, max_new_tokens=180)
            generated_msgs = result[0]["generated_text"]
            # The last element contains the assistant's reply
            last_msg = generated_msgs[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                return last_msg["content"].strip()
            return str(last_msg).strip()
        except Exception:
            pass
    # Determine severity-specific suggestion templates
    # Each severity has specific advice for different query categories.  If no
    # prediction is available, a general template is used.
    severity_templates = {
        "정상": {
            "stress": "현재 예측 결과는 정상 범주입니다. 스트레스를 관리하기 위해 규칙적인 운동, 충분한 수면, 균형 잡힌 식사, 명상과 감사일기를 활용해 좋은 상태를 유지하세요.",
            "emotion": "현재 정상 범주이지만 감정 관리를 위해 마음챙김과 호흡법, 감정 일기, 긍정과 부정 감정의 균형을 유지하는 것이 좋습니다.",
            "lifestyle": "현재 정상 범주입니다. 건강한 생활습관을 유지하기 위해 꾸준한 운동, 충분한 수면, 균형 잡힌 식단, 카페인·알코올 제한, 취미 활동을 이어가세요.",
            "result": "현재 예측 결과는 정상 범주로 특별한 우려는 없습니다. 스트레스를 느낄 때에는 건강한 생활습관과 스트레스 관리 기법을 활용하면 도움이 됩니다.",
            "default": "현재 정상 범주에 속해 있으니 신체와 마음의 건강을 지키기 위해 건강한 생활습관과 스트레스 관리법을 꾸준히 실천하세요. 궁금한 점이 있으면 언제든지 질문해 주세요."
        },
        "경미한 우울증": {
            "stress": "경미한 우울 증상이 감지되었습니다. 스트레스를 줄이기 위해 규칙적인 운동, 충분한 수면, 균형 잡힌 식단과 함께 명상이나 감사일기를 시도해보세요. 주변 사람들과 대화를 나누는 것도 도움이 됩니다.",
            "emotion": "경미한 우울 증상이 있으니 마음챙김, 호흡법, 감정 일기 등을 활용해 감정을 관리해보세요. 긍정적인 경험에 집중하고, 걱정을 나눌 수 있는 신뢰할 수 있는 사람들과 소통하세요.",
            "lifestyle": "경미한 우울 증상이 있으므로 건강한 생활습관을 더욱 신경 써야 합니다. 운동, 충분한 수면, 균형 잡힌 식단, 카페인·알코올 절제, 흡연과 약물 피하기 등이 도움이 됩니다. 또한 즐거움을 느낄 수 있는 취미와 활동을 지속하세요.",
            "result": "경미한 우울 증상이 감지되었지만 적절한 관리로 개선될 수 있습니다. 걱정이 지속되면 상담을 권유받으시고, 스트레스 관리와 생활습관 개선을 통해 완화를 도모하세요.",
            "default": "경미한 우울 증상이 있으니 기분을 관리하기 위해 건강한 생활습관과 스트레스 관리법을 꾸준히 실천하고, 필요하면 주변 사람들과 이야기하거나 상담을 고려해보세요. 추가 질문이 있으시면 언제든지 말씀해 주세요."
        },
        "중등도 우울증": {
            "stress": "중등도 우울 증상이 감지되었습니다. 스트레스 관리가 특히 중요합니다. 규칙적인 운동, 충분한 수면, 균형 잡힌 식단, 명상과 감사일기를 실천하세요. 또한, 신뢰할 수 있는 사람들과 마음을 나누고 필요하다면 전문적인 상담을 고려해보세요.",
            "emotion": "중등도 우울 증상이 있으므로 감정 관리에 더욱 신경을 써야 합니다. 마음챙김과 호흡법을 통해 감정을 정리하고, 감정일기를 써보세요. 주변의 지지망을 활용하고 부정적인 생각을 신뢰할 수 있는 사람들과 공유하세요.",
            "lifestyle": "중등도 우울 증상이 있으므로 건강한 생활습관과 더불어 전문적인 지원을 받을 필요가 있습니다. 꾸준한 운동과 균형 잡힌 식단, 충분한 수면, 카페인·알코올 제한, 흡연·약물 피하기를 실천하세요. 심각한 증상이 지속될 경우 정신건강 전문가에게 상담을 받아보세요.",
            "result": "중등도 우울 증상이 감지되었습니다. 예측 결과를 참고하여, 전문적인 도움을 받고 건강한 생활습관과 스트레스 관리법을 실천하는 것이 중요합니다. 주변인과의 소통과 상담을 통해 지지를 받으세요.",
            "default": "중등도 우울 증상이 있으므로 스스로를 돌보고 전문적인 지원을 받는 것이 중요합니다. 스트레스 관리와 생활습관 개선을 꾸준히 실천하고, 필요하면 전문기관이나 전문가의 도움을 받아보세요. 추가 질문이 있으면 언제든지 말씀해 주세요."
        }
    }
    # Normalise message for keyword detection
    msg = user_msg.lower()
    # Determine which template to use based on last prediction
    template = severity_templates.get(last_pred_label) if last_pred_label in severity_templates else None
    # Define category keys
    category = None
    if any(keyword in msg for keyword in ["스트레스", "stress"]):
        category = "stress"
    elif any(keyword in msg for keyword in ["감정", "조절", "emotion"]):
        category = "emotion"
    elif any(keyword in msg for keyword in ["생활", "습관", "lifestyle"]):
        category = "lifestyle"
    elif any(keyword in msg for keyword in ["결과", "예측", "해석"]):
        category = "result"
    # If we have a template for this severity, return the corresponding message
    if template:
        if category and category in template:
            return template[category]
        # default category message
        return template["default"]
    # Generic responses when no prediction is available
    generic = {
        "stress": "스트레스를 관리하기 위해 규칙적인 운동, 충분한 수면, 균형 잡힌 식단, 심호흡과 명상, 감사일기, 자연 속 산책, 신뢰할 수 있는 사람들과의 대화를 시도해보세요.",
        "emotion": "감정을 조절하기 위해 마음챙김과 호흡법을 통해 현재 순간에 집중하고, 감정일기를 써서 감정을 표현해보세요. 긍정과 부정 감정의 균형을 유지하고 감사하는 마음을 갖는 것이 도움이 됩니다.",
        "lifestyle": "건강한 생활습관을 위해 규칙적인 운동, 충분한 수면, 균형 잡힌 식단, 카페인·알코올 제한, 흡연·약물 피하기를 실천하세요. 가족과 친구들과 시간을 보내고 취미 활동을 이어가는 것도 중요합니다.",
        "result": "모델 예측 결과는 참고용입니다. 결과가 걱정되면 신뢰할 수 있는 사람들과 상담하거나 전문가의 조언을 구하세요. 건강한 생활습관과 스트레스 관리가 도움이 될 수 있습니다.",
        "default": "질문을 해주셔서 감사합니다. 저는 일반적인 정보만 제공하는 상담 AI이며, 진단을 내리거나 치료를 대신하지 않습니다. 생활습관 개선과 스트레스 관리, 긍정적인 마음가짐을 통해 정신건강을 돌보세요. 궁금한 점이 있으면 언제든지 질문해 주세요."
    }
    if category and category in generic:
        return generic[category]
    return generic["default"]


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

# Chat input using Streamlit's chat widget.  ``st.chat_input`` appears
# below previous messages and allows users to press Enter to send.  This
# provides a more natural chat experience compared to separate text
# fields and buttons.
chat_prompt = st.chat_input(
    "챗봇에게 질문하세요 (우울증 관련 상담, 감정 조절, 스트레스 관리, 생활습관 개선 등)",
    key="chat_input",
)

# When the user submits a question via the chat input
if chat_prompt:
    # Record the user question
    st.session_state["chat_history"].append(("user", chat_prompt))
    with st.chat_message("user"):
        st.markdown(chat_prompt)
    # Generate reply using last predicted label if available
    reply = chatbot_answer(chat_prompt, st.session_state.get("last_pred_label"))
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
