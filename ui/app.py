import os
import requests
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
# Will now pick up 300 from docker-compose
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SEC", "300"))

st.set_page_config(page_title="Personalized News", layout="centered")
st.title("Personalized News Pipeline")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Describe yourself (interests, tone, what to avoid)...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                payload = {
                    "user_description": user_input,
                    "news_blob": None,
                    "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 320},
                }
                resp = requests.post(f"{GATEWAY_URL}/process", json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("personalized_news", "")
                st.markdown(answer)

                # optional debug expander
                with st.expander("Debug meta"):
                    st.json(data.get("meta", {}))
                    st.text_area("Profile (LLM1 SYS_PROMPT)", data.get("profile", ""), height=200)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.HTTPError as e:
                # Try to surface detailed gateway error if provided
                try:
                    err_payload = e.response.json()
                except Exception:
                    err_payload = {"detail": str(e)}
                st.error(f"Gateway error: {err_payload}")
            except Exception as e:
                st.error(f"Request failed: {e}")
