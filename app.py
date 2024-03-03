import streamlit as st
from src.core.utils import setup_dbqa


def format_response(response, source_document):
    if len(source_document) > 0:
        tsb_source = [x.metadata['file_path'].split('../../data/')[1] for x in source_document]
        tsb_source = list(set(tsb_source))

        if len(tsb_source) > 1:
            tsb_source_fmt = "\n".join(tsb_source)
        else:
            tsb_source_fmt = tsb_source[0]

        format_response = f"{response} \n\nSource TSBs:\n\n{tsb_source_fmt}"
    else:
        format_response = response
    return format_response


def main():

    # st.set_page_config(layout="wide")
    st.title("KIA TSB Chat bot")

    prompt = st.chat_input("Enter your question regarding KIA vehicle TSBs:")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": 'How can I help you ?'}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize context
    context = ""
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            context += f"{message['content']}\n"

    # React to user input
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response, source_document = setup_dbqa(query=prompt)
            ai_response = format_response(response, source_document)
            st.markdown(ai_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    main()
