import os
import openai
import tempfile
import azure.cognitiveservices.speech as speechsdk
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DeepLake
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import streamlit as st 

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = '2022-12-01'
os.environ["OPENAI_API_KEY"] = "10618104e76848b5ab7f7de89c7dbb9b"
os.environ["OPENAI_API_BASE"] =  "https://opex-az-openai.openai.azure.com/"

speech_key = "5fedb525dc2944aba9d3a93c7153ac79" 
speech_region = "southeastasia"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_config.speech_recognition_language = 'en-US'
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
speech_config.speech_synthesis_voice_name = 'en-US-JennyMultilingualNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)


# Define the system message template
system_template = """You are a call center agent having a conversation with a customer calling in to enquire about policy matters in Prudential. Use the uploaded document as a step by step template to answer customer questions.


Question: {}
Answer: ## Input your answer here ##
"""


template = """Conversation between Me and AI.

Current Conversation:
{history}
Me: {input}
AI:
"""
prompt_msg = PromptTemplate(input_variables=['history', 'input'], template=template)

def loading_file(uploaded_file,text_splitter,embeddings):
    file_name = uploaded_file.name
    db = DeepLake(
        dataset_path="./deeplake", embedding_function=embeddings, overwrite=True
    )

    with st.spinner("Loading {} ...".format(file_name)):
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name,file_name)
        with open(temp_filepath,'wb') as f:
            f.write(uploaded_file.getvalue())
    
        loader = PyPDFLoader(temp_filepath)
        doc = loader.load()   
        texts = text_splitter.split_documents(doc)
        db.add_documents(texts)
    
    return db

def display_conversation(messages):
    for convo in messages:
        with st.chat_message("user"):
            st.write(convo[0])
        with st.chat_message("assistant"):
            st.write(convo[1])
               

def submit():
    st.session_state.question = st.session_state.widget
    st.session_state.widget = ''

# def stop_talking():
#     if speech_synthesizer and speech_synthesizer.is_speaking:
#         speech_synthesizer.stop_speaking()

def main():
    ### Initialise
    text_splitter = TokenTextSplitter(chunk_size=500,chunk_overlap=50) 
    embeddings = OpenAIEmbeddings(deployment='ada002embedding',chunk_size=1)
    recognition = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    #llm = AzureOpenAI(deployment_name="chatgpt",model_name="gpt-35-turbo")
    llm = AzureOpenAI(deployment_name="opexchatbottest",model_name="text-davinci-003")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)

    st.set_page_config(page_title="Talk2GenAI", layout="wide")     
    
    pru_logo, title, opex_logo = st.columns(3, gap="large")

    with pru_logo:
        st.image("./prudential.png",width=200, use_column_width=False)

    with title:
        st.title("Talk2GenAI")


    with opex_logo:
        st.image("./opex.jfif",width=130,use_column_width=False)

    uploaded_file = st.file_uploader(label='Upload a PDF Document')

    ## Initialise conversation and other variables
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'file_loaded' not in st.session_state:
        st.session_state.file_loaded = False

    if 'question' not in st.session_state:
        st.session_state.question = ''

    if st.session_state.file_loaded == False:
        if uploaded_file is not None:
            db = loading_file(uploaded_file,text_splitter,embeddings)
            st.session_state.db = db
            st.success("File Loaded Successfully!!")
            st.session_state.file_loaded = True
    else:
        if uploaded_file is None:
            st.session_state.file_loaded = False
        else:
            db = st.session_state.db
            st.success("File Loaded Successfully!!")

    # Query through LLM    
    question = st.chat_input(placeholder="Type your question here!")   
    
    # Talking Button
    talking = st.button("Start Talking!")
    # stop_talking = st.button("Stop Talking!")

    # if stop_talking:
    #     speech_synthesizer.cancel_all()


    if question or talking:
        with st.spinner("Loading Response from Model ..."):
            if question:
                final_query = system_template.format(question) 
                convo_qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), memory=memory)
                result = convo_qa({'question':final_query}, return_only_outputs=True)

                if "Question:" in result['answer']:
                    final_result = result['answer'].split('Question:')[0].strip()
                else:
                    final_result = result['answer']
                temp_convo = [question.strip(),final_result.strip().replace("<|im_end|>","")]
                st.session_state.conversation_history.append(temp_convo)
                display_conversation(st.session_state.conversation_history)
                speech_synthesis_result = speech_synthesizer.speak_text_async(final_result).get()
            elif talking:
                if uploaded_file:
                    speech_recognition_result = recognition.recognize_once_async().get()

                    # If speech is recognized, send it to Azure OpenAI and listen for the response.
                    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        print("Recognized speech: {}".format(speech_recognition_result.text))
                        question = speech_recognition_result.text
                        final_query = system_template.format(speech_recognition_result.text) 
                        convo_qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), memory=memory)
                        result = convo_qa({'question':final_query}, return_only_outputs=True)

                        
                        text = result['answer']
                        print("Responses: "+ text)
                        temp_convo = [question.strip(),text.strip().replace("<|im_end|>","")]
                        st.session_state.conversation_history.append(temp_convo)
                        display_conversation(st.session_state.conversation_history)
                        # Azure text to speech output
                        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
                    else:
                        st.session_state.conversation_history.append(temp_convo)
                        display_conversation(st.session_state.conversation_history)
                else:
                    speech_recognition_result = recognition.recognize_once_async().get()

                    # If speech is recognized, send it to Azure OpenAI and listen for the response.
                    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        print("Recognized speech: {}".format(speech_recognition_result.text))
                        question = speech_recognition_result.text
                        convo = ConversationChain(prompt=prompt_msg, llm=llm, verbose=True, memory=ConversationBufferMemory(human_prefix='Me'))

                        text = convo.run(input=question)

                        print("Responses: "+ text)
                        temp_convo = [question.strip(),text.strip().replace("<|im_end|>","")]
                        st.session_state.conversation_history.append(temp_convo)
                        display_conversation(st.session_state.conversation_history)
                        # Azure text to speech output
                        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
                    else:
                        st.session_state.conversation_history.append(temp_convo)
                        display_conversation(st.session_state.conversation_history)

if __name__ == "__main__":
    main()