# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os

from dataclasses import dataclass
import math
import pandas as pd
import openai
import streamlit as st
from PIL import Image
from openai.error import OpenAIError


from utils import (
    get_number_of_characters,
    get_number_of_words,
    convert_characters_to_lines,
    convert_lines_to_characters,
)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


# Config
html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """
TITLE = "KIRA - ServiceTeam"


DATE_COLUMN = "article_publish_date"
relevant_columns = [
    "article_title",
    "article_id",
    "article_content",
    "num_words",  #  renamed from article_content_length
    DATE_COLUMN,
    # "article_author",
    # "article_newsroom",
]


article_title = None
article_content = None


openai.api_key = st.secrets["openai_api_token"]
# os.environ["REPLICATE_API_KEY"] = st.secrets["replicate_api_token"]

models = ["gpt-3.5-turbo", "pt-3.5-turbo-0301"]


# Alternative Prompt / Use cases
# Multiple Title Suggestions
# SEO Title Suggestions that drive engagement
# seo description suggestions
# social media title/description suggestions
# teaser suggestions
# Add Writing Styles (e.g. formal, informal, funny, professional, etc.)
# Filter by Newsroom / Author / Topic


st.set_page_config(page_title=TITLE, page_icon=":robot:", layout="wide")

with st.sidebar:
    st.image(Image.open("images/kira_v1_round.png"), width=90)


@dataclass
class Article:
    content: str
    num_characters_per_line: int = 27

    def word_count(self):
        return len(self.content.split(" "))

    def character_count(self):
        return len(self.content)

    def line_count(self):
        return math.ceil(self.character_count() / self.num_characters_per_line)


def convert_word_length_to_token_length(num_words: int):
    return round(num_words * 1.35)


def convert_character_length_to_toklen_length(num_characters: int):
    """Converts a character length to a token length. Assuming that the average word length is 8 characters."""
    return round(num_characters * 8 * 1.35)


def article_summarizer(article_content, text_limit_type, text_limit):
    print("Running Article Summarizer")
    token_limit = text_limit
    if text_limit_type == "Zeichen" or text_limit_type == "Zeilen":
        if text_limit_type == "Zeilen":
            character_limit = convert_lines_to_characters(text_limit, 27)
        else:
            token_limit = text_limit
    else:  # text_limit_type == "Wörter"
        token_limit = convert_word_length_to_token_length(text_limit)
        # system_prompt = f"Du bist ein Redakteur. Du kürzt Texte. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Texte und sollst ihn in etwas weniger als {character_limit} maximal  {character_limit} Zeichen inklusive Leerzeichen zusammenfassen. Beachte unbedingt diese Limitierung."
    system_prompt = f"Du bist ein Redakteur. Du kürzt Texte. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Texte und sollst ihn in exakt {token_limit} Token zusammenfassen. Beachte unbedingt diese Limitierung."
    message = [
        # {
        # "role": "system",
        # "content": system_prompt,
        # },
        {
            "role": "user",
            "content": f"{system_prompt}. Der Artikel : {article_content}",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        max_tokens=1500,
        stream=True,
    )
    return response, system_prompt


@st.cache_data
def title_suggestions(article_content, max_character_length):
    system_prompt = f"Du bist ein Redakteur. Du schreibst Titel. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Text und antwortest mit einer Liste von 5 möglichen Titeln. Die maximale erlaubte Länge pro Titel sind {max_character_length} Zeichen."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": article_content,
            },
        ],
    )
    response_content = response["choices"][0]["message"]["content"]
    return response_content, system_prompt


# @st.cache_data
# def announcement_generation(keywords):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Du bist ein hilfreicher Assistent der Ankündigungsschreiben verfasst. Du sprichst deutsch. ",
#             },
#             {
#                 "role": "user",
#                 "content": f"Schreibe mir ein Ankündigungsschreiben basierend auf folgenden Stichwörtern: {keywords}",
#             },
#         ],
#     )
#     response_content = response["choices"][0]["message"]["content"]
#     return response_content


# @st.cache_data
# def load_articles(nrows=None):
#     if nrows is None:
#         df_articles = pd.read_csv("data/articles_2023_03.csv")
#     else:
#         df_articles = pd.read_csv("data/articles_2023_03.csv", nrows=nrows)
#     df_articles[DATE_COLUMN] = pd.to_datetime(df_articles[DATE_COLUMN])
#     df_articles["article_content_length"] = df_articles[
#         "article_content_length"
#     ].astype(float)
#     # Filter Data
#     df_articles = df_articles[~df_articles["article_title"].isna()]
#     df_articles = df_articles.sort_values(by=DATE_COLUMN, ascending=False)
#     df_articles = df_articles.rename(columns={"article_content_length": "num_words"})
#     return df_articles


# def get_article_content():
# return article_content


def get_number_of_words(article_content: str):
    if article_content is None:
        return
    else:
        return len(article_content.split(" ")) - 1


def get_number_of_characters(article_content: str):
    return len(article_content)


# df_articles = load_articles()
article_content = None
st.markdown(
    """ <style> .font {
font-size:250px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """,
    unsafe_allow_html=True,
)


# -----------

with st.sidebar:
    st.write("__Willkommen bei KIRA__")
    st.write("Dein :blue[KI] :blue[R]umble :blue[A]ssistent.")
    st.write("KIRA unterstützt bei der Erstellung von Texten.")
    st.warning(
        "KIRA befindet sich gerade noch in einer frühen Entwicklungsphase. Bei Fragen, Fehlern oder Anregungen bitte an Nikolai Smolnikow wenden."
    )
    st.markdown(
        """
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        Made by KI Squad / FBI.  <br/>
        Powered by AI.
        """,
        unsafe_allow_html=True,
    )


st.title(f":blue[{TITLE}]")

# col_debug = st.container()

# if col_debug.checkbox("Artikelliste anzeigen"):
# col_debug.subheader("Artikelliste (Daten sind nicht Live)")
# col_debug.dataframe(df_articles[relevant_columns])

col_left_config, col_right_config = st.columns([1, 2])
col_left_config.markdown("----")
col_right_config.markdown("----")


modus = col_left_config.selectbox(
    "Modus",
    (
        "Texte kürzen",
        # "Antwortschreiben",
        # "Ankündigungstexte",
    ),
)
system_prompt = ""

if "text_input_old" not in st.session_state:
    st.session_state.text_input_old = ""


def clear_text_input():
    st.session_state.text_input_old = st.session_state.text_input
    st.session_state.text_input = ""


with st.container():
    if modus == "Texte kürzen":
        text_limit_type = col_left_config.selectbox(
            "Textlimit",
            (
                "Wörter",
                "Zeilen",
                "Zeichen",
            ),
        )
        text_limit = col_left_config.number_input(
            f"Neue Textlänge (in {text_limit_type})", step=5, value=50, max_value=600
        )

        col_title_config_left, col_title_config_right = col_left_config.columns(2)
        max_character_length_title = col_title_config_right.number_input(
            "Titel Textlänge (in Zeichen)", min_value=5, value=30, step=5, max_value=150
        )
        is_title_generation_on = col_title_config_left.radio(
            "Titel Vorschläge", ("Ja", "Nein")
        )
        col_right_config.text_area(
            "Text eingeben / einfügen", height=250, key="text_input"
        )
        article_content = st.session_state.text_input
        if st.session_state.text_input == "":
            article_content = st.session_state.text_input_old
        else:
            article_content = st.session_state.text_input

        print(f"Artikel Content: {article_content}")
        col_left_config_output, col_right_config_output = st.columns(2)
        article = None
        if article_content:
            article = Article(content=article_content)
            col_right_config.text(
                f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
            )
        (
            col_right_config_button_1,
            col_right_config_button_2,
            _,
        ) = col_right_config.columns(3)

        button_generate = col_right_config_button_1.button("Vorschläge generieren")
        button_clear = col_right_config_button_2.button(
            "Text Leeren", on_click=clear_text_input
        )

    if button_generate and article_content:
        col_left_prompt, col_right_prompt = st.columns(2)
        col_left_output, col_right_output = st.columns(2)
        try:
            response, system_prompt = article_summarizer(
                article_content=article_content,
                text_limit=text_limit,
                text_limit_type=text_limit_type,
            )
            print(f"Prompt: {system_prompt}")
            with col_right_prompt:
                with st.expander("Prompt anzeigen"):
                    st.write(system_prompt)
            col_left_output.header("Text Original")
            col_left_output.markdown(article_content)
            col_left_output.text(
                f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
            )
            col_right_output.header("Textvorschlag")
            response_full = []
            article_shortened = ""
            article_output_container = col_right_output.empty()
            for chunk in response:
                chunk_message = chunk["choices"][0]["delta"]
                # if "content" in chunk_message:
                if "content" in chunk_message:
                    response_full.append(chunk_message["content"])
                    article_shortened = "".join(response_full).strip()
                # result = result.replace("\n", "")
                article_output_container.markdown(article_shortened)
            article = Article(content=article_shortened)
            col_right_output.text(
                f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
            )
        except Exception as e:
            print(e)
            st.error(
                "Etwas ist schief gelaufen... Bitte versuche es etwas später noch einmal. "
            )
            st.error(e.error_code)
        print(f"Title Generation: {is_title_generation_on}")
        if is_title_generation_on == "Ja":
            print("Generating Titles...")
            try:
                with col_right_output:
                    # with st.spinner("Titel werden generiert..."):
                    response_title_gen, system_prompt_title = title_suggestions(
                        article_content, max_character_length_title
                    )
                print(f"Prompt: {system_prompt_title}")
                col_right_output.header("Titel Vorschläge")
                col_right_output.markdown(response_title_gen)
            except Exception as e:
                print(e)

# elif modus == "Ankündigungstexte":
# pdf = st.file_uploader("Upload your PDF", type="pdf")

# # extract the text
# if pdf is not None:
#     pdf_reader = PdfReader(pdf)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()

#     # split into chunks
#     text_splitter = CharacterTextSplitter(
#         separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
#     )
#     chunks = text_splitter.split_text(text)

#     col_left_config.markdown("Not Yet Implemented")

# if button_generate and article_content is not None:
#     num_words_original = get_number_of_words(article_content)
#     col_left_config_output.header(f"Text im Original")
#     col_left_config_output.subheader(
#         f"{num_words_original} Wörter | {get_number_of_characters(article_content)} Zeichen"
#     )
#     if modus == "Texte kürzen":

# elif modus == "Ankündigungstexte":
#     st.file_uploader("Upload Files")
#     st.write("Not Yet Implemented")
