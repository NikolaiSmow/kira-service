# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os

from dataclasses import dataclass
import math
import pandas as pd
import openai
import streamlit as st
from PIL import Image


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
TITLE = "Kira - ServiceTeam"


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


# Alternative Prompt / Use cases
# Multiple Title Suggestions
# SEO Title Suggestions that drive engagement
# seo description suggestions
# social media title/description suggestions
# teaser suggestions
# Add Writing Styles (e.g. formal, informal, funny, professional, etc.)
# Filter by Newsroom / Author / Topic


st.set_page_config(page_title=TITLE, page_icon=":robot:", layout="wide")

st.image(Image.open("images/kira_v1.png"), width=120)


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


def article_summarizer(article_content, text_limit_type, text_limit):
    print("Running Article Summarizer")

    if text_limit_type == "Zeichen" or text_limit_type == "Zeilen":
        if text_limit_type == "Zeilen":
            character_limit = convert_lines_to_characters(text_limit, 27)
        else:
            character_limit = text_limit
        system_prompt = "Du bist ein Redakteur. Du kürzt Artikel. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Artikel und fasst ihn gut es geht in {character_limit} Zeichen zusammen."
    else:
        system_prompt = "Du bist ein Redakteur. Du kürzt Artikel. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Artikel und fasst ihn gut es geht in {text_limit} Wörtern zusammen."
    message = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": article_content,
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        max_tokens=500,
        stream=True,
    )
    return response, system_prompt


@st.cache_data
def title_generation(article_content, max_character_length):
    system_prompt = "Du bist ein Redakteur. Du schreibst Titel. Du erfindest keine neuen Informationen. Du antwortest auf deutsch. Du erhälst einen Artikel antwortest mit einer Liste von 5 möglichen suchmaschinenoptimierten Titeln. Die maximale Titellänge beträgt {max_character_length} Zeichen."
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


@st.cache_data
def load_articles(nrows=None):
    if nrows is None:
        df_articles = pd.read_csv("data/articles_2023_03.csv")
    else:
        df_articles = pd.read_csv("data/articles_2023_03.csv", nrows=nrows)
    df_articles[DATE_COLUMN] = pd.to_datetime(df_articles[DATE_COLUMN])
    df_articles["article_content_length"] = df_articles[
        "article_content_length"
    ].astype(float)
    # Filter Data
    df_articles = df_articles[~df_articles["article_title"].isna()]
    df_articles = df_articles.sort_values(by=DATE_COLUMN, ascending=False)
    df_articles = df_articles.rename(columns={"article_content_length": "num_words"})
    return df_articles


# def get_article_content():
# return article_content


def get_number_of_words(article_content: str):
    if article_content is None:
        return
    else:
        return len(article_content.split(" ")) - 1


def get_number_of_characters(article_content: str):
    return len(article_content)


df_articles = load_articles()
article_content = None

# st.markdown(
#     """
# <style>
# .big-font {
#     font-size:300px !important;
# }
# .normal-font {
#     font-size:100px !important;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )
st.markdown(
    """ <style> .font {
font-size:250px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """,
    unsafe_allow_html=True,
)


# -----------

with st.sidebar:
    st.write("__Willkommen bei KIRA - dem KI Rumble Assistenten.__")
    st.write("KIRA hilft dir bei der Erstellung von Texten.")

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
        <br />
        <br />
        Made by FBI / Nikolai Smolnikow. <br /> Powered by AI.
        """,
        unsafe_allow_html=True,
    )


st.title(TITLE)
col_debug = st.container()

if col_debug.checkbox("Artikelliste anzeigen"):
    col_debug.subheader("Artikelliste (Daten sind nicht Live)")
    col_debug.dataframe(df_articles[relevant_columns])

col_left_config, col_right_config = st.columns([1, 2])
col_left_config.markdown("----")
col_right_config.markdown("----")


col_left_output, col_right_output = st.columns(2)

modus = col_left_config.selectbox(
    "Modus",
    (
        "Texte kürzen",
        # "Antwortschreiben",
        "Ankündigungstexte",
    ),
)


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
        f"Neue Textlänge (in {text_limit_type})", step=5, value=50
    )

    max_character_length_title = col_left_config.number_input(
        "Titel Textlänge (in Zeichen)", min_value=5, value=60, step=10
    )

    article_content = col_right_config.text_area("Text eingeben / einfügen", height=200)

    col_left_config_output, col_right_config_output = st.columns(2)
    article = None
    if article_content:
        article = Article(content=article_content)
        col_right_config.subheader(
            f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
        )
    button_value = col_right_config.button("Vorschläge generieren")

    col_left_config.markdown("----")
    col_right_config.markdown("----")

    if button_value and article_content:
        response, system_prompt = article_summarizer(
            article_content=article_content,
            text_limit=text_limit,
            text_limit_type=text_limit_type,
        )
        col_left_output.header("Artikel")
        col_left_output.markdown(article_content)
        col_left_output.subheader(
            f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
        )
        col_right_output.header("Artikel gekürzt")
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
        col_right_output.subheader(
            f"Zeichen: {article.character_count()} | Wörter: {article.word_count()} | Zeilen: {article.line_count()}"
        )

        response_title_gen, system_prompt_title = title_generation(
            article_content, max_character_length_title
        )
        col_right_config.markdown("----")
        col_right_output.header("Titel Vorschläge")
        col_right_output.markdown(response_title_gen)

else:
    col_left_config.markdown("Not Yet Implemented")


# if button_value and article_content is not None:
#     num_words_original = get_number_of_words(article_content)
#     col_left_config_output.header(f"Text im Original")
#     col_left_config_output.subheader(
#         f"{num_words_original} Wörter | {get_number_of_characters(article_content)} Zeichen"
#     )
#     if modus == "Texte kürzen":


# elif modus == "Ankündigungstexte":
#     st.file_uploader("Upload Files")
#     st.write("Not Yet Implemented")
