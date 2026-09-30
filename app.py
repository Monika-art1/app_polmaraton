import os
import json
import pandas as pd
import joblib
import streamlit as st

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

from langfuse import get_client
from langfuse.openai import OpenAI


# ============================================================
# KONFIGURACJA
# ============================================================

st.set_page_config(
    page_title="Kalkulator czasu półmaratonu",
    page_icon="👟",
    layout="centered"
)


# ============================================================
# ENV
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv(
    "LANGFUSE_BASE_URL",
    "https://cloud.langfuse.com"
)

# Ustawiamy konfigurację przed inicjalizacją klientów.
os.environ["LANGFUSE_PUBLIC_KEY"] = (
    LANGFUSE_PUBLIC_KEY or ""
)

os.environ["LANGFUSE_SECRET_KEY"] = (
    LANGFUSE_SECRET_KEY or ""
)

os.environ["LANGFUSE_BASE_URL"] = (
    LANGFUSE_BASE_URL
)


# ============================================================
# SPRAWDZENIE OPENAI
# ============================================================

if not OPENAI_API_KEY:

    st.error(
        "Brak OPENAI_API_KEY. "
        "Dodaj klucz OPENAI_API_KEY do pliku .env."
    )

    st.stop()


# ============================================================
# OPENAI — KLIENT Z INTEGRACJĄ LANGFUSE
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# LANGFUSE
# ============================================================

langfuse = None

if (
    LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY
):

    try:

        langfuse = get_client()

    except Exception as e:

        st.warning(
            "Langfuse nie został poprawnie "
            f"zainicjalizowany: {e}"
        )


# ============================================================
# MODEL ML
# ============================================================

MODEL_PATH = "best_Time_model.pkl"

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(
        f"Nie udało się wczytać modelu "
        f"{MODEL_PATH}.\n\n{e}"
    )

    st.stop()


# ============================================================
# OBRAZ
# ============================================================

IMAGE_URL = (
    "https://img.freepik.com/premium-vector/"
    "woman-running-blue-purple-image-with-blue-background_951778-67216.jpg"
)


# ============================================================
# STYL
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: url("{IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(255, 255, 255, 0.90);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        max-width: 900px;
    }}

    .stApp p,
    .stApp label,
    .stApp span {{
        color: #0066cc !important;
    }}

    h1,
    h2,
    h3 {{
        color: #0052a3 !important;
    }}

    input,
    textarea {{
        color: #0066cc !important;
        background-color: white !important;
    }}

    [data-baseweb="select"] {{
        color: #0066cc !important;
        background-color: white !important;
    }}

    [data-baseweb="select"] * {{
        color: #0066cc !important;
    }}

    div.stButton > button {{
        width: 100% !important;
        background-color: #0066cc !important;
        color: #ffffff !important;
        border: 2px solid #0052a3 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }}

    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {{
        color: #ffffff !important;
    }}

    div.stButton > button:hover {{
        background-color: #0052a3 !important;
        border-color: #003d7a !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TYTUŁ
# ============================================================

st.title(
    "👟 Kalkulator czasu półmaratonu 👟"
)

st.write(
    "Podaj swoje dane, aby oszacować "
    "czas ukończenia półmaratonu."
)


# ============================================================
# SCHEMAT DANYCH LLM
# ============================================================

class DaneBiegacza(BaseModel):

    imie: Optional[str] = Field(
        default=None,
        description="Imię użytkownika."
    )

    wiek: Optional[int] = Field(
        default=None,
        description="Wiek użytkownika."
    )

    plec: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "0 = kobieta, "
            "1 = mężczyzna."
        )
    )

    czas_5km: Optional[str] = Field(
        default=None,
        description=(
            "Czas na 5 km "
            "w formacie MM:SS."
        )
    )


# ============================================================
# CZAS → SEKUNDY
# ============================================================

def parse_time_to_seconds(czas):

    if czas is None:

        raise ValueError(
            "Brak czasu na 5 km."
        )

    czas = str(czas).strip()

    czas = czas.replace(".", ":")
    czas = czas.replace(",", ":")

    parts = czas.split(":")

    if len(parts) != 2:

        raise ValueError(
            "Nieprawidłowy format czasu."
        )

    try:

        minutes = int(parts[0])
        seconds = int(parts[1])

    except ValueError:

        raise ValueError(
            "Czas musi być zapisany "
            "jako MM:SS."
        )

    if minutes < 0:

        raise ValueError(
            "Minuty nie mogą być ujemne."
        )

    if seconds < 0 or seconds >= 60:

        raise ValueError(
            "Sekundy muszą być "
            "w zakresie 0–59."
        )

    return minutes * 60 + seconds


# ============================================================
# WALIDACJA
# ============================================================

def validate_runner_data(dane):

    errors = []

    if dane.wiek is not None:

        if dane.wiek < 10:

            errors.append(
                "Wiek musi wynosić "
                "co najmniej 10 lat."
            )

        if dane.wiek > 100:

            errors.append(
                "Wiek nie może być "
                "większy niż 100 lat."
            )

    if dane.plec is not None:

        if dane.plec not in [0, 1]:

            errors.append(
                "Nieprawidłowa wartość płci."
            )

    if dane.czas_5km is not None:

        try:

            parse_time_to_seconds(
                dane.czas_5km
            )

        except ValueError:

            errors.append(
                "Nieprawidłowy czas na 5 km."
            )

    return errors


# ============================================================
# BRAKUJĄCE DANE
# ============================================================

def get_missing_data(dane):

    missing = []

    if dane.wiek is None:
        missing.append("wiek")

    if dane.plec is None:
        missing.append("płeć")

    if dane.czas_5km is None:
        missing.append("czas na 5 km")

    return missing


# ============================================================
# OPENAI — EKSTRAKCJA
# ============================================================

def extract_runner_data(user_text):

    response = client.chat.completions.parse(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",

                "content": """
Jesteś modułem ekstrakcji danych
dla kalkulatora czasu półmaratonu.

Twoim zadaniem jest WYŁĄCZNIE
wyciągnięcie danych użytkownika.

NIE OBLICZAJ czasu półmaratonu.

NIE PRZEWIDUJ wyniku.

NIE wykonuj predykcji.

Potrzebne dane:

- imię
- wiek
- płeć
- czas na 5 km

ZASADY:

1. Wiek musi być liczbą całkowitą.

2. Płeć:
   kobieta = 0
   mężczyzna = 1

3. Nigdy nie zgaduj płci
   na podstawie imienia.

4. Jeśli użytkownik nie poda
   konkretnej informacji,
   zwróć null.

5. Czas 5 km zawsze zwracaj
   jako MM:SS.

6. Nie wykonuj żadnych
   obliczeń czasu półmaratonu.
"""
            },

            {
                "role": "user",
                "content": user_text
            }
        ],

        response_format=DaneBiegacza,

        temperature=0,

        name="runner-data-extraction",

        metadata={
            "application":
                "half-marathon-calculator",

            "task":
                "structured-extraction"
        }
    )

    return response.choices[0].message.parsed


# ============================================================
# LANGFUSE — MONITOROWANIE EKSTRAKCJI
# ============================================================

def evaluate_llm_extraction(
    user_text,
    dane,
    missing,
    validation_errors
):

    if langfuse is None:

        return

    try:

        required_fields = 3

        extracted_fields = (
            required_fields
            - len(missing)
        )

        completeness = (
            extracted_fields
            / required_fields
        )

        validation_ok = (
            len(validation_errors) == 0
        )

        with langfuse.start_as_current_observation(

            as_type="span",

            name="runner-data-validation"

        ) as span:

            span.update(

                input={
                    "user_text": user_text
                },

                output={

                    "extracted_data": {

                        "imie":
                            dane.imie,

                        "wiek":
                            dane.wiek,

                        "plec":
                            dane.plec,

                        "czas_5km":
                            dane.czas_5km
                    },

                    "missing_fields":
                        missing,

                    "validation_errors":
                        validation_errors
                },

                metadata={

                    "application":
                        "half-marathon-calculator",

                    "task":
                        "structured-extraction"
                }
            )

            span.score(
                name="extraction_completeness",
                value=float(completeness)
            )

            span.score(
                name="extraction_valid",
                value=(
                    1.0
                    if validation_ok
                    else 0.0
                )
            )

        langfuse.flush()

    except Exception as e:

        # Podczas testów pokazujemy błąd,
        # zamiast go ukrywać.

        st.warning(
            "⚠️ Langfuse: "
            f"nie udało się zapisać "
            f"obserwacji: {e}"
        )


# ============================================================
# MODEL ML
# ============================================================

def predict_half_marathon(
    wiek,
    plec,
    czas_5km
):

    dane = pd.DataFrame({

        "Wiek": [wiek],

        "Płeć": [plec],

        "5 km Czas": [czas_5km]
    })

    prediction = model.predict(
        dane
    )[0]

    godziny = int(
        prediction // 3600
    )

    minuty = int(
        (prediction % 3600) // 60
    )

    sekundy = int(
        prediction % 60
    )

    return (
        f"{godziny:02d}:"
        f"{minuty:02d}:"
        f"{sekundy:02d}"
    )


# ============================================================
# WYNIK
# ============================================================

def show_result(wynik):

    st.markdown(
        "## 🏁 Szacowany czas półmaratonu"
    )

    st.markdown(
        f"# {wynik}"
    )


# ============================================================
# METODA 1 — LLM
# ============================================================

st.subheader(
    "🧠 Opowiedz o sobie"
)

st.write(
    "Opisz swoje dane własnymi słowami. "
    "OpenAI wyodrębni informacje potrzebne "
    "do predykcji."
)

st.info(
    "Przykład: "
    "„Mam 35 lat, jestem kobietą "
    "i przebiegam 5 km w 24:30.”"
)

opis_uzytkownika = st.text_area(

    "Twoja wiadomość",

    placeholder=(
        "Np. Mam 35 lat, jestem kobietą "
        "i 5 km przebiegam w 24 minuty "
        "i 30 sekund."
    ),

    height=130,

    key="opis_uzytkownika"
)


if st.button(
    "🧠 Odczytaj dane i oblicz",
    key="llm_button"
):

    if not opis_uzytkownika.strip():

        st.warning(
            "Napisz kilka informacji o sobie."
        )

    else:

        try:

            with st.spinner(
                "OpenAI odczytuje dane..."
            ):

                dane_llm = (
                    extract_runner_data(
                        opis_uzytkownika
                    )
                )

            missing = get_missing_data(
                dane_llm
            )

            validation_errors = (
                validate_runner_data(
                    dane_llm
                )
            )

            # ----------------------------------------
            # LANGFUSE
            # ----------------------------------------

            evaluate_llm_extraction(

                user_text=
                    opis_uzytkownika,

                dane=
                    dane_llm,

                missing=
                    missing,

                validation_errors=
                    validation_errors
            )

            # ----------------------------------------
            # BRAKUJĄCE
            # ----------------------------------------

            if missing:

                st.warning(
                    "Brakuje danych potrzebnych "
                    "do wykonania predykcji."
                )

                for item in missing:

                    st.markdown(
                        f"- **{item}**"
                    )

            # ----------------------------------------
            # WALIDACJA
            # ----------------------------------------

            elif validation_errors:

                for error in validation_errors:

                    st.error(error)

            # ----------------------------------------
            # PREDYKCJA
            # ----------------------------------------

            else:

                st.success(
                    "✅ Dane zostały poprawnie "
                    "odczytane przez OpenAI."
                )

                extracted_data = {

                    "imie":
                        dane_llm.imie,

                    "wiek":
                        dane_llm.wiek,

                    "plec":
                        dane_llm.plec,

                    "czas_5km":
                        dane_llm.czas_5km
                }

                with st.expander(
                    "🔎 Dane wyodrębnione przez LLM"
                ):

                    st.json(
                        extracted_data
                    )

                czas_5km_sekundy = (
                    parse_time_to_seconds(
                        dane_llm.czas_5km
                    )
                )

                # ====================================
                # JEDYNE ŹRÓDŁO PREDYKCJI
                # ====================================

                wynik = predict_half_marathon(

                    wiek=dane_llm.wiek,

                    plec=dane_llm.plec,

                    czas_5km=
                        czas_5km_sekundy
                )

                show_result(wynik)

        except Exception as e:

            st.error(
                "Wystąpił błąd podczas "
                f"przetwarzania: {e}"
            )


# ============================================================
# SEPARATOR
# ============================================================

st.divider()


# ============================================================
# METODA 2 — FORMULARZ RĘCZNY
# ============================================================

st.subheader(
    "✍️ Wprowadź dane bezpośrednio"
)

st.write(
    "Możesz również pominąć OpenAI "
    "i podać dane bezpośrednio."
)


wiek = st.number_input(

    "Wiek",

    min_value=10,

    max_value=100,

    value=35,

    step=1,

    key="wiek_manual"
)


plec = st.selectbox(

    "Płeć",

    options=[0, 1],

    format_func=lambda x:
        "Kobieta"
        if x == 0
        else "Mężczyzna",

    key="plec_manual"
)


czas_5km = st.text_input(

    "Czas na 5 km (MM:SS)",

    value="22:30",

    key="czas_manual"
)


if st.button(
    "🏃 Oblicz czas półmaratonu",
    key="manual_button"
):

    try:

        czas_5km_sekundy = (
            parse_time_to_seconds(
                czas_5km
            )
        )

        wynik = predict_half_marathon(

            wiek=wiek,

            plec=plec,

            czas_5km=
                czas_5km_sekundy
        )

        show_result(wynik)

    except ValueError:

        st.error(
            "Nieprawidłowy czas. "
            "Wpisz np. 22:30."
        )

    except Exception as e:

        st.error(
            f"Wystąpił błąd podczas "
            f"predykcji: {e}"
        )


# ============================================================
# INFORMACJA
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        color: #0066cc;
        margin-top: 25px;
        font-size: 14px;
    ">
        Wynik półmaratonu jest obliczany
        wyłącznie przez model uczenia maszynowego.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OBRAZEK NA DOLE
# ============================================================

st.markdown(
    "<div style='margin-top: 25px;'></div>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    st.image(
        IMAGE_URL,
        width=200
    )
