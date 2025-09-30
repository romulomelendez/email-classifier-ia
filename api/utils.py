import os
import requests
from dotenv import load_dotenv
from typing import TypedDict

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY or not BASE_URL:
    raise ValueError("API_KEY e BASE_URL devem estar definidos no arquivo .env")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

PROMPTS = {
    "system": "Você é um assistente de e-mails profissional.",

    "classify": (
        "Analise o email abaixo e diga se ele é Produtivo ou Improdutivo. "
        "Categorias: - Produtivo: requer ação ou resposta (ex.: suporte, dúvidas, solicitações); "
        "- Improdutivo: não requer ação imediata (ex.: felicitações, agradecimentos). "
        "Email: {email}. Responda apenas com a categoria em letras minúsculas."
    ),

    "response": (
        "Escreva uma resposta breve, cordial e profissional em português (até 5 frases) "
        "para o seguinte email classificado como {category}. E-mail: {email}"
    ),

    "summary": (
        "Resuma em 1–2 frases, em português, o objetivo principal do remetente neste email: {email}"
    )
}

# Create a type
class EmailAnalysis(TypedDict):
    category: str
    suggestedResponse: str
    emailSummary: str


# Call DeepSeek API
def call_deepseek(prompt: str) -> str:
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": PROMPTS["system"]},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    try:
        resp = requests.post(BASE_URL, headers=HEADERS, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("A requisição para o DeepSeek expirou (timeout).")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro na requisição para o DeepSeek: {e}")

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Resposta inesperada do DeepSeek: {data}") from e


def classify_and_generate(email_text: str) -> EmailAnalysis:
    # Classificação
    category_prompt = PROMPTS["classify"].format(email=email_text)
    category = call_deepseek(category_prompt)

    # Resposta sugerida
    response_prompt = PROMPTS["response"].format(category=category, email=email_text)
    suggested_response = call_deepseek(response_prompt)

    # Resumo
    summary_prompt = PROMPTS["summary"].format(email=email_text)
    email_summary = call_deepseek(summary_prompt)

    return EmailAnalysis(
        category=category,
        suggestedResponse=suggested_response,
        emailSummary=email_summary
    )
