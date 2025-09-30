from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader

from utils import classify_and_generate

app = FastAPI(title="E-mail Classifier API")

# CORS liberado para dev (ajuste depois para seu domínio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    content: str


# Extract PDF text
def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts).strip()


# Routes
@app.post("/analyze-text")
def analyze_text(req: TextRequest):
    text = req.content

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty e-mail content.")
    return classify_and_generate(text)


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    data = await file.read()
    content_type = (file.content_type or "").lower()

    if content_type in ("text/plain", "application/octet-stream"):
        text = data.decode("utf-8", errors="ignore")
    elif content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        try:
            text = extract_text_from_pdf(data)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read PDF.")
    else:
        raise HTTPException(status_code=415, detail="Only .txt or .pdf are supported.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    return classify_and_generate(text)
