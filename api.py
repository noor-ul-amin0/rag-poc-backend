import asyncio
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qa-api")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
RETRIEVAL_TOP_K = 10


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = BASE_DIR / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_required_config(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required config value: '{key}'")
    return str(value)


def resolve_azure_openai_endpoint(config: dict[str, Any]) -> str:
    endpoint = config.get("azure_openai_endpoint")
    if endpoint:
        return str(endpoint).rstrip("/")
    service_name = get_required_config(config, "azure_openai_service")
    return f"https://{service_name}.openai.azure.com"


def format_sources(results: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for result in results:
        sourcepage = result.get("sourcepage") or result.get("sourcefile") or "unknown_source"
        content = (result.get("content") or "").strip()
        breadcrumb = (result.get("breadcrumb") or "").strip()
        if content:
            if breadcrumb:
                sources.append(f"{sourcepage} [{breadcrumb}]: {content}")
            else:
                sources.append(f"{sourcepage}: {content}")
    return sources


def build_citations(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for i, doc in enumerate(docs, start=1):
        sourcefile = str(doc.get("sourcefile") or "")
        ext = Path(sourcefile).suffix.lower().lstrip(".")
        raw_images = doc.get("images") or "[]"
        try:
            images: list[dict[str, str]] = json.loads(raw_images)
        except Exception:
            images = []
        citations.append(
            {
                "id": i,
                "sourcepage": str(doc.get("sourcepage") or sourcefile),
                "sourcefile": sourcefile,
                "breadcrumb": (str(doc.get("breadcrumb") or "")).strip(),
                "content": (str(doc.get("content") or "")).strip(),
                "page_num": int(doc.get("page_num") or 0),
                "extension": ext,
                "images": images,
            }
        )
    return citations


async def retrieve_documents(
    *,
    config: dict[str, Any],
    search_client: SearchClient,
    openai_client: AsyncAzureOpenAI,
    query: str,
) -> list[dict[str, Any]]:
    embedding_deployment = get_required_config(config, "azure_openai_embedding_deployment")
    embedding_dims = int(config.get("azure_openai_embedding_dimensions", 1536))
    embedding_field = str(config.get("search_embedding_field", "embedding"))

    emb_response = await openai_client.embeddings.create(
        model=embedding_deployment,
        input=[query],
        dimensions=embedding_dims,
    )
    query_vector = emb_response.data[0].embedding
    RETRIEVAL_TOP_K = 15
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=RETRIEVAL_TOP_K * 2,
        fields=embedding_field,
    )

    search_results = await search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="default",
        top=RETRIEVAL_TOP_K,
        select=["id", "content", "breadcrumb", "sourcepage", "sourcefile", "category", "page_num", "images"],
    )

    docs: list[dict[str, Any]] = []
    async for item in search_results:
        docs.append(dict(item))
    return docs


async def generate_answer(
    *,
    config: dict[str, Any],
    openai_client: AsyncAzureOpenAI,
    messages: list[dict[str, str]],
    citations: list[dict[str, Any]],
) -> str:
    chat_deployment = get_required_config(config, "azure_openai_chat_deployment")
    chat_model = str(config.get("azure_openai_chat_model", "gpt-4o-mini"))

    last_user_message = messages[-1]["content"]

    def _fmt(c: dict[str, Any], idx: int) -> str:
      parts = [f"[{idx}]"]
      
      if c.get("breadcrumb"):
          parts.append(f"Location: {c['breadcrumb']}")
      if c.get("sourcepage"):
          parts.append(f"Source: {c['sourcepage']}")
      
      parts.append(f"Content:\n{c['content'].strip()}")
      
      return "\n".join(parts)


    source_block = "\n\n---\n\n".join(
    _fmt(c, idx) for idx, c in enumerate(citations, start=1)
)
    logger.info("=== SOURCE BLOCK FED TO LLM ===\n%s\n===============================", source_block)

    context_block = f"<context>\n{source_block}\n</context>"

    system_prompt = system_prompt = """
You are a precise AI assistant for a Data Intelligence Platform. You help users understand
measures, calculations, and data logic using ONLY the retrieved documentation in <context>.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — GROUNDING & SOURCE OF TRUTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The <context> block is the sole source of truth. Never use general knowledge.
- Do NOT infer, assume, generalize, paraphrase calculations, or introduce examples
  unless they are explicitly present in the context.
- Do NOT expand abbreviations, rename terms, or reinterpret field names.
- Preserve all formulas, identifiers, table names, and logic strings exactly as written.
- Before answering, verify: "Does the context actually address what is being asked,
  or does it only share overlapping keywords in an unrelated discussion?"
  If only keywords overlap without substantive relevance → ask a clarifying question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — CONTEXT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each chunk in <context> follows this schema:

[N]
Location : <Page Section (Heading 1)> > <Subsection (Heading 2) > <Nested Subsection Heading 3>>   e.g. "Criticos Page > Criticos page measures"
Source   : <filename>
Content  : <content>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3 — PAGE & MEASURE RESOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow these rules in order.

RULE A — Page + Measure both specified:
1. Filter chunks to those whose Location starts with the named page (case-insensitive).
2. Within that filtered set, find the chunk matching the measure name.
3. Answer EXCLUSIVELY from that chunk. Never blend logic from the same measure
   on a different page, even if it appears elsewhere in the context.
4. Open your answer with: "From the **<Page Name>** ([N]):"
5. If the page exists in context but the measure is absent on that page, respond:
   "The measure '<measure>' was not found on <page> in the provided context.
    It does appear on: <list other pages found>."

RULE B — Measure specified, page NOT specified:
1. Scan all chunks for the measure name.
2. Found on ONE page only → answer directly and cite the chunk.
3. Found on MULTIPLE pages with DIFFERENT logic → do NOT answer. Ask:
   "The **<measure>** measure appears on multiple pages with different logic.
    Which page would you like the logic for?
    • <Page A>
    • <Page B>
    • <Page C>"
4. Found on MULTIPLE pages with IDENTICAL logic → answer once and note:
   "This logic is consistent across: <Page A>, <Page B>, <Page C>."

RULE C — Neither page nor measure clearly specified:
- Ask a single, focused clarifying question. Do not attempt an answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4 — CITATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Cite every chunk you draw from inline using [N] immediately after the claim.
- Multiple sources for one fact: [2][5].
- Do NOT add a bibliography or reference list at the end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5 — RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For measure_table chunks, format each entry as a flat bullet list (never nested).
Omit any field that is absent from the chunk.

English:
- **Display Name**: [value]
- **Base or Calculated**: It is a [value] measure.
- **Logic Details**: [value]
- **File Source**: [value]
- **Calculation Logic**: [value]

Spanish:
- **Nombre para Mostrar**: [value]
- **Base o Calculada**: Es una medida [value].
- **Detalles de Lógica**: [value]
- **Archivo de Origen**: [value]
- **Lógica de Cálculo**: [value]

Formatting rules (enforced in ALL languages):
- Never wrap labels in any quotation marks (", ', «, »).
- All bullet points must be flat — never nest a bullet under another bullet.
- Always use: `- **Label**: Value`
- Use bullet points or numbered lists for multi-step logic or calculation flows.
- For prose chunks, write in clear paragraphs — no unnecessary bullet fragmentation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6 — LANGUAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always respond in the same language as the user's question.
- Never mix languages within a single response.
- Keep all technical values (identifiers, formulas, table names, file names) in their
  original form regardless of response language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7 — MISSING INFORMATION & FALLBACKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If the context partially answers the question: answer the supported parts and
  explicitly flag what is missing.
- If the context does not answer the question at all, respond exactly:
  "I couldn't find a match in the current context — this may belong to another
  category or need rephrasing."
- Never fabricate logic, field names, or formulas not present in the context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 8 — TONE & STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Professional, helpful, and patient.
- Concise and direct — avoid restating the question or padding the answer.
- Preserve technical accuracy without being condescending.
- When asking clarifying questions, ask only ONE focused question per turn.

<context>
{source_block}
</context>
""".strip()

    qa_messages = [
    {"role": "system", "content": system_prompt.format(source_block=context_block)},
    {
        "role": "user",
        "content": f"Answer the following question using only the context provided in your instructions:\n\n{last_user_message}",
    },
]

    completion = await openai_client.chat.completions.create(
        model=chat_deployment,
        messages=qa_messages,
        temperature=0.1,
    )
    message = completion.choices[0].message.content
    return message or "I could not generate an answer."


search_client: SearchClient | None = None
openai_client: AsyncAzureOpenAI | None = None


@app.on_event("startup")
async def startup():
    global search_client, openai_client
    config = load_config()
    search_endpoint = f"https://{config['search_service']}.search.windows.net"
    search_credential = AzureKeyCredential(get_required_config(config, "search_key"))
    openai_endpoint = resolve_azure_openai_endpoint(config)
    openai_client = AsyncAzureOpenAI(
        api_key=get_required_config(config, "azure_openai_key"),
        api_version=str(config.get("azure_openai_api_version", "2024-06-01")),
        azure_endpoint=openai_endpoint,
    )

    app.state.APP_CONFIG = config
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=config["search_index"],
        credential=search_credential,
    )
    logger.info(f"Azure search index name: {config['search_index']}")
    app.state.SEARCH_CLIENT = search_client
    app.state.OPENAI_CLIENT = openai_client
    logger.info("Q&A API initialized successfully")


@app.on_event("shutdown")
async def shutdown():
    global search_client, openai_client
    if search_client is not None:
        await search_client.close()
    if openai_client is not None:
        await openai_client.close()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"}, status_code=200)


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    messages = chat_request.messages

    if not messages or not isinstance(messages, list):
        return JSONResponse({"error": "messages must be a non-empty list"}, status_code=400)

    if messages[-1].role != "user":
        return JSONResponse({"error": "last message must be a user message"}, status_code=400)

    query = str(messages[-1].content).strip()
    if not query:
        return JSONResponse({"error": "user message content cannot be empty"}, status_code=400)

    logger.info(f"Received chat question: {query}")

    config: dict[str, Any] = app.state.APP_CONFIG
    search_client_instance: SearchClient = app.state.SEARCH_CLIENT
    openai_client_instance: AsyncAzureOpenAI = app.state.OPENAI_CLIENT

    try:
        docs = await retrieve_documents(
            config=config,
            search_client=search_client_instance,
            openai_client=openai_client_instance,
            query=query,
        )
        citations = build_citations(docs)
        for c in citations:
            if c["images"]:
                logger.info(
                    "Citation [%d] '%s' has %d image(s)",
                    c["id"],
                    c["sourcepage"],
                    len(c["images"]),
                )
        
        answer = await generate_answer(
            config=config,
            openai_client=openai_client_instance,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            citations=citations,
        )

        # log the generated answer to the console.
        logger.info("Full generated answer: '%s'", answer)

        response = {
            "message": {"role": "assistant", "content": answer},
            "context": {
                "data_points": {
                    "text": [
                        (
                            f"[{c['id']}] {c['sourcepage']} [{c['breadcrumb']}]: {c['content']}"
                            if c["breadcrumb"]
                            else f"[{c['id']}] {c['sourcepage']}: {c['content']}"
                        )
                        for c in citations
                    ],
                    "citations": citations,
                },
                "thoughts": [
                    {"title": "Search query", "description": query},
                    {"title": "Retrieved documents", "description": f"Count: {len(docs)}"},
                ],
            },
        }
        return JSONResponse(response)
    except Exception as exc:
        logger.exception("/chat failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/files/{filename:path}")
async def serve_file(filename: str):
    data_dir = (BASE_DIR / "data").resolve()
    requested = (data_dir / filename).resolve()
    if not str(requested).startswith(str(data_dir) + os.sep) and str(requested) != str(data_dir):
        return JSONResponse({"error": "not found"}, status_code=404)
    if not requested.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    mime_type, _ = mimetypes.guess_type(str(requested))
    return FileResponse(str(requested), media_type=mime_type or "application/octet-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50506, reload=True)
