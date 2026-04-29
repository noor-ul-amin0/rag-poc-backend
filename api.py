import logging
import mimetypes
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from pathlib import Path
import json
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse, StreamingResponse
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qa-api")
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    search_endpoint = f"https://{config['search_service']}.search.windows.net"
    search_credential = AzureKeyCredential(get_required_config(config, "search_key"))
    openai_endpoint = resolve_azure_openai_endpoint(config)
    openai_client = AsyncAzureOpenAI(
        api_key=get_required_config(config, "azure_openai_key"),
        api_version=str(config.get("azure_openai_api_version", "2024-06-01")),
        azure_endpoint=openai_endpoint,
    )
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=config["search_index"],
        credential=search_credential,
    )

    app.state.APP_CONFIG = config
    app.state.SEARCH_CLIENT = search_client
    app.state.OPENAI_CLIENT = openai_client
    logger.info(f"Azure search index name: {config['search_index']}")
    logger.info("Q&A API initialized successfully")

    try:
        yield
    finally:
        await search_client.close()
        await openai_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRIEVAL_TOP_K = 15


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


def load_config() -> dict[str, Any]:
    return get_settings().model_dump()


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


async def rewrite_query_for_retrieval(
    *,
    config: dict[str, Any],
    openai_client: AsyncAzureOpenAI,
    messages: list[dict[str, str]],
) -> str:
    latest_user_message = (messages[-1].get("content") or "").strip()
    if not latest_user_message:
        return latest_user_message

    chat_deployment = get_required_config(config, "azure_openai_chat_deployment")
    recent_messages = messages[-8:]
    history_block = "\n".join(
        f"{m.get('role', 'user').upper()}: {(m.get('content') or '').strip()}" for m in recent_messages
    )

    rewrite_system_prompt = """
You are a retrieval query optimizer for a domain-specific RAG system.
Your job is to rewrite the latest user message into the best possible standalone search query for
vector/semantic search over documentation chunks.

You must do ALL of the following in one rewrite:

1. RESOLVE CONTEXT — If the message references prior turns (e.g. "for both", "that one", "same",
   "which page?", pronouns, or the assistant's clarifying options), resolve those references using
   the conversation history so the query is fully self-contained.

2. IMPROVE CLARITY — Expand vague or abbreviated phrasing into specific, information-rich language.
   For example: "onhand?" → "What is the Onhand measure and how is it calculated?"

Hard constraints:
- DO NOT change the user's intent.
- DO NOT answer the question.
- DO NOT add facts or topics not implied by the conversation.
- Return ONLY the rewritten query text — no explanation, no prefix, no quotes.
""".strip()

    rewrite_user_prompt = (
        "Conversation (most recent last):\n"
        f"{history_block}\n\n"
        "Rewrite the latest USER message into an optimized standalone retrieval query."
    )

    try:
        completion = await openai_client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {"role": "system", "content": rewrite_system_prompt},
                {"role": "user", "content": rewrite_user_prompt},
            ],
            temperature=0,
            max_tokens=128,
        )
        rewritten = (completion.choices[0].message.content or "").strip().strip('"')
        if not rewritten:
            return latest_user_message
        logger.info(
            "Retrieval query rewrite | original='%s' | rewritten='%s'",
            latest_user_message,
            rewritten,
        )
        return rewritten
    except Exception:
        logger.exception("Query rewrite failed, falling back to latest user message")
        return latest_user_message


async def generate_answer(
    *,
    config: dict[str, Any],
    openai_client: AsyncAzureOpenAI,
    messages: list[dict[str, str]],
    citations: list[dict[str, Any]],
    resolved_question: str,
) -> str:
    chat_deployment = get_required_config(config, "azure_openai_chat_deployment")
    chat_model = str(config.get("azure_openai_chat_model", "gpt-4o-mini"))

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

    qa_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt.format(source_block=context_block)},
        {
            "role": "user",
            "content": (
                "Answer the following question using only the context provided in your instructions.\n\n"
                f"{resolved_question}"
            ),
        },
    ]

    completion = await openai_client.chat.completions.create(
        model=chat_deployment,
        messages=qa_messages,
        temperature=0.1,
    )
    message = completion.choices[0].message.content
    return message or "I could not generate an answer."


async def generate_answer_stream(
    *,
    config: dict[str, Any],
    openai_client: AsyncAzureOpenAI,
    messages: list[dict[str, str]],
    citations: list[dict[str, Any]],
    resolved_question: str,
) -> AsyncIterator[str]:
    chat_deployment = get_required_config(config, "azure_openai_chat_deployment")

    def _fmt(c: dict[str, Any], idx: int) -> str:
        parts = [f"[{idx}]"]
        if c.get("breadcrumb"):
            parts.append(f"Location: {c['breadcrumb']}")
        if c.get("sourcepage"):
            parts.append(f"Source: {c['sourcepage']}")
        parts.append(f"Content:\n{c['content'].strip()}")
        return "\n".join(parts)

    source_block = "\n\n---\n\n".join(_fmt(c, idx) for idx, c in enumerate(citations, start=1))
    context_block = f"<context>\n{source_block}\n</context>"

    system_prompt = """
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

    qa_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt.format(source_block=context_block)},
        {
            "role": "user",
            "content": (
                "Answer the following question using only the context provided in your instructions.\n\n"
                f"{resolved_question}"
            ),
        },
    ]

    stream = await openai_client.chat.completions.create(
        model=chat_deployment,
        messages=qa_messages,
        temperature=0.1,
        stream=True,
    )

    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def build_chat_response(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    query: str,
    retrieval_query: str,
    docs_count: int,
) -> dict[str, Any]:
    return {
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
                {"title": "User query", "description": query},
                {"title": "Retrieval query", "description": retrieval_query},
                {"title": "Retrieved documents", "description": f"Count: {docs_count}"},
            ],
        },
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"}, status_code=200)


@app.post("/chat")
async def chat(chat_request: ChatRequest) -> JSONResponse:
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
    message_payload = [{"role": m.role, "content": m.content} for m in messages]

    try:
        retrieval_query = await rewrite_query_for_retrieval(
            config=config,
            openai_client=openai_client_instance,
            messages=message_payload,
        )
        docs = await retrieve_documents(
            config=config,
            search_client=search_client_instance,
            openai_client=openai_client_instance,
            query=retrieval_query,
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
            messages=message_payload,
            citations=citations,
            resolved_question=retrieval_query,
        )

        # log the generated answer to the console.
        logger.info("Full generated answer: '%s'", answer)

        response = build_chat_response(
            answer=answer,
            citations=citations,
            query=query,
            retrieval_query=retrieval_query,
            docs_count=len(docs),
        )
        return JSONResponse(response)
    except Exception as exc:
        logger.exception("/chat failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/chat/stream", response_model=None)
async def chat_stream(chat_request: ChatRequest) -> Response:
    messages = chat_request.messages

    if not messages or not isinstance(messages, list):
        return JSONResponse({"error": "messages must be a non-empty list"}, status_code=400)

    if messages[-1].role != "user":
        return JSONResponse({"error": "last message must be a user message"}, status_code=400)

    query = str(messages[-1].content).strip()
    if not query:
        return JSONResponse({"error": "user message content cannot be empty"}, status_code=400)

    logger.info(f"Received streaming chat question: {query}")

    config: dict[str, Any] = app.state.APP_CONFIG
    search_client_instance: SearchClient = app.state.SEARCH_CLIENT
    openai_client_instance: AsyncAzureOpenAI = app.state.OPENAI_CLIENT
    message_payload = [{"role": m.role, "content": m.content} for m in messages]

    async def stream() -> AsyncIterator[bytes]:
        try:
            retrieval_query = await rewrite_query_for_retrieval(
                config=config,
                openai_client=openai_client_instance,
                messages=message_payload,
            )
            docs = await retrieve_documents(
                config=config,
                search_client=search_client_instance,
                openai_client=openai_client_instance,
                query=retrieval_query,
            )
            citations = build_citations(docs)

            full_answer_parts: list[str] = []
            async for delta in generate_answer_stream(
                config=config,
                openai_client=openai_client_instance,
                messages=message_payload,
                citations=citations,
                resolved_question=retrieval_query,
            ):
                full_answer_parts.append(delta)
                yield (json.dumps({"type": "delta", "content": delta}) + "\n").encode("utf-8")

            answer = "".join(full_answer_parts).strip() or "I could not generate an answer."
            logger.info("Full generated streaming answer: '%s'", answer)
            response_payload = build_chat_response(
                answer=answer,
                citations=citations,
                query=query,
                retrieval_query=retrieval_query,
                docs_count=len(docs),
            )
            yield (json.dumps({"type": "done", "response": response_payload}) + "\n").encode("utf-8")
        except Exception as exc:
            logger.exception("/chat/stream failed")
            yield (json.dumps({"type": "error", "error": str(exc)}) + "\n").encode("utf-8")

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
