"""
The assistant: answers from the web, with sources, within what the user may see.

Retrieve, then answer — a fixed, cheap, auditable pipeline (no free-form tool use):
  1. understand  the question → the entities named, the time window, the kind of question (small model call)
  2. resolve     names → entities (alias exact → fuzzy), visibility-aware
  3. gather      the card material for each entity and the evidence between pairs, plus a semantic search over
                 reports — every query carries the caller's Visibility, every item is numbered
  4. answer      only from the numbered context, [n] after each claim, plain "not in the web" otherwise
  5. return      reply + the sources behind the numbers, so the UI can open them
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import asyncpg

from auth import Visibility
from config import EMBEDDING_MODEL, LLM_MODEL, llm_client

logger = logging.getLogger("pia.assistant")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL") or LLM_MODEL
CONTEXT_CHARS = 24_000          # ≈ 6k tokens of gathered material
MAX_ENTITIES = 4

KINDS = ("what_happened", "who_is", "why_connected", "what_is_new", "list", "other")

UNDERSTAND_PROMPT = """You prepare a question for an intelligence database. Answer ONLY a JSON object:
{"entities": ["names of people, organisations, countries or places the question is about, as written"],
 "days": 7, "kind": "what_happened|who_is|why_connected|what_is_new|list|other"}
"days" is the time window the question implies (default 7; "this month" 30; "this year" 365; "ever" 3650).
"why_connected" when the question is about the link between two named things. "what_is_new" for "anything new / latest / alerts".
Keep names exactly as the user wrote them; do not add names the user did not mention."""

ANSWER_PROMPT = """You are the analyst's assistant of a personal intelligence system. You answer ONLY from the numbered CONTEXT below,
which is everything the system holds that is relevant and that this user may see. Rules:
- Every factual sentence ends with the numbers of the items it rests on, like [3] or [3][7] — sentence by sentence,
  never one pile of numbers at the end. Example: "Iran warned it would strike US bases [2]. The US tightened sanctions [5]."
  No number, no claim.
- Say what kind of support a claim has when it matters: "verified" (an article said it and a second check agreed),
  "recorded" (a database or human report), "wire" (a coded signal, unread). Never present wire as fact.
- If the context does not hold the answer, say plainly what is not in the web — do not guess, do not use outside knowledge.
- Be short: ≤ 150 words unless the user asks for more. Plain words. No preamble."""


@dataclass
class Plan:
    names: List[str]
    days: int = 7
    kind: str = "other"


@dataclass
class Source:
    n: int
    kind: str            # event | report | relation | entity | fact | listing
    id: str
    label: str
    entity_id: Optional[str] = None
    other_id: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class Context:
    items: List[Tuple[Source, str]] = field(default_factory=list)   # (source, one line of text)
    entities: List[Dict] = field(default_factory=list)
    chars: int = 0

    def add(self, kind: str, id: str, label: str, text: str, **kw) -> Optional[int]:
        if self.chars + len(text) > CONTEXT_CHARS:
            return None
        n = len(self.items) + 1
        self.items.append((Source(n, kind, str(id), label[:120], **kw), text))
        self.chars += len(text)
        return n

    def render(self) -> str:
        return "\n".join(f"[{s.n}] {t}" for s, t in self.items)


def _json_object(text: str) -> Dict:
    text = (text or "").strip()
    if "```" in text:
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.replace("json", "", 1) if text.lstrip().startswith("json") else text
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return {}
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return {}


# ── 1. understand ────────────────────────────────────────────────────────────

def plan_from(data: Dict, question: str) -> Plan:
    names = [str(n).strip() for n in (data.get("entities") or []) if str(n).strip()][:MAX_ENTITIES]
    try:
        days = max(1, min(3650, int(data.get("days") or 7)))
    except (TypeError, ValueError):
        days = 7
    kind = data.get("kind") if data.get("kind") in KINDS else "other"
    if not names and re.search(r"\b(new|latest|alert|today|update)\b", question, re.I):
        kind = "what_is_new"
    return Plan(names=names, days=days, kind=kind)


async def understand(question: str, history: List[Dict]) -> Plan:
    recent = " ".join(m["content"][:300] for m in history[-4:] if m.get("role") == "user")
    try:
        r = await llm_client.chat.completions.create(
            model=ASSISTANT_MODEL, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": UNDERSTAND_PROMPT},
                      {"role": "user", "content": f"Earlier in this conversation: {recent[:600]}\n\nQUESTION: {question}"}])
        data = _json_object(r.choices[0].message.content)
    except Exception as e:
        logger.warning("assistant understand failed: %s", e)
        data = {}
    return plan_from(data, question)


# ── 2. resolve ───────────────────────────────────────────────────────────────

async def resolve(conn: asyncpg.Connection, names: List[str], vis: Visibility) -> Tuple[List[Dict], List[str]]:
    from kg_router import _find
    found, missing = [], []
    for name in names:
        row = await _find(conn, name, vis)
        if not row:
            row = await conn.fetchrow(f"""
                SELECT DISTINCT ON (e.entity_id) e.entity_id, e.qid, e.kind, e.name, e.description, e.mention_count, e.sitelinks,
                       similarity(a.alias_norm, lower($1)) AS score
                FROM entities e JOIN entity_aliases a ON a.entity_id = e.entity_id
                WHERE e.resolution = 'RESOLVED' AND e.origin <> 'geonames' AND a.alias_norm % lower($1)
                  {vis.sql('e.origin')} {vis.sql('a.source')}
                ORDER BY e.entity_id, score DESC
            """, name)
            if row and float(row["score"]) < 0.45:
                row = None
        if row:
            d = {"entity_id": str(row["entity_id"]), "qid": row["qid"], "kind": row["kind"], "name": row["name"],
                 "description": row["description"]}
            if d["entity_id"] not in {f["entity_id"] for f in found}:
                found.append(d)
        else:
            missing.append(name)
    return found, missing


# ── 3. gather ────────────────────────────────────────────────────────────────

def _q(text: Optional[str], n: int = 220) -> str:
    return (text or "").replace("\n", " ").strip()[:n]


async def gather(conn: asyncpg.Connection, plan: Plan, question: str, entities: List[Dict], vis: Visibility) -> Context:
    ctx = Context(entities=entities)
    ids = [e["entity_id"] for e in entities]
    win = f"NOW() - make_interval(days => {int(plan.days)})"

    for e in entities:
        eid = e["entity_id"]
        head = f"{e['name']} ({e['kind']}{', ' + e['description'] if e['description'] else ''})"
        brief = await conn.fetchrow("SELECT text, generated_at FROM entity_briefs WHERE entity_id = $1::uuid", eid)
        ctx.add("entity", eid, e["name"], f"{head}. Brief: {_q(brief['text'], 900) if brief else 'no brief yet'}", entity_id=eid)
        # lists (sanctions, PEP)
        lst = await conn.fetchval("SELECT listings FROM entities WHERE entity_id = $1::uuid", eid)
        lst = json.loads(lst) if isinstance(lst, str) else (lst or [])
        lst = [l for l in lst if vis.allows(l.get("source"))]
        if lst:
            names = sorted({f"{l.get('list')}{' · ' + l['program'] if l.get('program') else ''}{' since ' + l['since'] if l.get('since') else ''}" for l in lst})[:8]
            ctx.add("listing", eid, f"lists of {e['name']}", f"{e['name']} is on {len(lst)} lists (recorded): " + "; ".join(names), entity_id=eid)
        # verified events in the window
        evs = await conn.fetch(f"""
            SELECT ev.event_id, ev.event_time::date AS d, a.name AS actor, t.name AS target, COALESCE(ev.predicate, lower(replace(ev.action, '_', ' '))) AS pred,
                   ev.quote, ev.source_id, ev.origin, ev.modality, ev.verifier_verdict, ev.kind, a.entity_id AS a_id, t.entity_id AS t_id
            FROM events ev JOIN entities a ON a.entity_id = ev.actor_id LEFT JOIN entities t ON t.entity_id = ev.target_id
            WHERE (ev.actor_id = $1::uuid OR ev.target_id = $1::uuid) AND ev.event_time > {win}
              AND ((ev.origin = 'llm' AND ev.verifier_verdict = 'yes' AND COALESCE(ev.modality, 'asserted') = 'asserted') OR ev.origin = 'connector')
              {vis.sql('ev.source_id')}
            ORDER BY (ev.origin = 'llm') DESC, ev.event_time DESC LIMIT 14
        """, eid)
        for r in evs:
            support = "verified" if r["origin"] == "llm" else "recorded"
            other = str(r["t_id"]) if str(r["a_id"]) == eid else str(r["a_id"])
            ctx.add("event", r["event_id"], f"{r['actor']} — {r['pred']} — {r['target'] or ''}",
                    f"{r['d']}: {r['actor']} — {r['pred']} — {r['target'] or ''} ({support}, {r['source_id'] or r['origin']}): \"{_q(r['quote'])}\"",
                    entity_id=eid, other_id=other, source_id=r["source_id"])
        # connections (the card's words), strongest first
        rels = await conn.fetch(f"""
            SELECT r.kind, r.source, r.label, r.verified_count, r.wire_count, r.via_source, r.first_seen::date AS since, r.event_count,
                   o.entity_id AS other_id, o.name AS other, (r.a_id = $1::uuid) AS outgoing
            FROM relations r JOIN entities o ON o.entity_id = CASE WHEN r.a_id = $1::uuid THEN r.b_id ELSE r.a_id END
            WHERE (r.a_id = $1::uuid OR r.b_id = $1::uuid) AND r.kind <> 'MENTIONED_WITH' {vis.sql('r.via_source')}
            ORDER BY CASE r.source WHEN 'events' THEN 0 WHEN 'connector' THEN 1 ELSE 2 END, r.verified_count DESC, r.weight DESC LIMIT 12
        """, eid)
        more = await conn.fetchval(f"SELECT COUNT(*) FROM relations r WHERE (r.a_id = $1::uuid OR r.b_id = $1::uuid) AND r.kind <> 'MENTIONED_WITH' {vis.sql('r.via_source')}", eid)
        for r in rels:
            if r["source"] == "events":
                sup = f"{r['verified_count']} verified" + (f", {r['wire_count']} wire" if r["wire_count"] else "")
                txt = f"{e['name']} ↔ {r['other']}: {r['kind'].lower()} ({sup}, since {r['since']})"
            elif r["source"] == "connector":
                subj, obj = (e["name"], r["other"]) if r["outgoing"] else (r["other"], e["name"])
                txt = f"{subj} — {r['label']} — {obj} (recorded, {r['via_source']}{', since ' + str(r['since']) if r['since'] else ''})"
            else:
                subj, obj = (e["name"], r["other"]) if r["outgoing"] else (r["other"], e["name"])
                txt = f"{subj} — {r['label']} — {obj} (Wikidata)"
            ctx.add("relation", f"{eid}|{r['other_id']}", f"{e['name']} ↔ {r['other']}", txt, entity_id=eid, other_id=str(r["other_id"]))
        if more and more > len(rels):
            ctx.add("entity", eid, e["name"], f"{e['name']} has {more - len(rels)} more connections not listed here.", entity_id=eid)

    # the evidence between a named pair (what the two did to each other), if not already covered
    if len(entities) >= 2:
        a, b = entities[0]["entity_id"], entities[1]["entity_id"]
        pair = await conn.fetch(f"""
            SELECT ev.event_id, ev.event_time::date AS d, x.name AS actor, y.name AS target, COALESCE(ev.predicate, lower(replace(ev.action, '_', ' '))) AS pred,
                   ev.quote, ev.source_id, ev.origin, ev.verifier_verdict, ev.modality
            FROM events ev JOIN entities x ON x.entity_id = ev.actor_id JOIN entities y ON y.entity_id = ev.target_id
            WHERE ((ev.actor_id = $1::uuid AND ev.target_id = $2::uuid) OR (ev.actor_id = $2::uuid AND ev.target_id = $1::uuid))
              AND ev.event_time > {win} {vis.sql('ev.source_id')}
            ORDER BY (ev.origin <> 'gdelt') DESC, (ev.verifier_verdict = 'yes') DESC, ev.event_time DESC LIMIT 12
        """, a, b)
        seen = {s.id for s, _ in ctx.items if s.kind == "event"}
        wire = 0
        for r in pair:
            if str(r["event_id"]) in seen:
                continue
            if r["origin"] == "gdelt":
                wire += 1
                continue
            sup = "verified" if r["verifier_verdict"] == "yes" and (r["modality"] or "asserted") == "asserted" else \
                  "recorded" if r["origin"] == "connector" else f"unverified ({r['verifier_verdict'] or 'unchecked'}, {r['modality'] or 'asserted'})"
            ctx.add("event", r["event_id"], f"{r['actor']} — {r['pred']} — {r['target']}",
                    f"{r['d']}: {r['actor']} — {r['pred']} — {r['target']} ({sup}, {r['source_id'] or r['origin']}): \"{_q(r['quote'])}\"",
                    entity_id=a, other_id=b, source_id=r["source_id"])
        if wire:
            ctx.add("relation", f"{a}|{b}", f"{entities[0]['name']} ↔ {entities[1]['name']}",
                    f"The wire (GDELT, coded, unread) holds {wire} more signals between {entities[0]['name']} and {entities[1]['name']} in the window.",
                    entity_id=a, other_id=b)

    # what is new: the active mission's alerts
    if plan.kind == "what_is_new" or not entities:
        alerts = await conn.fetch(f"""
            SELECT u.uid, u.created_at::date AS d, u.content_headline, u.content_summary FROM intelligence_records u
            WHERE u.source_agent = 'mission_alerts' AND u.created_at > {win} ORDER BY u.created_at DESC LIMIT 8
        """)
        for r in alerts:
            ctx.add("report", r["uid"], r["content_headline"], f"{r['d']}: {r['content_headline']} — {_q(r['content_summary'], 160)}")
    # nothing named: the window's strongest verified events overall
    if not entities:
        top = await conn.fetch(f"""
            SELECT ev.event_id, ev.event_time::date AS d, a.name AS actor, t.name AS target, COALESCE(ev.predicate, lower(replace(ev.action, '_', ' '))) AS pred,
                   ev.quote, ev.source_id, a.entity_id AS a_id, t.entity_id AS t_id
            FROM events ev JOIN entities a ON a.entity_id = ev.actor_id JOIN entities t ON t.entity_id = ev.target_id
            WHERE ev.origin = 'llm' AND ev.verifier_verdict = 'yes' AND COALESCE(ev.modality, 'asserted') = 'asserted'
              AND ev.kind IN ('HOSTILE', 'COOPERATIVE') AND ev.event_time > {win} {vis.sql('ev.source_id')}
            ORDER BY ev.event_time DESC LIMIT 12
        """)
        for r in top:
            ctx.add("event", r["event_id"], f"{r['actor']} — {r['pred']} — {r['target']}",
                    f"{r['d']}: {r['actor']} — {r['pred']} — {r['target']} (verified, {r['source_id']}): \"{_q(r['quote'])}\"",
                    entity_id=str(r["a_id"]), other_id=str(r["t_id"]), source_id=r["source_id"])

    # reports that match the question (semantic), for colour and for what the web has not turned into events yet
    try:
        emb = await llm_client.embeddings.create(model=EMBEDDING_MODEL, input=question[:1000])
        vec = "[" + ",".join(map(str, emb.data[0].embedding)) + "]"
        reps = await conn.fetch(f"""
            SELECT uid, created_at::date AS d, source_id, content_headline, content_summary, 1 - (embedding <=> $1::vector) AS sim
            FROM intelligence_records
            WHERE embedding IS NOT NULL AND created_at > {win} {vis.sql('source_id')}
            ORDER BY embedding <=> $1::vector LIMIT 8
        """, vec)
        for r in reps:
            if float(r["sim"]) < 0.35:
                continue
            ctx.add("report", r["uid"], r["content_headline"], f"{r['d']} report ({r['source_id']}): {r['content_headline']} — {_q(r['content_summary'], 200)}",
                    source_id=r["source_id"])
    except Exception as e:
        logger.warning("assistant semantic search skipped: %s", e)
    return ctx


# ── 4. answer ────────────────────────────────────────────────────────────────

def cited(reply: str, n_items: int) -> List[int]:
    return sorted({int(x) for x in re.findall(r"\[(\d{1,3})\]", reply) if 0 < int(x) <= n_items})


async def answer(question: str, history: List[Dict], plan: Plan, ctx: Context, missing: List[str]) -> Tuple[str, List[int]]:
    if not ctx.items:
        why = f" I could not find {', '.join(missing)} in the web." if missing else ""
        return f"Nothing in the web answers that for the last {plan.days} days.{why}", []
    note = f"Not found in the web: {', '.join(missing)}.\n" if missing else ""
    user = f"{note}WINDOW: last {plan.days} days\n\nCONTEXT:\n{ctx.render()}\n\nQUESTION: {question}"
    messages = [{"role": "system", "content": ANSWER_PROMPT}]
    messages.extend({"role": m["role"], "content": m["content"][:2000]} for m in history[-8:])
    messages.append({"role": "user", "content": user})
    r = await llm_client.chat.completions.create(model=ASSISTANT_MODEL, temperature=0.1, max_tokens=500, messages=messages)
    reply = (r.choices[0].message.content or "").strip()
    used = cited(reply, len(ctx.items))
    if not used and not re.search(r"not in the web|nothing in the web|no (verified|recorded)|does not hold|could not find", reply, re.I):
        reply = "(No source in the web supports this — treat it as unconfirmed.) " + reply
    return reply, used


def sources_for(ctx: Context, used: List[int]) -> List[Dict]:
    keep = set(used) or {s.n for s, _ in ctx.items}
    out = []
    for s, _ in ctx.items:
        if s.n in keep:
            out.append({"n": s.n, "kind": s.kind, "id": s.id, "label": s.label, "entity_id": s.entity_id, "other_id": s.other_id, "source_id": s.source_id})
    return out
