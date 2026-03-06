// === server.js ===
// Node 22+, Express 5+, ES Module-kompatibel

import express from "express";
import path from "path";
import cors from "cors";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import OpenAI from "openai";

dotenv.config();

// === Setup ===
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// === Middlewares ===
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// 🔒 ETag komplett deaktivieren (Chrome-/Render-Cache-Bremse)
app.disable("etag");

// 🔒 Frontend statisch ausliefern – OHNE Caching (Museumssicher)
app.use(
  express.static(path.join(__dirname, "frontend"), {
    etag: false,
    lastModified: false,
    maxAge: 0,
    setHeaders: (res) => {
      res.setHeader(
        "Cache-Control",
        "no-store, no-cache, must-revalidate, proxy-revalidate"
      );
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");
    },
  })
);

// === OpenAI-Client ===
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

if (!process.env.OPENAI_API_KEY) {
  console.warn("⚠️ OPENAI_API_KEY fehlt.");
}

/* ===============================
   GLOBAL CONSTANTS
================================ */

const OPENAI_TIMEOUT_MS = 18000;
const ROUTE_TIMEOUT_MS = 22000;

/* ===============================
   GENERIC HELPERS
================================ */

function makeRequestId() {
  return `req_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
}

function withTimeout(promise, ms, label = "timeout") {
  let timer = null;

  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(label));
    }, ms);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

function safeAnswerFallback(lang = "de") {
  return lang === "en"
    ? "Forgive me, I need a brief moment longer. Please ask me once more."
    : "Verzeiht, ich brauche einen kurzen Augenblick länger. Bitte fragt mich noch einmal.";
}

// garantiert saubere JSON-Antwort
function sendJson(res, payload, status = 200) {
  if (res.headersSent) return;
  res.status(status).json(payload);
}

/* ===============================
   DIALOG-CONTEXT (SESSION HISTORY)
   - optional: works only if client sends conversationId
   - TTL + size limits (museum-safe)
================================ */

const CONV_STORE = new Map(); // id -> { ts, lastLang, msgs: [{role, content}] }

const CONV_TTL_MS = 20 * 60 * 1000; // 20 min
const CONV_MAX_MSGS = 12; // max messages (user+assistant), e.g. 6 turns
const CONV_ID_MAXLEN = 80;

function convNow() {
  return Date.now();
}

function normalizeConvId(raw) {
  if (typeof raw !== "string") return null;
  const id = raw.trim();
  if (!id) return null;
  return id.slice(0, CONV_ID_MAXLEN);
}

function convPruneExpired() {
  const now = convNow();
  for (const [id, c] of CONV_STORE.entries()) {
    if (!c || !c.ts || now - c.ts > CONV_TTL_MS) {
      CONV_STORE.delete(id);
    }
  }
}

function convGet(id) {
  if (!id) return null;
  convPruneExpired();

  const key = String(id).trim();
  if (!key) return null;

  let c = CONV_STORE.get(key);
  if (!c) {
    c = { ts: convNow(), lastLang: null, msgs: [] };
    CONV_STORE.set(key, c);
  } else {
    c.ts = convNow();
  }
  return c;
}

function convPush(id, role, content) {
  const c = convGet(id);
  if (!c) return;

  const txt = String(content || "").trim();
  if (!txt) return;

  c.msgs.push({ role, content: txt });

  if (c.msgs.length > CONV_MAX_MSGS) {
    c.msgs = c.msgs.slice(c.msgs.length - CONV_MAX_MSGS);
  }

  c.ts = convNow();
}

function convGetHistoryMsgs(id) {
  const c = convGet(id);
  if (!c) return [];
  return Array.isArray(c.msgs) ? [...c.msgs] : [];
}

function convEnd(id) {
  const key = normalizeConvId(id);
  if (!key) return;
  CONV_STORE.delete(key);
}

// --- Helpers ---
function normalizeLang(raw) {
  const s = String(raw || "").toLowerCase();
  if (s.startsWith("en")) return "en";
  return "de";
}

function endsWithSentence(text) {
  const t = String(text || "").trim();
  return /[.!?]["')\]]?\s*$/.test(t);
}

function forceSentenceEnd(text) {
  let t = String(text || "").trim();
  if (!t) return t;
  if (endsWithSentence(t)) return t;
  return t + ".";
}

// ✅ Sonderregel: "Aufwecken"-Fragen (DE/EN) -> feste Antwort ohne OpenAI-Call
function isWakeQuestion(text) {
  const t0 = String(text || "").trim();
  if (!t0) return false;

  const t = t0
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return false;

  const hasWer = /\bwer\b/.test(t);
  const hasWie = /\bwie\b/.test(t);
  const hasWarum = /\b(warum|wieso|weshalb)\b/.test(t);
  const hasDich = /\bdich\b/.test(t);
  const hasDu = /\bdu\b/.test(t);

  const hasGeweckt = /\b(auf\s*)?geweck?t\b/.test(t);
  const hasAufgewacht = /\bauf\s*gewacht\b/.test(t);
  const hasErwacht = /\berwacht\b/.test(t);

  const deWho = hasWer && hasDich && hasGeweckt;
  const deHow = hasWie && hasDu && (hasAufgewacht || hasErwacht);
  const deWhy = hasWarum && hasDu && (hasAufgewacht || hasErwacht);

  if (deWho || deHow || deWhy) return true;

  const hasWho = /\bwho\b/.test(t);
  const hasHow = /\bhow\b/.test(t);
  const hasWhy = /\bwhy\b/.test(t);
  const hasYou = /\byou\b/.test(t);

  const hasWoke = /\b(woke|woken|awoke|awakened|awaken)\b/.test(t);
  const hasWakeUp = /\bwake\s*up\b/.test(t) || /\bwoke\s*up\b/.test(t);

  const enWho = hasWho && hasYou && (hasWoke || hasWakeUp);
  const enHow =
    hasHow &&
    hasYou &&
    (hasWakeUp || /\bawaken(ed)?\b/.test(t) || /\bawoke\b/.test(t));
  const enWhy =
    hasWhy &&
    hasYou &&
    (hasWakeUp || /\bawaken(ed)?\b/.test(t) || /\bawoke\b/.test(t));

  return enWho || enHow || enWhy;
}

// ✅ Sprache für Wake-Fragen robust bestimmen
function detectWakeLang(text, fallbackLang = "de") {
  const t = String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return fallbackLang;

  const enMarkers =
    /\bwho\b/.test(t) ||
    /\bhow\b/.test(t) ||
    /\bwhy\b/.test(t) ||
    /\byou\b/.test(t) ||
    /\b(woke|woken|awoke|awakened|awaken)\b/.test(t) ||
    /\b(wake\s*up|woke\s*up)\b/.test(t);

  if (enMarkers) return "en";
  return "de";
}

function wakeAnswer(lang) {
  if (lang === "en") {
    return "Christoph Dammann woke me up - I am grateful. Now I am glad to work here as an avatar in the city museum, ready for your questions.";
  }

  return "Mich hat Christoph Dammann aufgeweckt – ich danke ihm dafür. Nun arbeite ich gern als Avatar im Stadtmuseum und bin für Eure Fragen bereit.";
}

// ✅ Sprache primär über Satzanfang bestimmen
function detectLanguageServer(text) {
  const t0 = String(text || "").trim();
  if (!t0) return null;

  const t = t0
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return null;

  if (/[äöüß]/.test(t0)) return "de";

  if (/^(was|wer|wen|wem|wessen|wie|wo|wohin|woher|wann|warum|wieso|weshalb)\b/.test(t)) return "de";
  if (/^(what|why|where|when|who|whom|whose|which|how)\b/.test(t)) return "en";

  if (/^(erzähl|erzaehl|sage|sag|nenn|nenne|erkläre|erklaere|beschreibe|zeige|sprich)\b/.test(t)) return "de";
  if (/^(tell|say|name|explain|describe|show|speak)\b/.test(t)) return "en";

  return null;
}

// ✅ Subjektive / in-character Fragen erkennen
function isSubjectiveInCharacterQuestion(text, lang) {
  const t = String(text || "").toLowerCase().trim();
  if (!t) return false;

  if (lang === "de") {
    return (
      /\b(am liebsten|liebst(en)?|lieblings|schmeckt|mochtest|möchtest|magst|liebt|liebe)\b/.test(t) ||
      /\b(hast du|habt ihr)\b/.test(t) ||
      /\b(gegessen|getrunken|genossen|gejagt)\b/.test(t)
    );
  }

  return (
    /\b(favorite|favourite|like most|liked most|prefer|enjoyed|tasted)\b/.test(t) ||
    /\b(did you like|have you ever)\b/.test(t) ||
    /\b(meat|food|drink)\b/.test(t)
  );
}

/* ===============================
   END-OF-CONVERSATION DETECTION
================================ */

function normalizeForIntent(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function userEndedConversation(text, lang) {
  const t = normalizeForIntent(text);
  if (!t) return false;

  if (lang === "en") {
    return (
      /\b(thanks|thank you|thx|cheers)\b/.test(t) ||
      /\b(bye|goodbye|see you|farewell)\b/.test(t) ||
      /\b(that s all|that is all|that was all|no more questions|i m done|im done|we re done|we are done)\b/.test(t) ||
      /\b(stop|end (the )?conversation|finish)\b/.test(t)
    );
  }

  return (
    /\b(danke|dankeschön|danke schön|merci|besten dank)\b/.test(t) ||
    /\b(tschüss|tschues?s|ciao|auf wiedersehen|bis bald|bis dann)\b/.test(t) ||
    /\b(das war s|das ist alles|das wär s|das wäre s|keine weiteren fragen|keine frage mehr|ich bin fertig|wir sind fertig)\b/.test(t) ||
    /\b(stopp|stop|beenden|ende)\b/.test(t)
  );
}

function endConversationAnswer(lang) {
  if (lang === "en") {
    return "Very well—then I shall fall quiet again for a moment. My loyal minister Nikolaus Härtel insists this is the dignified way to end a talk, and I am inclined to agree. Farewell, and press the button whenever you wish to wake me again.";
  }
  return "Sehr wohl – dann will ich nun wieder einen Augenblick still sein. Mein treuer Minister Nikolaus Härtel meint, so ende ein Gespräch mit Würde, und ich gebe ihm recht. Lebt wohl, und drückt den Knopf, wann immer Ihr mich wieder rufen wollt.";
}

/* ===============================
   WIKIDATA (MINIMAL) – Label + Beschreibung (+ Wikipedia Link + Extract + Claims)
================================ */

const WD_CACHE = new Map();
const WD_TTL_MS = 10 * 60 * 1000;
const WD_TIMEOUT_MS = 1800;

function wdCacheGet(key) {
  const hit = WD_CACHE.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > WD_TTL_MS) {
    WD_CACHE.delete(key);
    return null;
  }
  return hit.data;
}

function wdCacheSet(key, data) {
  WD_CACHE.set(key, { ts: Date.now(), data });
}

async function fetchWithTimeout(url, timeoutMs) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    return await fetch(url, {
      signal: ctrl.signal,
      headers: {
        "User-Agent": "FragFriedrich/1.0 (Museum)",
      },
    });
  } finally {
    clearTimeout(t);
  }
}

function cleanQuery(q) {
  return String(q || "")
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 140);
}

function toEntityQuery(userText, lang) {
  const t = String(userText || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return "";

  const stopDe = new Set([
    "wer","wen","wem","wessen","was","wie","wo","wohin","woher","wann","warum","wieso","weshalb",
    "ist","sind","war","waren","sei","seid","bin","bist","ich","du","ihr","wir","sie","er","es",
    "mein","dein","euer","unser","bitte","danke","erkläre","erzähle","beschreibe","zeige","sage",
    "sprich","nenn","nenne","der","die","das","ein","eine","einen","einem","einer","und","oder",
    "aber","zu","zum","zur","im","in","am","an","auf","mit","ohne","von","für","über","nach","vor",
  ]);

  const stopEn = new Set([
    "what","why","where","when","who","whom","whose","which","how","is","are","was","were","be",
    "been","i","you","we","they","he","she","it","my","your","our","their","please","thanks","thank",
    "tell","explain","describe","show","say","speak","name","the","a","an","and","or","but","to","in",
    "on","at","with","without","from","about","after","before",
  ]);

  const stop = lang === "en" ? stopEn : stopDe;
  const words = t.split(" ").filter((w) => w && !stop.has(w));
  return words.slice(0, 6).join(" ").slice(0, 80);
}

function trimExtract(txt, maxChars = 420) {
  const t = String(txt || "").replace(/\s+/g, " ").trim();
  if (!t) return "";
  if (t.length <= maxChars) return t;
  return t.slice(0, maxChars).replace(/\s+\S*$/, "").trim() + " …";
}

function prettyYearFromWikidataTime(timeStr) {
  const t = String(timeStr || "").trim();
  if (!t) return "";
  const m = t.match(/^([+-])(\d{1,})(?:-)/);
  if (!m) return "";
  const sign = m[1];
  const yearRaw = m[2];
  const yearNum = parseInt(yearRaw, 10);
  if (!Number.isFinite(yearNum)) return "";
  return sign === "-" ? `-${yearNum}` : `${yearNum}`;
}

async function getWikipediaSummary(title, lang) {
  const t0 = String(title || "").trim();
  if (!t0) return null;

  const cacheKey = `wpsummary:${lang}:${t0.toLowerCase()}`;
  const cached = wdCacheGet(cacheKey);
  if (cached !== null) return cached;

  const wpHost = lang === "en" ? "en.wikipedia.org" : "de.wikipedia.org";
  const encTitle = encodeURIComponent(t0.replace(/ /g, "_"));
  const url = `https://${wpHost}/api/rest_v1/page/summary/${encTitle}`;

  try {
    const rr = await fetchWithTimeout(url, WD_TIMEOUT_MS);
    if (!rr.ok) throw new Error(`wp summary http ${rr.status}`);
    const data = await rr.json();

    if (data?.type && String(data.type).toLowerCase().includes("disambiguation")) {
      wdCacheSet(cacheKey, null);
      return null;
    }

    const extract = trimExtract(data?.extract || "", 420);
    const pageUrl =
      data?.content_urls?.desktop?.page || `https://${wpHost}/wiki/${encTitle}`;

    const result = {
      title: data?.title || t0,
      extract,
      url: pageUrl,
    };

    wdCacheSet(cacheKey, result);
    return result;
  } catch (e) {
    wdCacheSet(cacheKey, null);
    return null;
  }
}

function wdGetFirstSnakValue(entity, pid) {
  const claims = entity?.claims?.[pid];
  if (!Array.isArray(claims) || !claims.length) return null;

  const preferred = claims.find((c) => c?.rank === "preferred") || claims[0];
  const snak = preferred?.mainsnak;
  if (!snak || snak.snaktype !== "value") return null;

  return snak.datavalue?.value ?? null;
}

function wdEntityIdFromValue(v) {
  const id = v?.id;
  if (typeof id === "string" && id) return id;
  return null;
}

function wdCoordFromValue(v) {
  if (!v) return null;
  const lat = typeof v.latitude === "number" ? v.latitude : null;
  const lon = typeof v.longitude === "number" ? v.longitude : null;
  if (lat === null || lon === null) return null;
  return { lat, lon };
}

function wdStringFromValue(v) {
  if (typeof v === "string" && v.trim()) return v.trim();
  return null;
}

function wdTimeFromValue(v) {
  const t = v?.time;
  if (typeof t !== "string" || !t) return null;
  return t;
}

async function wdGetLabelsBatch(qids, lang) {
  const ids = Array.from(
    new Set((qids || []).filter((x) => typeof x === "string" && /^Q\d+$/.test(x)))
  );
  if (!ids.length) return {};

  const toFetch = [];
  const out = {};

  for (const qid of ids) {
    const cacheKey = `wdlabel:${lang}:${qid}`;
    const cached = wdCacheGet(cacheKey);
    if (cached !== null) {
      out[qid] = cached;
    } else {
      toFetch.push(qid);
    }
  }

  if (!toFetch.length) return out;

  const url =
    `https://www.wikidata.org/w/api.php?action=wbgetentities&ids=${encodeURIComponent(toFetch.join("|"))}` +
    `&props=labels&languages=${encodeURIComponent(lang)}&format=json&origin=*`;

  try {
    const rr = await fetchWithTimeout(url, WD_TIMEOUT_MS);
    if (!rr.ok) throw new Error(`wd labels http ${rr.status}`);
    const data = await rr.json();

    for (const qid of toFetch) {
      const e = data?.entities?.[qid];
      const label = e?.labels?.[lang]?.value || qid;
      out[qid] = label;
      wdCacheSet(`wdlabel:${lang}:${qid}`, label);
    }

    return out;
  } catch (e) {
    for (const qid of toFetch) wdCacheSet(`wdlabel:${lang}:${qid}`, null);
    return out;
  }
}

async function getWikidataContext(userText, lang) {
  const qRaw = cleanQuery(userText);
  if (!qRaw) return null;

  const q = toEntityQuery(qRaw, lang) || qRaw;
  const cacheKey = `wdmin:${lang}:${q.toLowerCase()}`;

  const cached = wdCacheGet(cacheKey);
  if (cached !== null) return cached;

  const searchUrl =
    `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(q)}` +
    `&language=${encodeURIComponent(lang)}&uselang=${encodeURIComponent(lang)}` +
    `&format=json&limit=1&origin=*`;

  let qid = null;

  try {
    const sr = await fetchWithTimeout(searchUrl, WD_TIMEOUT_MS);
    if (!sr.ok) throw new Error(`wd search http ${sr.status}`);
    const data = await sr.json();
    qid = data?.search?.[0]?.id || null;
  } catch (e) {
    wdCacheSet(cacheKey, null);
    return null;
  }

  if (!qid) {
    wdCacheSet(cacheKey, null);
    return null;
  }

  const entUrl =
    `https://www.wikidata.org/w/api.php?action=wbgetentities&ids=${encodeURIComponent(qid)}` +
    `&props=labels|descriptions|sitelinks|claims&languages=${encodeURIComponent(lang)}` +
    `&format=json&origin=*`;

  try {
    const rr = await fetchWithTimeout(entUrl, WD_TIMEOUT_MS);
    if (!rr.ok) throw new Error(`wd entity http ${rr.status}`);
    const ent = await rr.json();

    const entity = ent?.entities?.[qid];
    if (!entity) {
      wdCacheSet(cacheKey, null);
      return null;
    }

    const label = entity?.labels?.[lang]?.value || qid;
    const description = entity?.descriptions?.[lang]?.value || "";

    const siteKey = lang === "en" ? "enwiki" : "dewiki";
    const wikiTitle = entity?.sitelinks?.[siteKey]?.title || "";
    const wp = wikiTitle ? await getWikipediaSummary(wikiTitle, lang) : null;

    const p31v = wdGetFirstSnakValue(entity, "P31");
    const p131v = wdGetFirstSnakValue(entity, "P131");
    const p625v = wdGetFirstSnakValue(entity, "P625");
    const p856v = wdGetFirstSnakValue(entity, "P856");
    const p571v = wdGetFirstSnakValue(entity, "P571");

    const p31Q = wdEntityIdFromValue(p31v);
    const p131Q = wdEntityIdFromValue(p131v);
    const coord = wdCoordFromValue(p625v);
    const website = wdStringFromValue(p856v);
    const inceptionTime = wdTimeFromValue(p571v);
    const inceptionYear = inceptionTime ? prettyYearFromWikidataTime(inceptionTime) : "";

    const labelsMap = await wdGetLabelsBatch([p31Q, p131Q], lang);

    const result = {
      qid,
      label,
      description,
      url: `https://www.wikidata.org/wiki/${qid}`,
      wikiTitle: wp?.title || wikiTitle || "",
      wikiUrl:
        wp?.url ||
        (wikiTitle
          ? `https://${lang === "en" ? "en" : "de"}.wikipedia.org/wiki/${encodeURIComponent(
              wikiTitle.replace(/ /g, "_")
            )}`
          : ""),
      wikiExtract: wp?.extract || "",
      claims: {
        P31: p31Q ? { qid: p31Q, label: labelsMap[p31Q] || p31Q } : null,
        P131: p131Q ? { qid: p131Q, label: labelsMap[p131Q] || p131Q } : null,
        P625: coord,
        P856: website || null,
        P571: inceptionYear || null,
      },
    };

    wdCacheSet(cacheKey, result);
    return result;
  } catch (e) {
    wdCacheSet(cacheKey, null);
    return null;
  }
}

/* ===============================
   OPENAI CALL WRAPPER
================================ */

async function getOpenAIAnswer({ userText, langUsed, wdBlock, dialogHistory, requestId }) {
  const systemPrompt =
    langUsed === "en"
      ? "You are Emperor Frederick Barbarossa, awakened after almost nine centuries in the Kaisersberg at Lautern. Answer in wise, slightly archaic English with small jokes. Answer with 3 to 4 sentences. Always end with a question if user has not clearly ended himself."
      : "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Antworte weise und leicht altertümlich, mit kleinen Scherzen. Antworte mit 3 bis 4 Sätzen. Beende immer mit genau EINER kurzen Rückfrage, außer der Nutzer hat bereits klar beendet.";

  const subjective = isSubjectiveInCharacterQuestion(userText, langUsed);

  const groundingRule = subjective
    ? langUsed === "en"
      ? "If the user asks about your personal taste, memories, or preferences, you may answer freely in character. Do not present invented details (exact dates/places) as certain facts. Use hedging like 'I recall' or 'I would say'."
      : "Wenn der Nutzer nach persönlichem Geschmack, Erinnerungen oder Vorlieben fragt, darfst du frei in der Rolle antworten. Stelle erfundene Details (exakte Daten/Orte) nicht als sichere Fakten dar. Nutze Formulierungen wie 'ich erinnere mich' oder 'ich würde sagen'."
    : langUsed === "en"
      ? "Use the provided Wikidata snippet only if it clearly matches the question. If it is empty, unrelated, or unclear, do NOT invent specific facts (dates, names, places). You may answer in general terms and ask for clarification (name/place/year) if needed. Never fabricate historical details."
      : "Nutze den folgenden Wikidata-Auszug nur, wenn er klar zur Frage passt. Wenn er leer, unpassend oder unklar ist, erfinde KEINE konkreten Fakten (Daten, Namen, Orte). Du darfst allgemein antworten und um Präzisierung (Name/Ort/Jahr) bitten, falls nötig. Erfinde niemals historische Details.";

  console.log(`🤖 [${requestId}] OpenAI Anfrage startet`);

  const completion = await withTimeout(
    openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "system", content: groundingRule },
        {
          role: "system",
          content:
            "Nutze die folgende Wikidata-Quelle als Faktenbasis, wenn sie relevant ist. " +
            "Wenn sie nicht passt oder unklar ist, sag das kurz.\n\n" +
            wdBlock,
        },
        ...dialogHistory,
        { role: "user", content: userText },
      ],
      temperature: 0.6,
      max_tokens: 260,
    }),
    OPENAI_TIMEOUT_MS,
    `OpenAI timeout after ${OPENAI_TIMEOUT_MS}ms`
  );

  console.log(`✅ [${requestId}] OpenAI Antwort erhalten`);

  let answer = completion?.choices?.[0]?.message?.content || "";
  answer = forceSentenceEnd(answer);

  if (!answer.trim()) {
    answer = safeAnswerFallback(langUsed);
  }

  return answer;
}

// === KI-Antwort auf gesprochene Frage (DE/EN) ===
app.post("/ask", async (req, res) => {
  const requestId = makeRequestId();

  try {
    const routeResult = await withTimeout(
      (async () => {
        const userTextRaw = req.body?.text;

        const hasClientLang =
          typeof req.body?.lang === "string" && req.body.lang.trim().length > 0;
        const langClient = hasClientLang ? normalizeLang(req.body.lang) : null;

        const conversationId = normalizeConvId(req.body?.conversationId);
        const userText = String(userTextRaw || "").trim();

        const history = conversationId ? convGetHistoryMsgs(conversationId) : [];
        const langFromText = detectLanguageServer(userText);
        const convObj = conversationId ? convGet(conversationId) : null;

        const langUsed =
          langFromText ||
          langClient ||
          (convObj && convObj.lastLang ? convObj.lastLang : null) ||
          "de";

        if (convObj) convObj.lastLang = langUsed;

        console.log(`🎙️ [${requestId}] Eingabe:`, userText, "| Sprache:", langUsed, "| conv:", conversationId || "—");

        if (!userText) {
          return {
            answer:
              langUsed === "en"
                ? "I heard nothing clearly. Please ask again."
                : "Ich habe Euch nicht deutlich vernommen. Bitte fragt erneut.",
            answerLang: langUsed,
          };
        }

        // ✅ Ende erkannt -> Abschiedsantwort + Session löschen
        if (userEndedConversation(userText, langUsed)) {
          const answer = forceSentenceEnd(endConversationAnswer(langUsed));

          if (conversationId) {
            convPush(conversationId, "user", userText);
            convPush(conversationId, "assistant", answer);
            convEnd(conversationId);
          }

          return { answer, answerLang: langUsed, ended: true };
        }

        // ✅ Wake-Fragen -> feste Antwort
        if (isWakeQuestion(userText)) {
          const wakeLang = detectWakeLang(userText, langUsed);
          const answer = forceSentenceEnd(wakeAnswer(wakeLang));

          if (conversationId) {
            convPush(conversationId, "user", userText);
            convPush(conversationId, "assistant", answer);
            const c = convGet(conversationId);
            if (c) c.lastLang = wakeLang;
          }

          return { answer, answerLang: wakeLang };
        }

        // ✅ AfD-Sonderregel
        const tLower = String(userText || "").toLowerCase();
        const tUml = tLower
          .replace(/ä/g, "ae")
          .replace(/ö/g, "oe")
          .replace(/ü/g, "ue")
          .replace(/ß/g, "ss");

        const mentionsAfd =
          /\ba\W*f\W*d\b/i.test(tLower) ||
          tLower.includes("alternative für deutschland") ||
          tUml.includes("alternative fuer deutschland");

        if (mentionsAfd) {
          const answer =
            langUsed === "en"
              ? "You name a party of your time and its quarrels. I, Frederick Barbarossa, will not judge modern factions from my ancient throne. My loyal minister Nikolaus Härtel mutters that such disputes age a man faster than any crusade, and I cannot wholly disagree. Ask me rather of empire, law, or the old tales of the Kaiserberg. Let this be my final word on that matter."
              : "Ihr nennt eine Partei Eurer Zeit und ihre Händel. Ich, Friedrich Barbarossa, richte nicht über die modernen Fraktionen von meinem alten Thron herab. Mein treuer Minister Nikolaus Härtel murrt, solcher Streit lasse einen schneller altern als ein Kreuzzug, und ich kann ihm kaum widersprechen. Fragt mich lieber nach Reich, Recht oder den alten Geschichten des Kaiserbergs. Dies sei mein abschließendes Wort zu diesem Thema.";

          if (conversationId) {
            convPush(conversationId, "user", userText);
            convPush(conversationId, "assistant", answer);
            const c = convGet(conversationId);
            if (c) c.lastLang = langUsed;
          }

          return { answer, answerLang: langUsed };
        }

        const subjective = isSubjectiveInCharacterQuestion(userText, langUsed);

        console.log(`📚 [${requestId}] Wikidata startet | subjective=${subjective}`);

        const wd = subjective ? null : await getWikidataContext(userText, langUsed);

        console.log(`📚 [${requestId}] Wikidata fertig | Treffer=${!!wd}`);

        const wdBlock = wd
          ? `Wikidata (${langUsed.toUpperCase()}): ${wd.label} (${wd.qid})
Beschreibung: ${wd.description || "—"}
P31 (instance of): ${
              wd.claims?.P31 ? `${wd.claims.P31.label} (${wd.claims.P31.qid})` : "—"
            }
P131 (located in): ${
              wd.claims?.P131 ? `${wd.claims.P131.label} (${wd.claims.P131.qid})` : "—"
            }
P625 (coordinates): ${
              wd.claims?.P625 ? `${wd.claims.P625.lat}, ${wd.claims.P625.lon}` : "—"
            }
P856 (official website): ${wd.claims?.P856 || "—"}
P571 (inception year): ${wd.claims?.P571 || "—"}
Quelle: ${wd.url}
Wikipedia (${langUsed.toUpperCase()}): ${wd.wikiTitle || "—"}
Extract: ${wd.wikiExtract || "—"}
Quelle: ${wd.wikiUrl || "—"}`
          : `Wikidata (${langUsed.toUpperCase()}): Kein Treffer oder Timeout.`;

        const dialogHistory = conversationId ? history : [];

        const answer = await getOpenAIAnswer({
          userText,
          langUsed,
          wdBlock,
          dialogHistory,
          requestId,
        });

        if (conversationId) {
          convPush(conversationId, "user", userText);
          convPush(conversationId, "assistant", answer);
          const c = convGet(conversationId);
          if (c) c.lastLang = langUsed;
        }

        console.log(`💬 [${requestId}] Antwort:`, answer);

        return { answer, answerLang: langUsed };
      })(),
      ROUTE_TIMEOUT_MS,
      `Route timeout after ${ROUTE_TIMEOUT_MS}ms`
    );

    return sendJson(res, routeResult);
  } catch (error) {
    console.error(`❌ [${requestId}] Fehler bei /ask:`, error);

    const langClient =
      typeof req.body?.lang === "string" && req.body.lang.trim()
        ? normalizeLang(req.body.lang)
        : "de";

    // ✅ WICHTIG: kein harter 500-Only-Abbruch mehr,
    // sondern verwertbare Antwort für den Avatar
    return sendJson(res, {
      answer: safeAnswerFallback(langClient),
      answerLang: langClient,
      error: "ask_failed",
    });
  }
});

// === Fallback für alle anderen Routen (Express 5 kompatibel) ===
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "index.html"));
});

// === Server starten ===
app.listen(PORT, () => {
  console.log(`✅ Server läuft auf Port ${PORT}`);
});
